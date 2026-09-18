#!/usr/bin/env python3
"""
Builds the few-shot example set for the MOCS annotation pass, from REAL rows of the
ConstructionSite train split -- images included.

COMPOSITION (locked 2026-09-18): FIVE blocks, one per rule plus one all-null, each
carrying its real image.

    rule_1 violated  |  rule_2 violated  |  no violation  |  rule_3 violated  |  rule_4 violated

Every rule gets exactly one example, including rule_1. An earlier version left rule_1
out on the grounds that data/prompt_templates.py removed an inline rule_1 example to
avoid priming over-flagging -- but that reasoning belongs to the TRAINING prompt,
where over-flagging is the measured failure and nothing filters it. Here a human
reviews every row, and the real hazard runs the other way: if the teacher
under-reports rule_1, a reviewer working a row for rule_4 can accept it without
noticing rule_1 was missed, which injects a false negative into the strongest rule in
the project (0.82 precision). One block out of five also leaves rule_1 deliberately
UNDER-represented relative to its true 10.75% prevalence, which keeps the prior shift
small.

The all-null block sits in the MIDDLE rather than last. A hedge against a recency
effect, not a measured one: ending on a violation nudges toward flagging, ending on
null nudges toward abstaining, and this task wants neither pushed by example order.

WHY REAL ROWS. The blocks carry the register the student is ultimately scored
against. README_v2.md §11.3 measures that the BERTScore jump 0.46 -> 0.75 is
substantially "learning the dataset's phrasing", and that 56-64% of the model's
explanations are byte-identical strings. Hand-written examples would put the teacher
in a new voice, the harvest would teach the student that voice, and the student would
then be scored against the ORIGINAL test split's voice. Copying real strings costs
nothing.

Selection is deterministic: within each rule, the candidate whose reason length is
closest to that rule's median and whose caption length is closest to the split median
wins, ties broken by image_id. Re-running produces the same five blocks.

Outputs (both required by annotate.py):
    <out>                      fewshot.json -- the blocks, boxes already 0-1000
    <out>.parent/fewshot_images/  the five example images, downscaled to the same
                                  1.2 MP ceiling the annotator applies at load time,
                                  so the vision-token cost per block is predictable

Runs on ARC (needs the dataset). No GPU.

Usage:
    python -m mocs_annotation.build_fewshot --out $OUT/fewshot.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES
from core.logging import get_logger
from data.box_utils import clean_boxes, normalize_boxes, scale_01_to_1000

logger = get_logger(__name__)

# One example per rule. All four rules, one each -- see the module docstring.
PER_RULE = 1

# Block order. The all-null example is third of five on purpose.
BLOCK_ORDER = ("rule_1", "rule_2", "__safe__", "rule_3", "rule_4")

# Same 1.2 MP ceiling configs/sft.yaml applies everywhere in this repo. Applied when
# SAVING the example images so each block costs a predictable ~1,176 vision tokens
# (1204224 px / (patch 16^2 x merge 2^2 = 1024 px per token)).
MAX_PIXELS = 1204224

IMAGES_SUBDIR = "fewshot_images"


def _words(text: Optional[str]) -> int:
    return len(str(text or "").split())


def _row_violation(row: Dict[str, Any], rule: str) -> Optional[Dict[str, Any]]:
    v = row.get(f"{rule}_violation")
    return v if isinstance(v, dict) else None


def _to_example_violation(v: Dict[str, Any]) -> Dict[str, Any]:
    """Ground-truth violation ([0,1] boxes) -> example shape with [0,1000] boxes.

    The teacher is asked for 0-1000, so the examples must demonstrate 0-1000 or the
    format lesson teaches the wrong scale. Same conversion
    data/preprocessor.py::build_target_json applies when building an SFT target.
    """
    boxes = clean_boxes(normalize_boxes(v.get("bounding_box")))
    return {
        "bounding_box": [scale_01_to_1000(b) for b in boxes],
        "reason": (v.get("reason") or "").strip(),
    }


def _blank_rules() -> Dict[str, Any]:
    return {f"{r}_violation": None for r in RULES}


def _downscale(img, max_pixels: int = MAX_PIXELS):
    """Shrinks an image to at most `max_pixels` of area, preserving aspect ratio."""
    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img
    scale = (max_pixels / area) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def select_examples(split, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Picks the five example ROWS (metadata only -- images are fetched afterwards).

    Scans with the `image` column dropped, so no image is ever decoded during the
    scan. That is the difference between this taking seconds and it taking the
    multi-GB, multi-minute hit that data/preprocessor.py::build_sft_dataset pays when
    it materialises every decoded PIL image (the reason SFT runs at --mem=150G).
    """
    cols = [c for c in split.column_names if c != "image"]
    rows = split.select_columns(cols)
    if limit:
        rows = rows.select(range(min(limit, len(rows))))

    materialised: List[Dict[str, Any]] = list(rows)
    logger.info(f"Scanning {len(materialised)} train rows for few-shot candidates...")

    caption_median = st.median([_words(r.get("image_caption")) for r in materialised]) or 55
    reason_median: Dict[str, float] = {}
    for rule in RULES:
        lens = [
            _words(v.get("reason"))
            for r in materialised
            if (v := _row_violation(r, rule)) and (v.get("reason") or "").strip()
        ]
        reason_median[rule] = st.median(lens) if lens else 12.0
    logger.info(f"Median caption words: {caption_median}; median reason words: {reason_median}")

    used_ids: set = set()
    chosen: Dict[str, Dict[str, Any]] = {}

    # Rarest rule first, so a scarce rule is not denied its best candidate by a
    # common rule that happened to select the same image.
    rule_pool_size = {
        rule: sum(1 for r in materialised if _row_violation(r, rule)) for rule in RULES
    }
    for rule in sorted(RULES, key=lambda r: rule_pool_size[r]):
        scored: List[Tuple[float, str, int, Dict[str, Any], Dict[str, Any]]] = []
        for idx, row in enumerate(materialised):
            image_id = str(row.get("image_id", ""))
            if image_id in used_ids:
                continue
            v = _row_violation(row, rule)
            if not v:
                continue
            reason = (v.get("reason") or "").strip()
            boxes = clean_boxes(normalize_boxes(v.get("bounding_box")))
            caption = (row.get("image_caption") or "").strip()
            # A usable example must demonstrate every part of the contract: a real
            # box, a real one-sentence reason, and a real caption.
            if not reason or not boxes or not caption:
                continue
            score = (
                abs(_words(reason) - reason_median[rule])
                + 0.3 * abs(_words(caption) - caption_median)
            )
            scored.append((score, image_id, idx, row, v))

        scored.sort(key=lambda t: (t[0], t[1]))
        if not scored:
            logger.warning(f"{rule}: NO usable candidate found -- that block will be missing")
            continue

        _, image_id, idx, row, v = scored[0]
        used_ids.add(image_id)
        payload = _blank_rules()
        payload[f"{rule}_violation"] = _to_example_violation(v)
        # Any OTHER rule this image genuinely violates is kept too. A multi-rule
        # example is honest and demonstrates that rules are independent; zeroing
        # them would teach the teacher that finding one rule means the rest are null.
        for other in RULES:
            if other == rule:
                continue
            ov = _row_violation(row, other)
            if ov and (ov.get("reason") or "").strip():
                payload[f"{other}_violation"] = _to_example_violation(ov)
        chosen[rule] = {
            "label": f"{rule} violated",
            "source": f"train/{image_id}",
            "row_index": idx,
            "caption": (row.get("image_caption") or "").strip(),
            **payload,
        }

    # The all-null block.
    safe_scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
    for idx, row in enumerate(materialised):
        image_id = str(row.get("image_id", ""))
        if image_id in used_ids:
            continue
        if any(_row_violation(row, r) for r in RULES):
            continue
        caption = (row.get("image_caption") or "").strip()
        if not caption:
            continue
        safe_scored.append((abs(_words(caption) - caption_median), image_id, idx, row))
    safe_scored.sort(key=lambda t: (t[0], t[1]))
    if safe_scored:
        _, image_id, idx, row = safe_scored[0]
        chosen["__safe__"] = {
            "label": "no violation",
            "source": f"train/{image_id}",
            "row_index": idx,
            "caption": (row.get("image_caption") or "").strip(),
            **_blank_rules(),
        }
    else:
        logger.warning("No all-null candidate found -- the negative example is missing.")

    return [chosen[key] for key in BLOCK_ORDER if key in chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Path to write fewshot.json")
    ap.add_argument(
        "--subdir", default=None,
        help="Dataset subdir to read. Defaults to base.yaml's raw_processed_subdir "
             "(datasets/processed) -- the UN-augmented base, so an example is never one "
             "of the 16x pixel-augmented duplicates.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Scan only the first N rows (debug)")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                     help="Area ceiling for the saved example images")
    args = ap.parse_args()

    from core.config import load_base_config
    from data.loader import load_processed_dataset

    subdir = args.subdir or load_base_config()["dataset"].get(
        "raw_processed_subdir", "datasets/processed"
    )
    logger.info(f"Reading train split from {subdir}")
    splits = load_processed_dataset(subdir=subdir)
    train = splits["train"]

    examples = select_examples(train, limit=args.limit)
    if not examples:
        raise SystemExit("No few-shot examples could be selected -- check the dataset.")

    out = Path(args.out)
    images_dir = out.parent / IMAGES_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)

    # Fetch ONLY the selected rows' images -- five decodes, not 6,308.
    logger.info(f"Saving {len(examples)} example image(s) to {images_dir}")
    for ex in examples:
        idx = ex.pop("row_index")
        img = train[idx]["image"].convert("RGB")
        before = img.size
        img = _downscale(img, args.max_pixels)
        name = ex["source"].replace("/", "_") + ".jpg"
        path = images_dir / name
        img.save(path, format="JPEG", quality=92)
        ex["image_file"] = f"{IMAGES_SUBDIR}/{name}"
        logger.info(
            f"  {ex['label']:<18} {ex['source']:<22} {before[0]}x{before[1]} -> "
            f"{img.size[0]}x{img.size[1]}  ({img.size[0] * img.size[1] / 1e6:.2f} MP)  {name}"
        )

    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)

    logger.info("")
    logger.info(f"Wrote {len(examples)} few-shot block(s) to {out}")
    for ex in examples:
        flagged = [r for r in RULES if ex.get(f"{r}_violation")]
        logger.info(
            f"  {ex['label']:<18} {ex['source']:<22} caption={_words(ex['caption']):>3}w "
            f"rules={flagged or ['none']}"
        )
    logger.info("")
    logger.info(
        f"Each block costs ~{args.max_pixels // 1024} vision tokens at load time, so "
        f"{len(examples)} blocks + the query image is "
        f"~{(len(examples) + 1) * (args.max_pixels // 1024)} vision tokens per request. "
        "Size --batch-size against that."
    )


if __name__ == "__main__":
    main()
