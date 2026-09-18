#!/usr/bin/env python3
"""
Builds the few-shot block set for the MOCS annotation pass, from REAL rows of the
ConstructionSite train split.

WHY REAL ROWS AND NOT HAND-WRITTEN ONES. The few-shot blocks carry no images, so
they cannot teach the teacher what a violation looks like -- their entire job is
format and REGISTER conditioning. Register is the part that matters: README_v2.md
§11.3 measures that the BERTScore jump from 0.46 to 0.75 is substantially "learning
the dataset's phrasing", and that 56-64% of the model's explanations are byte-
identical strings. If the examples are written in a new voice, the teacher writes in
that voice, the harvested rows teach the student that voice, and the student is then
scored against the ORIGINAL test split's voice. Copying real strings out of the
train split costs nothing and removes the whole problem.

Composition, and why:
  * 2 examples each for rule_2, rule_3, rule_4 -- the three rules being harvested.
  * 1 all-null example, so the teacher is not primed to always find something.
  * rule_1 is deliberately NOT given an example. It is already the dominant class
    (10.75% of images vs 0.8-2.1% for the rare rules) and data/prompt_templates.py
    documents removing an inline rule_1 example for exactly this reason: "it pushed
    the model toward the very over-flagging the rest of the prompt works against."

Selection is deterministic: within each rule, the candidate whose reason length is
closest to that rule's median and whose caption length is closest to the split
median wins, ties broken by image_id. Re-running gives the same blocks.

Runs on ARC (needs the dataset). No GPU.

Usage:
    python -m mocs_annotation.build_fewshot \
        --out $VLM_DATA_ROOT/datasets/mocs_annotation/fewshot.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES
from core.logging import get_logger
from data.box_utils import clean_boxes, normalize_boxes, scale_01_to_1000

logger = get_logger(__name__)

TARGET_RULES = ("rule_2", "rule_3", "rule_4")
PER_RULE = 2

# Authored, NOT measured. Shown in two blocks only, to demonstrate that the
# confidence field accepts intermediate values -- if every example were 0.95/0.05
# the teacher would learn to emit a binary flag, which is useless for ordering a
# review queue. Tagged so nobody later mistakes these for calibrated numbers.
_AUTHORED_CONFIDENCE = {
    0: {"rule_1": 0.31, "rule_2": 0.78, "rule_3": 0.09, "rule_4": 0.06},
    1: {"rule_1": 0.12, "rule_2": 0.05, "rule_3": 0.44, "rule_4": 0.08},
}


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


def build(split, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cols = [c for c in split.column_names if c != "image"]
    rows = split.select_columns(cols)
    if limit:
        rows = rows.select(range(min(limit, len(rows))))

    materialised: List[Dict[str, Any]] = list(rows)
    logger.info(f"Scanning {len(materialised)} train rows for few-shot candidates...")

    caption_median = st.median([_words(r.get("image_caption")) for r in materialised]) or 55
    reason_median: Dict[str, float] = {}
    for rule in TARGET_RULES:
        lens = [
            _words(v.get("reason"))
            for r in materialised
            if (v := _row_violation(r, rule)) and (v.get("reason") or "").strip()
        ]
        reason_median[rule] = st.median(lens) if lens else 12.0
    logger.info(f"Median caption words: {caption_median}; median reason words: {reason_median}")

    used_ids: set = set()
    examples: List[Dict[str, Any]] = []

    for rule in TARGET_RULES:
        scored = []
        for row in materialised:
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
            scored.append((score, image_id, row, v))

        scored.sort(key=lambda t: (t[0], t[1]))
        if len(scored) < PER_RULE:
            logger.warning(
                f"{rule}: only {len(scored)} usable candidate(s) found (wanted {PER_RULE})"
            )

        for _, image_id, row, v in scored[:PER_RULE]:
            used_ids.add(image_id)
            payload = _blank_rules()
            payload[f"{rule}_violation"] = _to_example_violation(v)
            # Any OTHER rule this image genuinely violates is included too -- a
            # multi-rule example is honest and demonstrates that rules are
            # independent. Zeroing them would teach the teacher that finding one
            # rule means the others are null.
            for other in RULES:
                if other == rule:
                    continue
                ov = _row_violation(row, other)
                if ov and (ov.get("reason") or "").strip():
                    payload[f"{other}_violation"] = _to_example_violation(ov)
            examples.append({
                "label": f"{rule} violated",
                "source": f"train/{image_id}",
                "caption": (row.get("image_caption") or "").strip(),
                **payload,
            })

    # One all-null example so the teacher is not primed to always report something.
    safe_scored = []
    for row in materialised:
        image_id = str(row.get("image_id", ""))
        if image_id in used_ids:
            continue
        if any(_row_violation(row, r) for r in RULES):
            continue
        caption = (row.get("image_caption") or "").strip()
        if not caption:
            continue
        safe_scored.append((abs(_words(caption) - caption_median), image_id, row))
    safe_scored.sort(key=lambda t: (t[0], t[1]))
    if safe_scored:
        _, image_id, row = safe_scored[0]
        examples.append({
            "label": "no violation",
            "source": f"train/{image_id}",
            "caption": (row.get("image_caption") or "").strip(),
            **_blank_rules(),
        })
    else:
        logger.warning("No all-null candidate found -- the negative example is missing.")

    # Attach the authored confidence demo to two blocks only.
    for idx, conf in _AUTHORED_CONFIDENCE.items():
        if idx < len(examples):
            examples[idx]["confidence"] = conf
            examples[idx]["confidence_provenance"] = "authored (format demo, not measured)"

    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Path to write fewshot.json")
    ap.add_argument(
        "--subdir", default=None,
        help="Dataset subdir to read. Defaults to base.yaml's raw_processed_subdir "
             "(datasets/processed) -- the UN-augmented base, so a few-shot example is "
             "never one of the 16x pixel-augmented duplicates.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Scan only the first N rows (debug)")
    args = ap.parse_args()

    from core.config import load_base_config
    from data.loader import load_processed_dataset

    subdir = args.subdir or load_base_config()["dataset"].get(
        "raw_processed_subdir", "datasets/processed"
    )
    logger.info(f"Reading train split from {subdir}")
    splits = load_processed_dataset(subdir=subdir)
    examples = build(splits["train"], limit=args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)

    logger.info(f"Wrote {len(examples)} few-shot block(s) to {out}")
    for ex in examples:
        flagged = [r for r in RULES if ex.get(f"{r}_violation")]
        logger.info(
            f"  {ex['label']:<18} {ex['source']:<22} caption={_words(ex['caption']):>3}w "
            f"rules={flagged or ['none']}"
        )


if __name__ == "__main__":
    main()
