#!/usr/bin/env python3
"""
Turns proposals.jsonl into a review queue for the civil engineer, and reports yield.

Two outputs:

  review.csv   one row per image, ordered for triage, with empty `verify_*` columns
               for the reviewer to fill in. Opens in Excel / LibreOffice.
  a console yield table  proposals per rule, split by selection bucket. This is the
               number that says whether the geometric mining was worth anything:
               compare rule_4 yield in `rule_4_geometric` against `random_control`.
               If they match, the miner is doing nothing and the quotas should change.

TWO THINGS THE REVIEWER MUST BE TOLD, both easy to get wrong:

  1. VERIFY ALL FOUR RULES, not just the rare ones. MOCS has no hard-hat category and
     rule_1 is ~10.75% prevalent in ConstructionSite, so many of these images
     genuinely do violate rule_1. Accepting a row with rule_1 left null injects a
     false negative into the strongest rule in the whole project (0.82 precision).
  2. DO NOT TRUST THE TEACHER'S BOXES. A zero-shot VLM's violation IoU is ~23% at
     best in the literature against this repo's own 45.6%. For rule_4 the CSV carries
     `mocs_suggested_box_1000`, derived from MOCS's HUMAN-annotated worker and
     machine boxes -- prefer that. For rule_2 and rule_3 the box has to be drawn by
     hand.

Runs anywhere. No GPU, no dataset, no model.

Usage:
    python -m mocs_annotation.export_review \
        --proposals $OUT/proposals.jsonl \
        --selection $OUT/selection.json \
        --out       $OUT/review.csv \
        --rules rule_2 rule_3 rule_4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES
from core.logging import get_logger
from mocs_annotation.schema import STATUS_OK

logger = get_logger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                logger.warning(f"{path.name}:{lineno} unparseable -- skipped")
    return out


def _boxes_to_1000(boxes: Optional[Sequence[Sequence[float]]]) -> str:
    """Records store boxes in dataset [0,1] scale; show them 0-1000 so the reviewer
    reads the same numbers the model was asked to produce."""
    if not boxes:
        return ""
    return "; ".join(
        "[" + ", ".join(str(int(round(c * 1000))) for c in box) + "]" for box in boxes
    )


def _suggested_rule4_box(record: Dict[str, Any]) -> str:
    pairs = record.get("worker_machine_pairs") or []
    if not pairs:
        return ""
    return "; ".join(
        "[" + ", ".join(str(int(round(c * 1000))) for c in p["union_box"]) + "]"
        for p in pairs[:4]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals", required=True, help="proposals.jsonl from annotate.py")
    ap.add_argument("--selection", default=None,
                     help="selection.json, to recover the images root and MOCS geometry")
    ap.add_argument("--out", required=True, help="Path to write review.csv")
    ap.add_argument("--rules", nargs="+", default=["rule_2", "rule_3", "rule_4"],
                     help="Only queue images where at least one of these was proposed. "
                          "Pass all four to review everything")
    ap.add_argument("--include-empty", action="store_true",
                     help="Also queue images where nothing was proposed. Worth sampling: it "
                          "is the only way to estimate the teacher's FALSE NEGATIVE rate, "
                          "which one-sided verification otherwise hides completely")
    ap.add_argument("--images-root", default=None)
    args = ap.parse_args()

    proposals = _load_jsonl(Path(args.proposals))
    logger.info(f"Loaded {len(proposals)} proposal record(s)")

    images_root = args.images_root
    geometry: Dict[str, Dict[str, Any]] = {}
    if args.selection:
        with open(args.selection, "r", encoding="utf-8") as f:
            sel = json.load(f)
        images_root = images_root or sel["manifest"].get("images_root")
        geometry = {r["new_image_id"]: r for r in sel["records"]}

    wanted = list(args.rules)

    # ---- yield table: the whole point of keeping a random_control bucket ----
    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    bucket_totals: Counter = Counter()
    status_counts: Counter = Counter()
    for rec in proposals:
        status_counts[rec.get("status", "?")] += 1
        bucket = rec.get("selection_bucket", "?")
        bucket_totals[bucket] += 1
        if rec.get("status") != STATUS_OK:
            continue
        for r in RULES:
            if rec.get(f"{r}_violation") is not None:
                by_bucket[bucket][r] += 1

    logger.info("Status counts: " + ", ".join(f"{k}={v}" for k, v in status_counts.most_common()))
    logger.info("")
    logger.info("PROPOSAL YIELD BY BUCKET (these are proposals, not verified labels)")
    header = f"  {'bucket':<26}{'images':>8}" + "".join(f"{r:>10}" for r in RULES)
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for bucket in sorted(bucket_totals):
        n = bucket_totals[bucket]
        cells = "".join(
            f"{by_bucket[bucket][r]:>6} {100 * by_bucket[bucket][r] / n:>3.0f}%" if n else f"{'-':>10}"
            for r in RULES
        )
        logger.info(f"  {bucket:<26}{n:>8}{cells}")
    if "random_control" in bucket_totals:
        logger.info("")
        logger.info(
            "  Read the mined buckets against random_control. If rule_4 yield in "
            "rule_4_geometric is not clearly above random_control, the geometric miner "
            "is adding nothing and the quotas should be rebalanced."
        )

    # ---- review queue ----
    queue: List[Dict[str, Any]] = []
    for rec in proposals:
        if rec.get("status") != STATUS_OK:
            continue
        flagged = [r for r in RULES if rec.get(f"{r}_violation") is not None]
        hit = [r for r in flagged if r in wanted]
        if not hit and not args.include_empty:
            continue
        conf = rec.get("confidence") or {}
        # Triage order: highest self-reported confidence on a wanted rule first, so
        # the reviewer's first hour is spent where the acceptance rate is highest.
        priority = max((float(conf.get(r, 0.0)) for r in wanted), default=0.0)
        queue.append({"_rec": rec, "_priority": priority, "_hit": hit})

    queue.sort(key=lambda q: (-len(q["_hit"]), -q["_priority"], q["_rec"]["new_image_id"]))

    fieldnames = [
        "new_image_id", "file_name", "image_path", "selection_bucket",
        "proposed_rules", "max_confidence", "caption",
    ]
    for r in RULES:
        fieldnames += [f"{r}_proposed", f"{r}_reason", f"{r}_boxes_1000", f"{r}_confidence"]
    fieldnames += [
        "mocs_categories", "mocs_suggested_rule4_box_1000",
        # Reviewer fills these in. Left empty on purpose.
        "verify_decision", "verify_rule_1", "verify_rule_2", "verify_rule_3", "verify_rule_4",
        "verify_caption_ok", "verify_corrected_reason", "verify_corrected_boxes_1000",
        "verify_notes",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in queue:
            rec = item["_rec"]
            row: Dict[str, Any] = {
                "new_image_id": rec["new_image_id"],
                "file_name": rec["file_name"],
                "image_path": str(Path(images_root) / rec["file_name"]) if images_root else "",
                "selection_bucket": rec.get("selection_bucket", ""),
                "proposed_rules": " ".join(item["_hit"]),
                "max_confidence": f"{item['_priority']:.2f}",
                "caption": rec.get("image_caption", ""),
                "mocs_categories": " ".join(rec.get("mocs_categories") or []),
                "mocs_suggested_rule4_box_1000": _suggested_rule4_box(
                    geometry.get(rec["new_image_id"], rec)
                ),
            }
            conf = rec.get("confidence") or {}
            for r in RULES:
                v = rec.get(f"{r}_violation")
                row[f"{r}_proposed"] = 1 if v is not None else 0
                row[f"{r}_reason"] = (v or {}).get("reason", "") if v else ""
                row[f"{r}_boxes_1000"] = _boxes_to_1000((v or {}).get("bounding_box")) if v else ""
                row[f"{r}_confidence"] = f"{float(conf.get(r, 0.0)):.2f}" if conf else ""
            writer.writerow(row)

    logger.info("")
    logger.info(f"Wrote {len(queue)} review row(s) to {out_path}")
    logger.info(
        "Reviewer instructions: confirm ALL FOUR rules on every row you accept (a null "
        "rule_1 on an image that really violates it is a false negative in the project's "
        "strongest rule), and prefer mocs_suggested_rule4_box_1000 over the model's own "
        "box -- it comes from human annotations."
    )
    if not args.include_empty:
        logger.info(
            "Also worth doing once: re-run with --include-empty and hand-check ~40 rows "
            "where nothing was proposed. That is the only measurement of what the teacher "
            "MISSED, and one-sided verification cannot see it."
        )


if __name__ == "__main__":
    main()
