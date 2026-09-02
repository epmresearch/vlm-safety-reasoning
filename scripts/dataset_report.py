#!/usr/bin/env python3
"""
Full A-to-Z inventory of every prepared dataset root. No GPU, no tokenizer, no images
decoded.

Why this exists: the prep scripts only logged to stdout, so once a SLURM job's output
scrolled away there was no persisted record of what the data actually contained --
and every downstream decision (true-negative constants, augmentation multipliers,
stratification, token budgets, which task is even viable) turns on those numbers.

Writes one machine-readable file plus a readable console summary:

    $VLM_DATA_ROOT/datasets/stats/dataset_report.json

Covers, per root and per split:

  * row counts, and how many rows are augmentation duplicates (``_aug`` suffix)
  * object classes: images containing each, total boxes, boxes per image, prevalence,
    and the implied ``(1-p)^32`` batch-starvation rate
  * violation rules: images per rule, contentless-vs-substantive assertions, the safe
    (rule_0) count, and the full co-occurrence matrix
  * captions: word and sentence distribution (mean/median/p10/p90/max), blanks
  * violation reasons: word distribution per rule -- what the length penalty targets
  * image condition labels (illumination, camera distance, view, quality of info)
  * box hygiene: out-of-[0,1] coordinates, degenerate/zero-area boxes, area percentiles
  * derived decision numbers: per-class break-even IoU at the configured
    ``grounding_tn_constant``, and the violation-abstention crossover

Usage:
    python scripts/dataset_report.py                       # every root it can find
    python scripts/dataset_report.py --roots processed
    python scripts/dataset_report.py --splits train        # faster on a big root
    python scripts/dataset_report.py --out my_report.json
"""
import argparse
import json
import math
import os
import re
import statistics as st
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.io import ensure_dir, get_drive_path
from core.logging import get_logger

logger = get_logger(__name__)

OBJECT_CLASSES = ("excavator", "rebar", "worker_with_white_hard_hat")
RULES = (1, 2, 3, 4)
FEATURE_COLUMNS = ("illumination", "camera_distance", "view", "quality_of_info")
BATCH = 32  # effective SFT batch, for the starvation figure

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


def _dist(values: List[float]) -> Dict[str, Any]:
    """Distribution summary. Percentiles by index so it works on tiny inputs too."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    at = lambda q: s[min(int(q * len(s)), len(s) - 1)]
    return {
        "n": len(s),
        "mean": round(st.mean(s), 2),
        "median": round(st.median(s), 2),
        "min": round(s[0], 2),
        "p10": round(at(0.10), 2),
        "p90": round(at(0.90), 2),
        "p99": round(at(0.99), 2),
        "max": round(s[-1], 2),
    }


def _boxes(value: Any) -> List[List[float]]:
    if not isinstance(value, list):
        return []
    return [b for b in value if isinstance(b, (list, tuple)) and len(b) == 4]


def _violation(row: Dict[str, Any], rule: int) -> Optional[Dict[str, Any]]:
    v = row.get(f"rule_{rule}_violation")
    return v if isinstance(v, dict) else None


def analyse_split(rows) -> Dict[str, Any]:
    """One pass over a split, collecting everything. Images are never touched."""
    n = len(rows)
    out: Dict[str, Any] = {"rows": n}

    aug = 0
    obj_imgs = Counter()
    obj_boxes = Counter()
    rule_imgs = Counter()
    rule_contentless = Counter()
    rule_boxes = Counter()
    reason_words: Dict[int, List[int]] = {r: [] for r in RULES}
    cooccur = Counter()
    safe = 0
    cap_words: List[int] = []
    cap_sents: List[int] = []
    cap_blank = 0
    features = {c: Counter() for c in FEATURE_COLUMNS}
    areas: List[float] = []
    bad_range = 0
    degenerate = 0

    cols = [c for c in rows.column_names if c != "image"]
    for row in rows.select_columns(cols):
        if isinstance(row.get("image_id"), str) and re.search(r"_aug\d+$", row["image_id"]):
            aug += 1

        for c in OBJECT_CLASSES:
            bs = _boxes(row.get(c))
            if bs:
                obj_imgs[c] += 1
                obj_boxes[c] += len(bs)
            for b in bs:
                try:
                    x0, y0, x1, y1 = (float(v) for v in b)
                except (TypeError, ValueError):
                    degenerate += 1
                    continue
                if min(x0, y0, x1, y1) < 0.0 or max(x0, y0, x1, y1) > 1.0:
                    bad_range += 1
                if x1 <= x0 or y1 <= y0:
                    degenerate += 1
                else:
                    areas.append((x1 - x0) * (y1 - y0))

        present = []
        for r in RULES:
            v = _violation(row, r)
            if v is None:
                continue
            present.append(r)
            rule_imgs[r] += 1
            bs = _boxes(v.get("bounding_box"))
            rule_boxes[r] += len(bs)
            reason = v.get("reason")
            has_reason = isinstance(reason, str) and reason.strip()
            if has_reason:
                reason_words[r].append(len(reason.split()))
            # An assertion with neither a reason nor a box: counted as a prediction for
            # precision but never credited a true positive. See reward_violation_id.
            if not has_reason and not bs:
                rule_contentless[r] += 1
        if present:
            cooccur[tuple(present)] += 1
        else:
            safe += 1

        cap = row.get("image_caption")
        if isinstance(cap, str) and cap.strip():
            cap_words.append(len(cap.split()))
            cap_sents.append(len(re.findall(r"[.!?]+", cap)) or 1)
        else:
            cap_blank += 1

        for c in FEATURE_COLUMNS:
            v = row.get(c)
            if v is not None:
                features[c][str(v)] += 1

    out["augmented_rows"] = aug
    out["augmented_pct"] = _pct(aug, n)

    out["objects"] = {
        c: {
            "images_with": obj_imgs[c],
            "prevalence_pct": _pct(obj_imgs[c], n),
            "total_boxes": obj_boxes[c],
            "boxes_per_image_with": round(obj_boxes[c] / obj_imgs[c], 2) if obj_imgs[c] else 0.0,
            "batch_starved_pct": round(100.0 * (1.0 - obj_imgs[c] / n) ** BATCH, 2) if n else 0.0,
        }
        for c in OBJECT_CLASSES
    }
    # images_with_no_object is filled by _fix_no_object, which needs its own pass.

    out["violations"] = {
        f"rule_{r}": {
            "images_with": rule_imgs[r],
            "prevalence_pct": _pct(rule_imgs[r], n),
            "contentless_assertions": rule_contentless[r],
            "total_boxes": rule_boxes[r],
            "reason_words": _dist([float(x) for x in reason_words[r]]),
            "batch_starved_pct": round(100.0 * (1.0 - rule_imgs[r] / n) ** BATCH, 2) if n else 0.0,
        }
        for r in RULES
    }
    out["violations"]["rule_0_safe"] = {"images": safe, "prevalence_pct": _pct(safe, n)}
    out["rule_cooccurrence"] = {
        ("+".join(f"rule_{r}" for r in k) if k else "safe"): v
        for k, v in sorted(cooccur.items(), key=lambda kv: -kv[1])
    }

    out["captions"] = {
        "with_caption": len(cap_words),
        "blank": cap_blank,
        "blank_pct": _pct(cap_blank, n),
        "words": _dist([float(x) for x in cap_words]),
        "sentences": _dist([float(x) for x in cap_sents]),
    }
    out["image_features"] = {
        c: {k: {"count": v, "pct": _pct(v, n)} for k, v in cnt.most_common()}
        for c, cnt in features.items()
    }
    out["box_hygiene"] = {
        "object_boxes_out_of_unit_range": bad_range,
        "object_boxes_degenerate": degenerate,
        "object_box_area_fraction": _dist(areas),
    }
    return out


def _fix_no_object(rows, split_report: Dict[str, Any]) -> None:
    """Counts images with zero boxes across all three classes. Its own tiny pass keeps
    analyse_split readable, and it is the number that decides how much of an
    object_only GRPO run produces no gradient at all."""
    cols = [c for c in rows.column_names if c != "image"]
    none_ = sum(
        1 for row in rows.select_columns(cols)
        if not any(_boxes(row.get(c)) for c in OBJECT_CLASSES)
    )
    split_report["images_with_no_object"] = none_
    split_report["images_with_no_object_pct"] = _pct(none_, len(rows))


def decision_numbers(train_report: Dict[str, Any]) -> Dict[str, Any]:
    """Turns measured prevalence into the constants the reward design depends on.

    Emitting boxes for class k is positive-EV only when
        E[IoU_k] > c_k * (1 - p_k) / p_k
    so this is where you check whether the configured grounding_tn_constant still puts
    every class within reach. validate_rewards.py fails the build above 0.75.
    """
    from rewards.reward_utils import grounding_tn_constant

    out: Dict[str, Any] = {}
    for task in ("unified", "object_only"):
        rows = {}
        for c in OBJECT_CLASSES:
            p = train_report["objects"][c]["prevalence_pct"] / 100.0
            try:
                ck = float(grounding_tn_constant(task, c))
            except Exception:
                ck = 0.15
            be = (ck * (1 - p) / p) if p > 0 else float("inf")
            rows[c] = {
                "prevalence": round(p, 4),
                "grounding_tn_constant": ck,
                "break_even_iou": (round(be, 3) if math.isfinite(be) else "inf"),
                "reachable": bool(math.isfinite(be) and be <= 0.75),
            }
        out[task] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["processed", "augmented", "grpo_pool"],
                    help="Dataset roots under datasets/ to inventory.")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Restrict to these splits (default: all present).")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    args = ap.parse_args()

    from datasets import load_from_disk

    report: Dict[str, Any] = {"roots": {}, "batch_size_for_starvation": BATCH}

    for root in args.roots:
        path = get_drive_path("datasets", root)
        if not os.path.isdir(path):
            print(f"{DIM}skip {root}: not present at {path}{RESET}")
            continue
        print(f"\n{YELLOW}{'=' * 74}{RESET}")
        print(f"{YELLOW}ROOT: datasets/{root}{RESET}  {DIM}{path}{RESET}")
        print(f"{YELLOW}{'=' * 74}{RESET}")

        ds = load_from_disk(str(path))
        splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
        if args.splits:
            splits = [s for s in splits if s in args.splits]

        report["roots"][root] = {"path": str(path), "splits": {}}
        for sp in splits:
            rows = ds[sp] if hasattr(ds, "keys") else ds
            print(f"\n{DIM}analysing {sp} ({len(rows)} rows)...{RESET}")
            r = analyse_split(rows)
            _fix_no_object(rows, r)
            report["roots"][root]["splits"][sp] = r

            print(f"  {sp}: {r['rows']} rows"
                  + (f", {r['augmented_rows']} augmented ({r['augmented_pct']}%)"
                     if r["augmented_rows"] else ""))
            print(f"    {'OBJECT CLASS':30s} {'imgs':>6s} {'prev':>7s} {'boxes':>7s} "
                  f"{'b/img':>6s} {'batch starved':>14s}")
            for c in OBJECT_CLASSES:
                o = r["objects"][c]
                col = RED if o["batch_starved_pct"] > 20 else (
                    YELLOW if o["batch_starved_pct"] > 5 else GREEN)
                print(f"    {c:30s} {o['images_with']:>6d} {o['prevalence_pct']:>6.2f}% "
                      f"{o['total_boxes']:>7d} {o['boxes_per_image_with']:>6.2f} "
                      f"{col}{o['batch_starved_pct']:>13.2f}%{RESET}")
            print(f"    {'images with NO object':30s} {r['images_with_no_object']:>6d} "
                  f"{r['images_with_no_object_pct']:>6.2f}%   "
                  f"{DIM}<- zero grounding gradient for object_only{RESET}")

            print(f"    {'RULE':30s} {'imgs':>6s} {'prev':>7s} {'boxes':>7s} "
                  f"{'contentless':>12s} {'reason words':>13s}")
            for rr in RULES:
                v = r["violations"][f"rule_{rr}"]
                rw = v["reason_words"]
                print(f"    {'rule_' + str(rr):30s} {v['images_with']:>6d} "
                      f"{v['prevalence_pct']:>6.2f}% {v['total_boxes']:>7d} "
                      f"{v['contentless_assertions']:>12d} "
                      f"{(str(rw.get('mean', '-')) + ' mean'):>13s}")
            s0 = r["violations"]["rule_0_safe"]
            print(f"    {'rule_0 (safe)':30s} {s0['images']:>6d} {s0['prevalence_pct']:>6.2f}%")

            c = r["captions"]
            print(f"    captions: {c['with_caption']} present, {c['blank']} blank "
                  f"({c['blank_pct']}%) | words mean {c['words'].get('mean')} "
                  f"median {c['words'].get('median')} p10 {c['words'].get('p10')} "
                  f"p90 {c['words'].get('p90')} max {c['words'].get('max')} "
                  f"| sentences mean {c['sentences'].get('mean')}")
            bh = r["box_hygiene"]
            print(f"    box hygiene: {bh['object_boxes_out_of_unit_range']} out of [0,1], "
                  f"{bh['object_boxes_degenerate']} degenerate")

        # Decision numbers off the train split of this root
        tr = report["roots"][root]["splits"].get("train")
        if tr:
            report["roots"][root]["decision_numbers"] = decision_numbers(tr)

    if not report["roots"]:
        print(f"\n{RED}No dataset roots found. Run the prep first.{RESET}")
        return 1

    out_path = args.out or (ensure_dir(get_drive_path("datasets", "stats")) / "dataset_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{GREEN}Wrote {out_path}{RESET}")

    # The one derived table worth surfacing: is every object class still worth detecting?
    for root, blob in report["roots"].items():
        dn = blob.get("decision_numbers")
        if not dn:
            continue
        print(f"\n{YELLOW}BREAK-EVEN IoU from datasets/{root} train prevalence{RESET}")
        for task, classes in dn.items():
            print(f"  {task}")
            for c, d in classes.items():
                col = GREEN if d["reachable"] else RED
                print(f"    {c:30s} p={d['prevalence']:<7} c={d['grounding_tn_constant']:<6} "
                      f"break-even={col}{d['break_even_iou']}{RESET}")
        print(f"  {DIM}emitting a class is positive-EV only above its break-even; "
              f"validate_rewards.py fails above 0.75{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
