#!/usr/bin/env python3
"""
Validates a pre-baked ``thinking`` column before a single GPU-hour is spent on it.

WHY THIS EXISTS. The ``violations_think`` arm consumes a ``thinking`` column that data
prep bakes into the dataset, so **the dataset IS the format**. A bad row is frozen in,
and by the time training starts nothing downstream can detect it: no reward reads the
block, no metric scores it, and structural repair treats it as preamble. The worst case
is silent -- a block whose verdict contradicts its own label teaches the model to invert
evidence ("the trench is protected, therefore rule_3 is violated"), and every number in
the run would still look completely normal.

``data/preprocessor.py`` raises on the first bad row it meets, which is the right
behaviour for a training job but a miserable way to fix a dataset. This script reports
every offender instead: stdout is capped (``--max-print``, and 3 problems per row) to
stay readable, and the JSON report carries the COMPLETE list under ``offenders``.

Checks, per row (all via ``core/think_format.py::think_row_problems``, so this script
and the SFT target builder can never disagree about what valid means):

  1. the column is present, a string, and non-blank
  2. exactly 5 lines: caption + one per rule, in rule_1 -> rule_4 order
  3. **the block's verdict matches that row's own label, for every rule**  <-- the one
     that matters
  4. box counts in the block match the label's box counts
  5. asserted rules carry a reason, and it matches the label's reason
  6. the caption line matches ``image_caption`` verbatim
  7. no braces (which can divert structural repair's brace fallback) and no code fences

Checks, per dataset:

  8. the ``test`` split holds the SAME ``image_id`` SET as the reference dataset's
     (not byte-identical rows -- content, row order and duplicate ids are not compared).
     Every arm is scored on the same 3,004 images: ``experiments/run_inference.py`` loads
     the default root for every task, so a divergent test split here would mean the
     training data and the scored data disagree about what the test set is.
  9. no ``image_id`` appears in both train/val and test
 10. row counts, provenance histogram, per-rule violation counts, block length stats --
     reported, not enforced

Runs on a CPU login node in seconds. Needs no GPU, no model, no Java.

Usage:
    python scripts/validate_think_dataset.py
    python scripts/validate_think_dataset.py --subdir datasets/augmented_v3 --strict
    python scripts/validate_think_dataset.py --splits train --limit 500
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_config
from core.constants import RULES, VALID_TASKS
from core.io import get_drive_path
from core.think_format import (
    THINK_FIELD,
    build_think_body,
    is_violation_asserted,
    problem_bucket,
    think_row_problems,
    violations_from_row,
)

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

# Everything the checks touch. Selected explicitly so the `image` column is never
# decoded -- iterating an HF dataset with an Image() feature decodes a fresh PIL object
# per row, which on a 12k-row set is minutes of work and gigabytes of RAM for a
# string-only validation.
NEEDED_COLUMNS = ["image_id", "image_caption", THINK_FIELD, "provenance"] + [
    f"{r}_violation" for r in RULES
]


def _info(msg):
    print(f"  {msg}")


def _strip_images(ds):
    """A view carrying only the columns the checks read, with `image` dropped."""
    keep = [c for c in NEEDED_COLUMNS if c in ds.column_names]
    try:
        return ds.select_columns(keep)
    except (AttributeError, ValueError):
        drop = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(drop) if drop else ds


def check_split(ds, strict=False, limit=None):
    """Per-row checks. Returns (n_checked, offenders, stats)."""
    view = _strip_images(ds)
    n = len(view) if limit is None else min(limit, len(view))

    offenders = []                    # [(image_id, [problems])]
    problem_tally = Counter()
    provenance = Counter()
    violated = Counter()
    words = []
    non_canonical = 0

    for i in range(n):
        row = view[i]
        problems = think_row_problems(row)

        if strict and not problems:
            # Byte-identity against the canonical builder. Kept OUT of the default run:
            # a benign formatting choice by data prep (a trailing period left in place)
            # is not a reason to block a 12k-row dataset, and think_row_problems already
            # rejects everything that changes meaning. As a reported count it is still
            # worth seeing -- a nonzero value means the baked column and this repo's
            # canonical format have drifted.
            try:
                if row.get(THINK_FIELD) != build_think_body(
                    row.get("image_caption"), violations_from_row(row)
                ):
                    non_canonical += 1
                    problems = ["not byte-identical to build_think_body (--strict)"]
            except ValueError as exc:
                problems = [f"canonical form unbuildable: {exc}"]

        if problems:
            offenders.append((row.get("image_id", f"<row {i}>"), problems))
            for p in problems:
                problem_tally[problem_bucket(p)] += 1

        provenance[row.get("provenance") or "<unset>"] += 1
        for rule, v in violations_from_row(row).items():
            if is_violation_asserted(v):
                violated[rule] += 1
        body = row.get(THINK_FIELD)
        if isinstance(body, str) and body.strip():
            words.append(len(body.split()))

    stats = {
        "rows_checked": n,
        "rows_total": len(ds),
        "offender_count": len(offenders),
        "problem_tally": dict(problem_tally),
        "provenance": dict(provenance),
        "violated_by_rule": {r: violated.get(r, 0) for r in RULES},
        "block_words": {
            "mean": round(sum(words) / len(words), 2) if words else 0,
            "min": min(words) if words else 0,
            "max": max(words) if words else 0,
        },
        "non_canonical_count": non_canonical if strict else None,
        # EVERY offender, not just the ones stdout prints. stdout is capped
        # (--max-print, and 3 problems per row) to stay readable; a 12k-row dataset
        # needs the full list somewhere machine-readable, and this is it.
        "offenders": [{"image_id": i, "problems": ps} for i, ps in offenders],
    }
    return n, offenders, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="violations_think", choices=VALID_TASKS,
                    help="Task whose sft_dataset_subdir to validate (default: "
                         "violations_think).")
    ap.add_argument("--subdir", default=None,
                    help="Dataset directory to validate, relative to VLM_DATA_ROOT. "
                         "Overrides the task's sft_dataset_subdir.")
    ap.add_argument("--reference-subdir", default="datasets/augmented",
                    help="Dataset whose test split this one's must match "
                         "(default: datasets/augmented, which is what inference reads "
                         "for every task). Pass '' to skip the comparison.")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="Splits to row-check (default: train val). The test split is "
                         "never row-checked: it carries no SFT target, so it needs no "
                         "thinking column at all.")
    ap.add_argument("--strict", action="store_true",
                    help="Also require byte-identity with core/think_format.py::"
                         "build_think_body. Off by default; see check_split().")
    ap.add_argument("--limit", type=int, default=None,
                    help="Check only the first N rows per split (smoke test).")
    ap.add_argument("--max-print", type=int, default=15,
                    help="Offending rows to print per split (default: 15).")
    ap.add_argument("--report", default=None,
                    help="Where to write the JSON report (default: "
                         "datasets/stats/think_validation_<dataset>.json).")
    args = ap.parse_args()

    subdir = args.subdir
    if not subdir:
        cfg = load_config(task=args.task, training_kind="sft")
        subdir = cfg.get("sft_dataset_subdir")
        if not subdir:
            print(f"{RED}Task {args.task!r} sets no sft_dataset_subdir; pass --subdir.{RESET}")
            return 2

    from data.loader import load_processed_dataset

    print("=" * 74)
    print(f"Validating the {THINK_FIELD!r} column in {subdir}")
    print("=" * 74)

    try:
        ds = load_processed_dataset(subdir=subdir)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 2

    failures = []
    report = {"subdir": subdir, "strict": args.strict, "splits": {}}

    for split in args.splits:
        if split not in ds:
            failures.append(f"split {split!r} is missing from {subdir}")
            continue
        print(f"\n[{split}] {len(ds[split])} rows")
        n, offenders, stats = check_split(ds[split], strict=args.strict,
                                          limit=args.limit)
        report["splits"][split] = stats

        _info(f"checked           : {n}")
        _info(f"violated by rule  : " + "  ".join(
            f"{r}={stats['violated_by_rule'][r]}" for r in RULES))
        _info(f"provenance        : {stats['provenance']}")
        _info(f"block words       : {stats['block_words']}")
        if args.strict and stats["non_canonical_count"]:
            _info(f"{YELLOW}non-canonical     : {stats['non_canonical_count']}{RESET}")

        if offenders:
            failures.append(f"{split}: {len(offenders)} of {n} rows are invalid")
            print(f"  {RED}{len(offenders)} INVALID ROW(S){RESET}  "
                  f"(showing up to {args.max_print})")
            for image_id, problems in offenders[: args.max_print]:
                print(f"    {image_id}: {problems[0]}")
                for extra in problems[1:3]:
                    print(f"      + {extra}")
            print("  problem tally:")
            for k, v in sorted(stats["problem_tally"].items(), key=lambda kv: -kv[1]):
                print(f"    {v:6}  {k}")
        else:
            print(f"  {GREEN}all {n} rows valid{RESET}")

    # --- dataset-level checks ------------------------------------------------
    print("\n[dataset-level]")
    if "test" in ds:
        ids_test = set(_strip_images(ds["test"])["image_id"])
        _info(f"test rows          : {len(ids_test)}")
        if args.reference_subdir:
            try:
                ref = load_processed_dataset(subdir=args.reference_subdir)
                ids_ref = set(_strip_images(ref["test"])["image_id"])
                if ids_test == ids_ref:
                    print(f"  {GREEN}test split matches {args.reference_subdir}{RESET}")
                else:
                    failures.append(
                        f"test split differs from {args.reference_subdir} "
                        f"(+{len(ids_test - ids_ref)} / -{len(ids_ref - ids_test)})"
                    )
                    print(f"  {RED}test split DIFFERS from {args.reference_subdir}: "
                          f"{len(ids_test - ids_ref)} extra, "
                          f"{len(ids_ref - ids_test)} missing{RESET}")
                report["test_matches_reference"] = ids_test == ids_ref
            except FileNotFoundError:
                print(f"  {YELLOW}reference {args.reference_subdir} not found; "
                      f"skipped{RESET}")
        leaked = set()
        for split in args.splits:
            if split in ds:
                leaked |= set(_strip_images(ds[split])["image_id"]) & ids_test
        if leaked:
            failures.append(f"{len(leaked)} image_id(s) appear in both train/val and test")
            print(f"  {RED}LEAK: {len(leaked)} id(s) in both train/val and test{RESET}  "
                  f"e.g. {sorted(leaked)[:5]}")
        else:
            print(f"  {GREEN}no train/val -> test id leakage{RESET}")
        report["test_leakage_count"] = len(leaked)
    else:
        print(f"  {YELLOW}no test split in this dataset{RESET}")

    # --- report --------------------------------------------------------------
    report["failures"] = failures
    report_path = Path(args.report) if args.report else Path(
        get_drive_path("datasets/stats")) / f"think_validation_{Path(subdir).name}.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {report_path}")
    except OSError as exc:
        print(f"\n{YELLOW}Could not write the report ({exc}){RESET}")

    print("\n" + "=" * 74)
    if failures:
        print(f"{RED}{len(failures)} CHECK(S) FAILED{RESET}")
        for f in failures:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print(f"{GREEN}ALL CHECKS PASSED{RESET} - the thinking column is safe to train on.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
