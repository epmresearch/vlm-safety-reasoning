#!/usr/bin/env python3
"""
Discovers every finished (or partially finished) pipeline run under a results
tree and consolidates it into ONE index file.

This is the entire "download results" step, replacing manually copying each
run's metrics.json down from ARC one at a time. Run it directly on the ARC
login node against the live tree -- no GPU, no SLURM job, pure file I/O --
then transfer only the one resulting JSON file back (or keep working on ARC
directly, e.g. over a MobaXterm SSH session).

This script is READ-ONLY with respect to the pipeline: it only reads
results/inference/<run_name>/{evaluation_results,repair_applied}/*.json,
the same artifacts experiments/run_evaluation.py and
preprocessing/structural_repair.py already write. It never touches
checkpoints/, datasets/, or any file the training/inference/eval pipeline
itself depends on.

Usage
-----
    # On ARC, from the repo root -- discovers everything under $VLM_DATA_ROOT
    python -m experiments.build_results_index

    # Narrow to specific tasks/tiers/versions
    python -m experiments.build_results_index --tasks violations_only unified --tiers 2b 4b

    # Index a local flat cache (e.g. an archived evaluation_results/ folder)
    python -m experiments.build_results_index --layout flat --source evaluation_results_archive \\
        --out results_index/legacy_index.json

    # Fail loudly instead of warning if some directory doesn't match the
    # naming convention
    python -m experiments.build_results_index --strict
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import VALID_TASKS
from experiments.results_lib import discover_and_index, write_index


def _default_arc_source() -> Path:
    from core.io import get_drive_path
    return Path(get_drive_path("results", "inference"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                     help="Root containing <run_name>/ directories. Defaults to "
                          "$VLM_DATA_ROOT/results/inference for --layout arc/auto, "
                          "or ./evaluation_results for --layout flat.")
    ap.add_argument("--layout", choices=["arc", "flat", "auto"], default="auto",
                     help="arc = the real pipeline-written tree "
                          "(<run>/{evaluation_results,repair_applied}/...). "
                          "flat = a locally-cached tree (<run>/metrics.json directly). "
                          "auto (default) = try both shapes per run.")
    ap.add_argument("--tasks", nargs="+", default=None, choices=VALID_TASKS,
                     help="Restrict to these tasks. Omit to discover every task found.")
    ap.add_argument("--tiers", nargs="+", default=None,
                     help="Restrict to these tiers (e.g. 2b 4b). Omit to discover every tier found.")
    ap.add_argument("--versions", nargs="+", default=None,
                     help="Restrict to these version tags (e.g. v1). Omit to discover every version found.")
    ap.add_argument("--out", default="results_index/index.json",
                     help="Output index path (default: results_index/index.json).")
    ap.add_argument("--strict", action="store_true",
                     help="Exit nonzero if any directory fails to parse as a run name "
                          "(default: warn and skip it).")
    args = ap.parse_args()

    if args.source:
        source = Path(args.source)
    elif args.layout == "flat":
        source = Path("evaluation_results")
    else:
        source = _default_arc_source()

    print(f"Source root : {source}")
    print(f"Layout      : {args.layout}")

    if not source.is_dir():
        raise SystemExit(
            f"Source root does not exist: {source}\n"
            "Pass --source explicitly, or set VLM_DATA_ROOT (for --layout arc/auto), "
            "or point --source at a local flat cache (for --layout flat)."
        )

    runs, metrics, repair, skipped, counted = discover_and_index(
        source, args.layout, tasks=args.tasks, tiers=args.tiers, versions=args.versions,
    )

    out_path = Path(args.out)
    write_index(out_path, runs, metrics, repair, skipped, args.layout, str(source))

    print(f"\nIndexed {len(runs)} run(s), {len(metrics)} metric rows, {len(repair)} repair rows.")
    if counted:
        print("\nBy task / tier / phase / version:")
        for (task, tier, phase, version), n in sorted(counted.items()):
            print(f"  {task:<16} {tier:<6} {phase:<9} {version:<6}  ({n})")
    else:
        print("\nNothing matched. Check --tasks/--tiers/--versions filters, or that the "
              "source root actually contains run directories yet.")

    missing_metrics = [r["run_id"] for r in runs if not r["has_metrics"]]
    if missing_metrics:
        print(f"\n{len(missing_metrics)} discovered run(s) have no metrics.json yet "
              "(phase likely still running or not yet evaluated) -- they're still in "
              "the index with has_metrics=false, just contribute no metric rows:")
        for rid in missing_metrics[:20]:
            print(f"  - {rid}")
        if len(missing_metrics) > 20:
            print(f"  ... and {len(missing_metrics) - 20} more")

    if skipped:
        print(f"\nSkipped {len(skipped)} director{'y' if len(skipped) == 1 else 'ies'} "
              "not matching <prefix>-<phase>-<tier>-<version>[_best|_final] "
              "(legacy naming or unrelated content):")
        for name in skipped[:20]:
            print(f"  - {name}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
        if args.strict:
            raise SystemExit(1)

    print(f"\nWrote index to: {out_path.resolve()}")
    print("Next: python -m experiments.compare_all --index "
          f"{out_path} --out {out_path.parent}")


if __name__ == "__main__":
    main()
