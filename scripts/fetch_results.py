#!/usr/bin/env python3
"""
Materialises the LOCAL analysis layout from the ARC pipeline layout.

This closes BUG-23: three different result-layout assumptions coexisted across four
analysis scripts, and the step that reconciles them was never written down.

  * The pipeline WRITES (on ARC, under $VLM_DATA_ROOT):
        results/inference/<run_name>/evaluation_results/metrics.json
        results/inference/<run_name>/repair_applied/repair_report.json

  * compare_results.py READS that layout directly (it probes both
    repair_applied/evaluation_results/ and evaluation_results/).

  * plot_metrics.py, plot_metrics_vo.py, generate_comparison_csv.py and
    extract_qualitative.py read a FLAT local layout instead:
        evaluation_results/<run_name>/metrics.json
        evaluation_results/<run_name>/repair_report.json

Nothing created that flat layout, so those four scripts silently found nothing (or
found a hand-copied subset). This script builds it, from either a local
$VLM_DATA_ROOT or an rsync'd copy of the ARC results tree.

Usage
-----
    # from a local/rsync'd copy of the ARC results tree
    python scripts/fetch_results.py --task object_only --version v1 --tiers 2b 4b 8b

    # explicit source root (e.g. a directory you rsync'd into)
    python scripts/fetch_results.py --task caption_only --version v1 \\
        --source /mnt/arc_copy/results/inference

    # see what it would do
    python scripts/fetch_results.py --task unified --version v1 --dry-run

To get the tree off the cluster in the first place, from your workstation:

    rsync -av --include='*/' \\
      --include='metrics.json' --include='repair_report.json' \\
      --include='predictions_with_eval.json' --include='eval_manifest.json' \\
      --exclude='*' \\
      nabeel.shan@arc:/home/nabeel.shan/vlm-finetuning-project1/results/inference/ \\
      ./arc_results/

then point --source at ./arc_results.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import VALID_TASKS
from core.naming import results_dir_names

# Files the analysis scripts actually read, and where each lives in the source tree
# relative to results/inference/<run_name>/.
WANTED = {
    "metrics.json": ["evaluation_results/metrics.json",
                     "repair_applied/evaluation_results/metrics.json"],
    "repair_report.json": ["repair_applied/repair_report.json",
                           "repair_report.json"],
    "predictions_with_eval.json": ["evaluation_results/predictions_with_eval.json",
                                   "repair_applied/evaluation_results/predictions_with_eval.json"],
    "eval_manifest.json": ["evaluation_results/eval_manifest.json",
                           "repair_applied/evaluation_results/eval_manifest.json"],
    "still_broken.json": ["repair_applied/still_broken.json"],
    "change_manifest.json": ["repair_applied/change_manifest.json"],
}

LOCAL_ROOT = Path("evaluation_results")


def _default_source() -> Path:
    from core.io import get_drive_path
    return Path(get_drive_path("results", "inference"))


def fetch_run(source_root: Path, run_name: str, dry_run: bool) -> dict:
    src = source_root / run_name
    dest = LOCAL_ROOT / run_name
    report = {"run": run_name, "found": [], "missing": []}

    if not src.is_dir():
        report["missing"].append(f"(whole run) {src}")
        return report

    for flat_name, candidates in WANTED.items():
        for rel in candidates:
            candidate = src / rel
            if candidate.is_file():
                if not dry_run:
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, dest / flat_name)
                report["found"].append(f"{flat_name}  <-  {rel}")
                break
        else:
            report["missing"].append(flat_name)
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=VALID_TASKS)
    ap.add_argument("--version", required=True, help="Run version tag, e.g. v1")
    ap.add_argument("--tiers", nargs="+", default=["2b", "4b", "8b"])
    ap.add_argument("--source", default=None,
                    help="Root containing <run_name>/ directories. Defaults to "
                         "$VLM_DATA_ROOT/results/inference.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_root = Path(args.source) if args.source else _default_source()
    print(f"Source: {source_root}")
    print(f"Dest:   {LOCAL_ROOT.resolve()}")
    if args.dry_run:
        print("(dry run — nothing will be copied)")
    print()

    if not source_root.is_dir():
        raise SystemExit(
            f"Source root does not exist: {source_root}\n"
            "Pass --source, or set VLM_DATA_ROOT, or rsync the ARC tree down first "
            "(see this script's docstring)."
        )

    any_missing = False
    for tier in args.tiers:
        names = results_dir_names(args.task, tier, args.version)
        print(f"--- {args.task} / {tier} / {args.version} ---")
        for phase in ("baseline", "sft", "grpo"):
            rep = fetch_run(source_root, names[phase], args.dry_run)
            status = "OK" if not rep["missing"] else "PARTIAL"
            print(f"  [{status:7s}] {phase:8s} {rep['run']}")
            for f in rep["found"]:
                print(f"              + {f}")
            for m in rep["missing"]:
                any_missing = True
                print(f"              - MISSING {m}")
        print()

    if any_missing:
        print("Some files were missing. That is expected if a phase has not finished "
              "yet; the analysis scripts will simply show blanks for those.")
    print("Now run, from the repo root:")
    pfx = __import__("core.naming", fromlist=["task_prefix"]).task_prefix(args.task)
    print(f"  python -m experiments.compare_results --task {args.task} "
          f"--tier {args.tiers[-1]} --version {args.version}")
    print(f"  python -m experiments.plot_metrics --task {args.task} "
          f"--tier {args.tiers[-1]} --version {args.version}")
    print(f"  python -m experiments.generate_comparison_csv --task {args.task} "
          f"--version {args.version}")
    print(f"  # outputs land in evaluation_results/plots_{pfx}_<tier>_{args.version}/ "
          f"and csv_comparisons_{pfx}_{args.version}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
