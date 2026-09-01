#!/usr/bin/env python3
"""
Gate G1 — dump each task's SFT target text so you can eyeball the wire format.

This is an INSPECTION tool: by default it writes 5 rows per task so you can read
them, not a full-dataset export. Pass --limit 0 for everything.

Three things it deliberately does NOT do, each of which the previous version did:

  * It does not decode images. The target builders never look at the pixels, and
    the cache format only keeps image_id + target_json, so building full
    conversations just to throw the PIL objects away meant decoding ~48,000 images
    (4 tasks x 3 splits) for output that contains none of them.

  * It does not read the augmented split for every task. It honours each task's
    `sft_dataset_subdir`, so object_only and caption_only are inspected against the
    6308-row un-augmented split they actually train on, rather than the 8198-row
    augmented one. The target FORMAT is identical either way, but the row counts
    now match reality.

  * It does not write inside an Arrow dataset directory. Output goes to
    `datasets/inspection/<task>/`, not `datasets/processed/<task>/` — dropping
    subdirectories into a `save_to_disk` output is asking for trouble the next time
    that dataset is rebuilt.

Usage:
    python scripts/export_data_for_inspection.py                      # 5 rows/task, train
    python scripts/export_data_for_inspection.py --limit 20
    python scripts/export_data_for_inspection.py --tasks object_only caption_only
    python scripts/export_data_for_inspection.py --splits train val test --limit 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_task_config
from core.constants import VALID_TASKS
from core.io import ensure_dir, get_drive_path
from core.tasks import is_plain_text_task
from data.loader import load_processed_dataset
from data.preprocessor import build_target_json
from data.prompt_templates import SYSTEM_PROMPT, get_prompt_for_task
from evaluation.output_parser import parse_output_for_task
from data.schemas import get_output_schema


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(VALID_TASKS), choices=VALID_TASKS)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=5,
                    help="Rows per task per split. 0 = all. Default 5 (inspection).")
    args = ap.parse_args()

    out_root = ensure_dir(get_drive_path("datasets", "inspection"))
    print(f"Writing to: {out_root}\n")

    # One dataset load per distinct subdir, not per task.
    cache = {}

    def _splits_for(subdir):
        key = subdir or "__default__"
        if key not in cache:
            cache[key] = load_processed_dataset(subdir=subdir)
        return cache[key]

    for task in args.tasks:
        task_cfg = load_task_config(task)
        subdir = task_cfg.get("sft_dataset_subdir")
        splits = _splits_for(subdir)

        print("=" * 72)
        print(f"TASK: {task}"
              f"   wire format: {'bare prose' if is_plain_text_task(task) else 'fenced JSON'}")
        print(f"  SFT input: {subdir or 'default (base.yaml processed_subdir -> augmented)'}")
        print("=" * 72)

        task_dir = ensure_dir(out_root / task)
        prompt = get_prompt_for_task(task)
        schema = get_output_schema(task)
        (task_dir / "prompt.txt").write_text(
            f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{prompt}\n", encoding="utf-8")

        for split_name in args.splits:
            if split_name not in splits:
                print(f"  split '{split_name}' not present — skipping")
                continue

            rows = splits[split_name]
            n = len(rows) if args.limit == 0 else min(args.limit, len(rows))
            path = task_dir / f"{split_name}_targets.jsonl"

            ok = bad = 0
            with open(path, "w", encoding="utf-8") as f:
                for i in range(n):
                    # Column-wise access, so the "image" column is never materialised.
                    raw = {c: rows[i][c] for c in rows.column_names if c != "image"}
                    target = build_target_json(raw, task=task)

                    # Round-trip the target through the task's own parser + schema.
                    # If a task cannot parse its own SFT target, nothing downstream
                    # will work and it is better to learn that here than in a job.
                    parsed = parse_output_for_task(target, task=task)
                    valid = False
                    if parsed is not None:
                        try:
                            schema(**parsed)
                            valid = True
                        except Exception:
                            valid = False
                    ok += valid
                    bad += not valid

                    f.write(json.dumps({
                        "image_id": str(raw.get("image_id", f"row_{i}")),
                        "target": target,
                        "parses_and_validates": valid,
                    }, ensure_ascii=False) + "\n")

            flag = "" if bad == 0 else f"   <-- {bad} FAILED round-trip"
            print(f"  {split_name:5s}  {n:>6d} of {len(rows):>6d} rows  ->  "
                  f"{path.name}   valid={ok}/{n}{flag}")

            if n:
                sample = json.loads(open(path, encoding="utf-8").readline())
                preview = sample["target"]
                if len(preview) > 400:
                    preview = preview[:400] + " …[truncated]"
                print(f"\n  first target ({sample['image_id']}):")
                for line in preview.splitlines():
                    print(f"    {line}")
                print()

    print("=" * 72)
    print("WHAT TO CHECK, per task:")
    print("  unified          8 keys: caption + 3 object classes + 4 rule_N_violation")
    print("  violations_only  4 keys only, NO caption, NO object classes")
    print("  object_only      exactly 3 keys, boxes integers in [0,1000]")
    print("  caption_only     BARE PROSE — no ``` fence, no braces, no keys")
    print("  every task       valid=N/N. Any failure means the task cannot parse its")
    print("                   own SFT target, which breaks every downstream stage.")
    print("=" * 72)


if __name__ == "__main__":
    main()
