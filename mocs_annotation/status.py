#!/usr/bin/env python3
"""
One-command status across every MOCS annotation run, plus a disjointness audit.

Written for the multi-run sweep: once the corpus is split across four or five
directories there is no single progress.json to read, and the question that actually
matters -- "is any image being annotated twice?" -- cannot be answered by looking at
one run at a time.

Two things it reports:

  1. PER-RUN PROGRESS. Selected / attempted / ok / failed, throughput and ETA, read
     straight from each run's selection.json, proposals.jsonl and progress.json.
     Also flags a run whose parse-failure rate is above 5%, which is the signal that
     the prompt or the token budget needs attention rather than the model.

  2. PAIRWISE OVERLAP between every pair of selections. This MUST be zero
     everywhere. Chained --exclude is the only thing preventing two runs from
     annotating the same image, and a mistake there is invisible until you combine
     the proposal files and find duplicate ids. Checking it costs a second.

Runs anywhere -- pure file I/O, no GPU, no model, no dataset.

Usage:
    python -m mocs_annotation.status --dirs $OUT $OUT2 $OUT3 $OUT4 $OUT5
    python -m mocs_annotation.status --root $VLM_DATA_ROOT/datasets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES

# Above this share of parse/schema/generation failures, the run is telling you
# something is wrong with the prompt or the token budget -- not with the model.
FAILURE_RATE_ALERT = 0.05


def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _selection_ids(run_dir: Path) -> set:
    payload = _read_json(run_dir / "selection.json")
    if not payload:
        return set()
    return {r["new_image_id"] for r in payload.get("records", []) if "new_image_id" in r}


def _scan_proposals(run_dir: Path) -> Tuple[Counter, Counter, int]:
    """Status counts, per-rule proposal counts, and total lines.

    Streams the file rather than loading it: these reach tens of thousands of lines
    and this is meant to be run repeatedly while jobs are in flight. A truncated
    final line (a job killed mid-write) is skipped, not fatal.
    """
    statuses: Counter = Counter()
    rules: Counter = Counter()
    total = 0
    path = run_dir / "proposals.jsonl"
    if not path.exists():
        return statuses, rules, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            statuses[rec.get("status", "?")] += 1
            for r in RULES:
                if rec.get(f"{r}_violation") is not None:
                    rules[r] += 1
    return statuses, rules, total


def _fmt_eta(progress: Optional[dict]) -> str:
    if not progress:
        return "-"
    eta = progress.get("eta_seconds")
    rate = progress.get("images_per_second")
    if eta is None or not rate:
        return "-"
    return f"{rate:.3f}/s  eta {eta / 3600:.1f}h"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=None, help="Run directories, in order")
    ap.add_argument("--root", default=None,
                     help="Parent directory to scan for mocs_annotation* subdirectories")
    ap.add_argument("--pattern", default="mocs_annotation*", help="Glob used with --root")
    args = ap.parse_args()

    if args.dirs:
        dirs = [Path(d) for d in args.dirs]
    elif args.root:
        dirs = sorted(p for p in Path(args.root).glob(args.pattern) if p.is_dir())
    else:
        raise SystemExit("Pass --dirs or --root.")

    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        raise SystemExit("No run directories found.")

    print(f"\n{'run':<28}{'selected':>9}{'done':>8}{'ok':>8}{'fail':>7}{'fail%':>7}  progress")
    print("-" * 96)

    all_ids: Dict[str, set] = {}
    grand_rules: Counter = Counter()
    grand_selected = grand_done = grand_ok = grand_fail = 0

    for d in dirs:
        ids = _selection_ids(d)
        all_ids[d.name] = ids
        statuses, rules, total = _scan_proposals(d)
        progress = _read_json(d / "progress.json")

        ok = statuses.get("ok", 0)
        fail = total - ok
        rate = (fail / total) if total else 0.0
        flag = "  <-- high failure rate" if rate > FAILURE_RATE_ALERT else ""

        print(f"{d.name:<28}{len(ids):>9}{total:>8}{ok:>8}{fail:>7}{rate * 100:>6.1f}%  "
              f"{_fmt_eta(progress)}{flag}")

        grand_selected += len(ids)
        grand_done += total
        grand_ok += ok
        grand_fail += fail
        grand_rules.update(rules)

    print("-" * 96)
    pct = (grand_done / grand_selected * 100) if grand_selected else 0.0
    print(f"{'TOTAL':<28}{grand_selected:>9}{grand_done:>8}{grand_ok:>8}{grand_fail:>7}"
          f"{(grand_fail / grand_done * 100) if grand_done else 0:>6.1f}%  "
          f"{pct:.1f}% of the corpus attempted")

    print(f"\nPROPOSALS BY RULE (successful records only, NOT verified labels)")
    for r in RULES:
        print(f"  {r:<10}{grand_rules[r]:>7}")

    # --- the audit that actually matters -------------------------------------
    print(f"\nPAIRWISE SELECTION OVERLAP (every cell must be 0)")
    names = list(all_ids)
    worst = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(all_ids[a] & all_ids[b])
            worst = max(worst, n)
            mark = "  <-- DUPLICATE IMAGES" if n else ""
            print(f"  {a:<26} x {b:<26} {n:>6}{mark}")
    if worst == 0:
        print("\n  OK -- no image appears in more than one run.")
    else:
        print(f"\n  PROBLEM -- {worst} image(s) shared between runs. Chained --exclude was "
              "missed somewhere; those images will be annotated twice and produce "
              "duplicate ids when the proposal files are combined.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
