#!/usr/bin/env python3
"""
Merges every MOCS annotation run into ONE self-contained corpus directory.

The five runs partition the MOCS corpus disjointly, but each one lives in its own
directory with its own selection.json / proposals.jsonl / manifests, and a proposal
record does NOT carry enough on its own to find its image. This turns all of that
into a single file plus the provenance needed to defend it later.

WHAT IT FIXES, all of it discovered from the code rather than assumed:

  1. NO RECORD CARRIES `images_root`. select_images.py writes it per record into
     selection.json, but schema.py::proposal_to_record does not copy it through. So
     the image path has to be re-derived. Three tiers, in order:
        a. the per-record `images_root` in that run's selection.json (runs 2-5)
        b. the record's `source` ("val" / "test") mapped through --images-root-val /
           --images-root-test
        c. the filename number range documented in schema.py:175 --
           val 19406-23406, test 23407-41672
     Run 1 predates multi-source support (its manifest carries `sources: []`), so it
     falls through to (a)'s backward-compatible manifest-level `images_root`, and is
     100% val either way.

  2. TWO RECORD SHAPES. proposal_to_record (status ok) carries image_caption, the four
     rule_N_violation fields, flagged_rules, mocs_categories and worker_machine_pairs.
     failure_record carries none of those -- just status/error/raw_output. Combined
     output is ONE uniform schema; the sidecar fields are back-filled onto failures
     from selection.json so downstream code never has to branch on status.

  3. BOXES ARE [0,1], NOT 0-1000. proposal_to_record already ran
     _violation_to_dataset_scale, so what is on disk is dataset scale -- the same
     convention data/preprocessor.py::build_target_json consumes. This file keeps that
     untouched and VALIDATES it (CLAUDE.md: a box in [0,1] divided by 1000 again
     collapses to a point and silently zeroes every IoU). Only the reviewer-facing CSV
     converts up to 0-1000.

  4. `worker_machine_pairs` EXISTS ONLY FOR val-SOURCED RECORDS. MOCS's test split
     ships no annotations at all, so the human-annotated rule_4 union box -- the one
     worth more than anything the teacher drew -- is present on ~4,000 of ~22,000
     records. The manifest reports that count so nobody mistakes an empty column for a
     broken one.

This step is PRE-verification. It produces proposals, never training data. Turning
verified rows into a HuggingFace dataset that matches datasets/processed column for
column is a separate, later step (see schema.py::proposal_to_record's docstring).

Runs anywhere -- pure file I/O. No GPU, no model, no dataset, no unsloth.

Usage:
    python -m mocs_annotation.combine \
        --root $VLM_DATA_ROOT/datasets \
        --out  $VLM_DATA_ROOT/datasets/mocs_annotation_combined
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES
from core.logging import get_logger
from mocs_annotation.schema import STATUS_OK

logger = get_logger(__name__)

# schema.py:175 documents these ranges as the fallback that distinguishes a val image
# from a test one when the `source` field is absent (run 1's selection predates it).
VAL_ID_RANGE = (19406, 23406)
TEST_ID_RANGE = (23407, 41672)

# One uniform key order for every output record, so the file diffs cleanly and a
# reader can rely on position as well as name.
FIELD_ORDER = (
    # identity / origin -- everything needed to trace a row back to a MOCS file
    "new_image_id", "mocs_image_id", "file_name", "width", "height", "source",
    # provenance -- which run produced it, under which teacher and contract
    "run", "prompt_sha256", "model", "batch_size", "selection_bucket",
    # resolved location. `images_root` is deliberately NOT carried: it is exactly the
    # prefix of image_path, recoverable two ways, and cost 4.5% of the file.
    "image_path",
    # payload
    "status", "image_caption",
    *[f"{r}_violation" for r in RULES],
    "flagged_rules", "n_flagged",
    # MOCS sidecar (val-sourced records only)
    "mocs_categories", "worker_machine_pairs",
    # failure detail / audit trail
    "error", "raw_output",
)


def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"could not read {path}: {e}")
        return None


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _file_number(file_name: str) -> Optional[int]:
    stem = Path(file_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _source_from_filename(file_name: str) -> str:
    """Last-resort val/test discrimination, per schema.py:175."""
    n = _file_number(file_name)
    if n is None:
        return ""
    if VAL_ID_RANGE[0] <= n <= VAL_ID_RANGE[1]:
        return "val"
    if TEST_ID_RANGE[0] <= n <= TEST_ID_RANGE[1]:
        return "test"
    return ""


def discover_runs(root: Path, out_dir: Path) -> List[Path]:
    """Every sibling directory that actually holds a proposals.jsonl.

    Deliberately checks for the file rather than matching the name pattern alone: the
    output directory is itself called mocs_annotation_combined and must never be read
    back in as an input.
    """
    runs = []
    for d in sorted(root.glob("mocs_annotation*")):
        if not d.is_dir() or d.resolve() == out_dir.resolve():
            continue
        if (d / "proposals.jsonl").exists():
            runs.append(d)
        else:
            logger.warning(f"{d.name}: no proposals.jsonl -- skipping")
    return runs


def load_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """(proposal records, selection-by-id, run provenance)."""
    records: List[Dict[str, Any]] = []
    bad_lines = 0
    with open(run_dir / "proposals.jsonl", "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # A walltime kill mid-write leaves a truncated final line; annotate.py
                # tolerates it on resume and so do we.
                bad_lines += 1
                logger.warning(f"{run_dir.name}/proposals.jsonl:{lineno} unparseable -- skipped")

    sel_payload = _read_json(run_dir / "selection.json") or {}
    sel_records = sel_payload.get("records", []) if isinstance(sel_payload, dict) else []
    selection = {r["new_image_id"]: r for r in sel_records if "new_image_id" in r}
    sel_manifest = sel_payload.get("manifest", {}) if isinstance(sel_payload, dict) else {}

    ann_manifest = _read_json(run_dir / "annotate_manifest.json") or {}
    progress = _read_json(run_dir / "progress.json") or {}

    prov = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "prompt_sha256": ann_manifest.get("prompt_sha256"),
        # annotate_manifest.json records `model` as None in the runs done so far --
        # the id was only ever on the sbatch line. --model stamps it back.
        "model": ann_manifest.get("model"),
        "batch_size": ann_manifest.get("batch_size"),
        "max_new_tokens": ann_manifest.get("max_new_tokens"),
        "repetition_penalty": ann_manifest.get("repetition_penalty"),
        "box_hints": ann_manifest.get("box_hints"),
        "manifest_images_root": ann_manifest.get("images_root"),
        "selection_total": sel_manifest.get("selected_total", len(sel_records)),
        "selection_sources": [
            Path(s.get("annotations", "")).name for s in sel_manifest.get("sources", [])
        ],
        "quotas_filled": sel_manifest.get("quotas_filled", {}),
        "proposals_lines": len(records),
        "unparseable_lines": bad_lines,
        "progress": progress,
    }
    return records, selection, prov


def resolve_images_root(
    rec: Dict[str, Any],
    sel_rec: Optional[Dict[str, Any]],
    prov: Dict[str, Any],
    root_val: Optional[str],
    root_test: Optional[str],
) -> Tuple[str, str]:
    """(images_root, how_it_was_resolved). See the module docstring for the order."""
    if sel_rec and sel_rec.get("images_root"):
        return str(sel_rec["images_root"]), "selection_record"

    source = rec.get("source") or (sel_rec or {}).get("source") or ""
    if source == "val" and root_val:
        return root_val, "source_field"
    if source == "test" and root_test:
        return root_test, "source_field"

    guessed = _source_from_filename(rec.get("file_name", ""))
    if guessed == "val" and root_val:
        return root_val, "filename_range"
    if guessed == "test" and root_test:
        return root_test, "filename_range"

    if prov.get("manifest_images_root"):
        return str(prov["manifest_images_root"]), "run_manifest"
    return "", "unresolved"


def normalise(
    rec: Dict[str, Any],
    sel_rec: Optional[Dict[str, Any]],
    prov: Dict[str, Any],
    images_root: str,
) -> Dict[str, Any]:
    """One uniform schema for ok and failed records alike."""
    out: Dict[str, Any] = {}

    source = rec.get("source") or (sel_rec or {}).get("source") or _source_from_filename(
        rec.get("file_name", "")
    )
    file_name = rec.get("file_name", "")

    flagged = [r for r in RULES if rec.get(f"{r}_violation") is not None]

    base = {
        "new_image_id": rec.get("new_image_id"),
        "mocs_image_id": rec.get("mocs_image_id"),
        "file_name": file_name,
        "width": rec.get("width"),
        "height": rec.get("height"),
        "source": source,
        "run": prov["run"],
        "prompt_sha256": prov.get("prompt_sha256"),
        "model": prov.get("model"),
        "batch_size": prov.get("batch_size"),
        "selection_bucket": rec.get("selection_bucket", ""),
        "images_root": images_root,
        "image_path": str(Path(images_root) / file_name) if images_root and file_name else "",
        "status": rec.get("status"),
        "image_caption": rec.get("image_caption", ""),
        "flagged_rules": flagged,
        "n_flagged": len(flagged),
        # Back-filled from selection.json when the record is a failure, so the schema
        # is uniform and a later consumer never branches on status to find geometry.
        "mocs_categories": rec.get("mocs_categories")
        or (sel_rec or {}).get("mocs_categories")
        or [],
        "worker_machine_pairs": rec.get("worker_machine_pairs")
        or (sel_rec or {}).get("worker_machine_pairs")
        or [],
        "error": rec.get("error", ""),
        "raw_output": rec.get("raw_output", ""),
    }
    for r in RULES:
        base[f"{r}_violation"] = rec.get(f"{r}_violation")

    for k in FIELD_ORDER:
        out[k] = base.get(k)
    return out


def check_boxes(rec: Dict[str, Any]) -> List[str]:
    """Dataset-scale sanity. A [0,1] box scaled down a second time collapses to a
    point and scores identically to omitting it -- silent, and documented in CLAUDE.md
    as this repo's most expensive box bug."""
    problems = []
    for r in RULES:
        v = rec.get(f"{r}_violation")
        if not isinstance(v, dict):
            continue
        for box in v.get("bounding_box") or []:
            if len(box) != 4:
                problems.append(f"{r}: box has {len(box)} coords")
                continue
            x1, y1, x2, y2 = box
            if not all(isinstance(c, (int, float)) for c in box):
                problems.append(f"{r}: non-numeric coord")
            elif not all(-0.001 <= c <= 1.001 for c in box):
                problems.append(f"{r}: coord outside [0,1] -- {box}")
            elif x2 <= x1 or y2 <= y1:
                problems.append(f"{r}: degenerate box {box}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Directory holding the mocs_annotation* run dirs "
                                    "(auto-discovers every one with a proposals.jsonl)")
    src.add_argument("--runs", nargs="+", help="Explicit list of run directories")
    ap.add_argument("--out", required=True, help="Output corpus directory")
    ap.add_argument("--images-root-val", default=None,
                    help="Directory of the MOCS val jpgs. Only needed when a run's "
                         "selection.json has no per-record images_root "
                         "(default: $VLM_DATA_ROOT/datasets/filtered/instances_val)")
    ap.add_argument("--images-root-test", default=None,
                    help="Same for the test jpgs "
                         "(default: $VLM_DATA_ROOT/datasets/filtered/instances_test)")
    ap.add_argument("--model", default=None,
                    help="Stamp the teacher's model id onto every record. The runs "
                         "recorded `model: null`, so pass it (e.g. "
                         "Qwen/Qwen3-VL-32B-Instruct) to keep the corpus self-describing")
    ap.add_argument("--fewshot-from", default=None,
                    help="Run directory whose fewshot.json + fewshot_images/ to copy in "
                         "as the prompt contract (default: the first run that has one)")
    ap.add_argument("--raw", choices=["failures", "all", "none"], default="failures",
                    help="Which records keep `raw_output`. It is 30%% of the file, and on "
                         "an ok record it merely duplicates image_caption and the rule "
                         "fields. On a FAILURE it is irreplaceable -- it is what lets a "
                         "smarter parser recover truncated replies later without re-running "
                         "the GPU. Default 'failures' saves ~12 MB and loses nothing that "
                         "cannot be regenerated")
    ap.add_argument("--no-copy-sources", action="store_true",
                    help="Do not copy each run's selection.json / manifests into "
                         "source_runs/. They are the only record of how images were "
                         "chosen, so copying is the default")
    ap.add_argument("--check-images", choices=["all", "sample", "none"], default="all",
                    help="Verify resolved image paths exist. 'all' is ~22k stat calls, "
                         "a few seconds, and catches a wrong images root before the "
                         "reviewer does")
    args = ap.parse_args()

    data_root = os.environ.get("VLM_DATA_ROOT", "")
    root_val = args.images_root_val or (
        str(Path(data_root) / "datasets" / "filtered" / "instances_val") if data_root else None
    )
    root_test = args.images_root_test or (
        str(Path(data_root) / "datasets" / "filtered" / "instances_test") if data_root else None
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = (
        [Path(r) for r in args.runs]
        if args.runs
        else discover_runs(Path(args.root), out_dir)
    )
    if not run_dirs:
        raise SystemExit("no run directories with a proposals.jsonl were found")
    logger.info(f"Combining {len(run_dirs)} run(s): {', '.join(d.name for d in run_dirs)}")

    all_records: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}
    duplicates: List[Dict[str, str]] = []
    resolution_counts: Counter = Counter()
    per_run_status: Dict[str, Counter] = {}

    for run_dir in run_dirs:
        records, selection, prov = load_run(run_dir)
        if args.model:
            prov["model"] = args.model
        logger.info(
            f"  {run_dir.name:<28} {len(records):>6} record(s), "
            f"selection {prov['selection_total']}, prompt {str(prov['prompt_sha256'])[:12]}"
        )

        status_counter: Counter = Counter()
        for rec in records:
            rid = rec.get("new_image_id")
            sel_rec = selection.get(rid)
            images_root, how = resolve_images_root(rec, sel_rec, prov, root_val, root_test)
            resolution_counts[how] += 1

            norm = normalise(rec, sel_rec, prov, images_root)
            if args.raw == "none" or (args.raw == "failures" and norm["status"] == STATUS_OK):
                norm["raw_output"] = ""
            status_counter[norm["status"] or "?"] += 1

            if rid in seen_ids:
                duplicates.append({"new_image_id": rid, "first": seen_ids[rid], "second": prov["run"]})
            else:
                seen_ids[rid] = prov["run"]
                all_records.append(norm)

        per_run_status[prov["run"]] = status_counter
        prov["status_counts"] = dict(status_counter)
        provenance.append(prov)

    all_records.sort(key=lambda r: (r["source"] or "", r["new_image_id"] or ""))

    # ---------------------------------------------------------------- checks
    logger.info("")
    logger.info("CHECKS")

    prompts = {p["prompt_sha256"] for p in provenance if p.get("prompt_sha256")}
    prompt_ok = len(prompts) == 1
    logger.info(
        f"  prompt contract        {'OK  one prompt across all runs' if prompt_ok else 'MISMATCH ' + str(prompts)}"
        + (f"  ({list(prompts)[0][:16]})" if prompt_ok else "")
    )
    if not prompt_ok:
        logger.warning(
            "  Runs used different few-shot blocks. They can still be combined, but do "
            "not pool them in any rate comparison -- filter on prompt_sha256 first."
        )

    logger.info(
        f"  duplicate ids          {'OK  none' if not duplicates else f'{len(duplicates)} DUPLICATE(S) -- first kept'}"
    )
    for d in duplicates[:5]:
        logger.warning(f"    {d['new_image_id']} in {d['first']} and {d['second']}")

    expected = sum(p["proposals_lines"] for p in provenance)
    count_ok = len(all_records) + len(duplicates) == expected
    logger.info(
        f"  record count           {'OK' if count_ok else 'MISMATCH'}  "
        f"{len(all_records)} kept + {len(duplicates)} dup = {expected} read"
    )

    box_problems: List[Dict[str, Any]] = []
    empty_captions = 0
    source_mismatch = 0
    for rec in all_records:
        if rec["status"] == STATUS_OK:
            probs = check_boxes(rec)
            if probs:
                box_problems.append({"new_image_id": rec["new_image_id"], "problems": probs})
            if not (rec.get("image_caption") or "").strip():
                empty_captions += 1
        guessed = _source_from_filename(rec.get("file_name", ""))
        if guessed and rec.get("source") and guessed != rec["source"]:
            source_mismatch += 1

    logger.info(f"  box scale [0,1]        {'OK  all valid' if not box_problems else f'{len(box_problems)} record(s) with bad boxes'}")
    for b in box_problems[:5]:
        logger.warning(f"    {b['new_image_id']}: {'; '.join(b['problems'])}")
    logger.info(f"  captions non-empty     {'OK' if not empty_captions else f'{empty_captions} ok-record(s) with a blank caption'}")
    logger.info(f"  source vs filename     {'OK  agree' if not source_mismatch else f'{source_mismatch} disagreement(s)'}")

    missing_images: List[str] = []
    if args.check_images != "none":
        to_check = all_records if args.check_images == "all" else all_records[::50]
        for rec in to_check:
            p = rec.get("image_path")
            if not p or not Path(p).exists():
                missing_images.append(rec["new_image_id"])
        logger.info(
            f"  image paths            {'OK  all ' + str(len(to_check)) + ' exist' if not missing_images else f'{len(missing_images)} of {len(to_check)} MISSING'}"
        )
        for m in missing_images[:5]:
            bad = next(r for r in all_records if r["new_image_id"] == m)
            logger.warning(f"    {m} -> {bad.get('image_path') or '<unresolved>'}")

    logger.info("  path resolution        " + ", ".join(f"{k}={v}" for k, v in resolution_counts.most_common()))

    # ---------------------------------------------------------------- write
    corpus_path = out_dir / "proposals_all.jsonl"
    tmp = corpus_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, corpus_path)

    if not args.no_copy_sources:
        for run_dir in run_dirs:
            dest = out_dir / "source_runs" / run_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("selection.json", "annotate_manifest.json", "progress.json"):
                if (run_dir / name).exists():
                    shutil.copy2(run_dir / name, dest / name)

    fewshot_src = Path(args.fewshot_from) if args.fewshot_from else next(
        (d for d in run_dirs if (d / "fewshot.json").exists()), None
    )
    if fewshot_src and (fewshot_src / "fewshot.json").exists():
        fs_dest = out_dir / "fewshot"
        fs_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fewshot_src / "fewshot.json", fs_dest / "fewshot.json")
        img_src = fewshot_src / "fewshot_images"
        if img_src.is_dir():
            shutil.copytree(img_src, fs_dest / "fewshot_images", dirs_exist_ok=True)
        logger.info(f"  few-shot contract      copied from {fewshot_src.name}")

    # ---------------------------------------------------------------- summary
    status_totals: Counter = Counter(r["status"] or "?" for r in all_records)
    rule_totals: Counter = Counter()
    rule_by_source: Dict[str, Counter] = defaultdict(Counter)
    flagged_images = 0
    with_geometry = 0
    for rec in all_records:
        if rec["status"] != STATUS_OK:
            continue
        if rec["n_flagged"]:
            flagged_images += 1
        if rec.get("worker_machine_pairs"):
            with_geometry += 1
        for r in rec["flagged_rules"] or []:
            rule_totals[r] += 1
            rule_by_source[rec["source"] or "?"][r] += 1

    manifest = {
        "built_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "corpus_file": corpus_path.name,
        "records_total": len(all_records),
        "records_by_status": dict(status_totals),
        "records_by_source": dict(Counter(r["source"] or "?" for r in all_records)),
        "records_by_run": dict(Counter(r["run"] for r in all_records)),
        "ok_records_with_a_flag": flagged_images,
        "proposals_by_rule": dict(rule_totals),
        "proposals_by_rule_and_source": {k: dict(v) for k, v in rule_by_source.items()},
        "ok_records_with_mocs_geometry": with_geometry,
        "teacher_model": args.model,
        "prompt_sha256": list(prompts)[0] if prompt_ok else sorted(prompts),
        "prompt_identical_across_runs": prompt_ok,
        "images_root_val": root_val,
        "images_root_test": root_test,
        "path_resolution_counts": dict(resolution_counts),
        "raw_output_kept_for": args.raw,
        "checks": {
            "prompt_identical": prompt_ok,
            "duplicate_ids": len(duplicates),
            "duplicate_examples": duplicates[:20],
            "record_count_matches": count_ok,
            "bad_boxes": len(box_problems),
            "bad_box_examples": box_problems[:20],
            "empty_captions": empty_captions,
            "source_filename_mismatch": source_mismatch,
            "images_checked": args.check_images,
            "missing_images": len(missing_images),
            "missing_image_examples": missing_images[:20],
        },
        "runs": provenance,
        "field_order": list(FIELD_ORDER),
        "box_scale": "[0,1] xyxy, matching datasets/processed and build_target_json",
    }
    _write_json_atomic(manifest, out_dir / "combine_manifest.json")

    logger.info("")
    logger.info(f"Wrote {len(all_records)} record(s) -> {corpus_path}")
    logger.info(f"       manifest            -> {out_dir / 'combine_manifest.json'}")
    logger.info("")
    logger.info("CORPUS SUMMARY")
    logger.info("  by status: " + ", ".join(f"{k}={v}" for k, v in status_totals.most_common()))
    logger.info(f"  ok records with >=1 proposed rule: {flagged_images}")
    logger.info("  proposals by rule: " + ", ".join(f"{r}={rule_totals[r]}" for r in RULES))
    logger.info(
        f"  ok records carrying MOCS worker/machine geometry: {with_geometry} "
        "(val-sourced only -- the test split ships no boxes, so "
        "mocs_suggested_rule4_box_1000 is legitimately empty elsewhere)"
    )
    logger.info("")
    logger.info("Next: build the reviewer's work packages")
    logger.info(f"  python -m mocs_annotation.build_review --corpus {corpus_path} --out {out_dir / 'review'}")


if __name__ == "__main__":
    main()
