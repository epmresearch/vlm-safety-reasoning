#!/usr/bin/env python3
"""
Picks which MOCS images to send to the annotator, and writes selection.json.

Separated from annotate.py on purpose: selection is deterministic, CPU-only, takes
seconds, and is the thing you will want to inspect and re-run with different quotas.
The GPU job then just consumes the file. It also means a resumed annotation run
covers exactly the same images as the first attempt.

BUCKETS, and what each is worth (pool sizes measured on the real
annotation_val.json, 4,000 images):

    rule_4_geometric         1,253 (31.3%)  a Worker box overlaps a machine box
                                            expanded by 25% -- a real geometric
                                            shortlist, and both boxes are human
                                            annotated
    rule_2_height_prior        ~600         Worker + (Static crane | Crane |
                                            Hanging head | Pile driving). A WEAK
                                            prior: MOCS has no scaffold, ladder or
                                            harness category, so height work cannot
                                            be mined, only guessed at
    rule_3_excavation_prior  ~1,900         Excavator or Bulldozer present, i.e.
                                            digging is happening, so an open
                                            excavation probably exists. Also weak --
                                            no trench or guard-rail category
    random_control             all          An unbiased sample. Keep it: it is the
                                            only way to tell whether the mined
                                            buckets actually beat chance, and
                                            therefore whether the mining is worth
                                            anything

Runs anywhere -- needs only the MOCS json. No GPU, no dataset, no unsloth.

Usage:
    python -m mocs_annotation.select_images \
        --annotations $VLM_DATA_ROOT/datasets/filtered/annotation_val.json \
        --images-root $VLM_DATA_ROOT/datasets/filtered/instances_val \
        --out        $VLM_DATA_ROOT/datasets/mocs_annotation/selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.logging import get_logger
from mocs_annotation.mocs_data import (
    BUCKETS,
    MACHINE_CATEGORIES,
    load_mocs,
    merge_splits,
    select_candidates,
)

logger = get_logger(__name__)

# Sized against the MEASURED pools to land near 2,000 images.
#
# Pools, measured on annotation_val.json at --max-aspect-ratio 3.0:
#     rule_2_height_prior         467   <- THE BINDING CONSTRAINT
#     rule_4_geometric          1,253
#     rule_3_excavation_prior   1,945
#     random_control            3,999
#
# rule_2's pool is only 467 images, so asking for 500 gets you 467 and the run totals
# ~2,017 rather than 2,000. That is not a bug to tune away, it is the real ceiling:
# MOCS has no scaffold, ladder or harness category, so height work can only be guessed
# at from crane/hanging-hook presence. If the rule_2 harvest comes up short, the fix is
# NOT a bigger quota here -- it is to scan more images, and for rule_2 specifically the
# 18,264-image image_info_test.json is fair game, because nothing is being mined from
# boxes for that rule anyway.
DEFAULT_QUOTAS = {
    "rule_4_geometric": 750,
    "rule_2_height_prior": 500,
    "rule_3_excavation_prior": 600,
    "random_control": 200,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, nargs="+",
                     help="One or more MOCS json files. annotation_val.json carries boxes "
                          "(so it can be mined geometrically); image_info_test.json is "
                          "image metadata only and contributes to random_control alone. "
                          "Pass both to draw from all 22,264 images")
    ap.add_argument("--images-root", required=True, nargs="+",
                     help="Image directory for each --annotations file, in the same order "
                          "(e.g. .../instances_val .../instances_test). Carried per record, "
                          "because a mixed pool jpgs live in different directories")
    ap.add_argument("--exclude", nargs="*", default=[],
                     help="Paths to a previous selection.json and/or proposals.jsonl. Every "
                          "image id found in them is dropped from all buckets BEFORE quotas "
                          "are filled, so a follow-up run cannot re-annotate an image or "
                          "produce a second conflicting record for it")
    ap.add_argument("--out", required=True, help="Path to write selection.json")
    for bucket in BUCKETS:
        ap.add_argument(f"--{bucket.replace('_', '-')}", type=int,
                         default=DEFAULT_QUOTAS.get(bucket, 0),
                         dest=bucket, help=f"How many images to draw from the {bucket} bucket")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin-frac", type=float, default=0.25,
                     help="Operating-radius margin as a fraction of the machine box's longest "
                          "side (0.25 -> 1,253 candidate images; 0.0 -> 1,064; 0.40 -> 1,320)")
    ap.add_argument("--machine-categories", nargs="+", default=list(MACHINE_CATEGORIES),
                     help="MOCS categories treated as having an operating radius")
    ap.add_argument("--max-megapixels", type=float, default=None,
                     help="Drop images above this. Usually unnecessary -- annotate.py caps "
                          "pixels at load time -- but MOCS val does contain a 14.63 MP image")
    ap.add_argument("--max-aspect-ratio", type=float, default=3.0,
                     help="Drop extreme panoramas, which tokenize badly. MOCS val has one "
                          "image at 4.55")
    args = ap.parse_args()

    if len(args.annotations) != len(args.images_root):
        raise SystemExit(
            f"--annotations has {len(args.annotations)} entries but --images-root has "
            f"{len(args.images_root)}; they are paired positionally and must match."
        )

    splits = []
    for ann, root in zip(args.annotations, args.images_root):
        root_path = Path(root)
        if not root_path.is_dir():
            raise SystemExit(f"--images-root is not a directory: {root_path}")
        sp = load_mocs(ann, images_root=str(root_path))
        logger.info(
            f"Loaded {len(sp)} images from {Path(sp.path).name} -> {root_path} "
            f"(annotations present: {sp.has_annotations})"
        )
        if not sp.has_annotations:
            logger.info(
                f"  {Path(sp.path).name} carries no boxes, so its images can only enter "
                "random_control -- the geometric miner and both weak priors need geometry."
            )
        splits.append(sp)

    split = merge_splits(*splits)
    if len(splits) > 1:
        logger.info(f"Merged {len(splits)} source(s) -> {len(split)} unique images")

    exclude_ids = set()
    for path in args.exclude:
        f = Path(path)
        if not f.exists():
            raise SystemExit(f"--exclude file not found: {f}")
        if f.suffix == ".jsonl":
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            exclude_ids.add(json.loads(line)["new_image_id"])
                        except Exception:
                            pass
        else:
            with open(f, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            rows = payload.get("records", []) if isinstance(payload, dict) else payload
            for r in rows:
                if isinstance(r, dict) and "new_image_id" in r:
                    exclude_ids.add(r["new_image_id"])
        logger.info(f"Exclusions after {f.name}: {len(exclude_ids)} id(s)")

    quotas = {b: getattr(args, b) for b in BUCKETS}
    records, manifest = select_candidates(
        split,
        quotas=quotas,
        seed=args.seed,
        machine_categories=args.machine_categories,
        margin_frac=args.margin_frac,
        max_megapixels=args.max_megapixels,
        max_aspect_ratio=args.max_aspect_ratio,
        exclude_ids=exclude_ids,
    )

    # Fail now, not after a queue wait, if an image root does not match its json.
    missing = [
        f"{r['images_root']}/{r['file_name']}"
        for r in records[:80]
        if not (Path(r["images_root"]) / r["file_name"]).exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of the first 80 selected files do not exist on disk. "
            f"First few: {missing[:5]}. Check each --images-root is paired with the right "
            "--annotations file (instances_val goes with annotation_val.json)."
        )

    # Kept for backward compatibility with annotate.py fallback; the per-record
    # images_root is what a mixed-source selection actually uses.
    manifest["images_root"] = str(Path(args.images_root[0]))
    manifest["sources"] = [
        {"annotations": a, "images_root": r} for a, r in zip(args.annotations, args.images_root)
    ]
    payload = {"manifest": manifest, "records": records}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)

    logger.info("Bucket pool sizes (before quota):")
    for b, n in manifest["bucket_pool_sizes"].items():
        logger.info(f"  {b:<26}{n:>6}")
    logger.info("Selected:")
    for b, n in manifest["quotas_filled"].items():
        logger.info(f"  {b:<26}{n:>6}  (requested {quotas.get(b, 0)})")
    from collections import Counter
    logger.info("By source: " + ", ".join(
        f"{k}={v}" for k, v in Counter(r["source"] for r in records).most_common()))
    if manifest.get("excluded_from_pools"):
        logger.info(
            f"Excluded {manifest['excluded_from_pools']} already-annotated image slot(s) "
            f"from the pools ({manifest['excluded_ids_supplied']} ids supplied)")
    logger.info(f"TOTAL selected: {manifest['selected_total']} -> {out}")
    n_pairs = sum(len(r["worker_machine_pairs"]) for r in records)
    logger.info(f"Worker/machine pairs carried on the selection: {n_pairs}")


if __name__ == "__main__":
    main()
