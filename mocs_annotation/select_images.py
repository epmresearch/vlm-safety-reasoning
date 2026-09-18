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
    ap.add_argument("--annotations", required=True,
                     help="MOCS annotation_val.json (has boxes) or image_info_test.json (no boxes)")
    ap.add_argument("--images-root", required=True,
                     help="Directory holding the .jpg files, e.g. .../instances_val")
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

    images_root = Path(args.images_root)
    if not images_root.is_dir():
        raise SystemExit(f"--images-root is not a directory: {images_root}")

    split = load_mocs(args.annotations)
    logger.info(
        f"Loaded {len(split)} images from {split.path.name} "
        f"(annotations present: {split.has_annotations}; categories: {len(split.category_names)})"
    )
    if not split.has_annotations:
        logger.warning(
            "This file carries NO annotations, so only the random_control bucket can be "
            "filled -- the rule_4 geometric miner and both weak priors need boxes. "
            "MOCS's test split is image_info only (18,264 images); use annotation_val.json "
            "for anything that needs mining."
        )

    quotas = {b: getattr(args, b) for b in BUCKETS}
    records, manifest = select_candidates(
        split,
        quotas=quotas,
        seed=args.seed,
        machine_categories=args.machine_categories,
        margin_frac=args.margin_frac,
        max_megapixels=args.max_megapixels,
        max_aspect_ratio=args.max_aspect_ratio,
    )

    # Fail now, not after a queue wait, if the image root does not match the json.
    missing = [r["file_name"] for r in records[:50] if not (images_root / r["file_name"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of the first 50 selected files are absent from {images_root}. "
            f"First few: {missing[:5]}. Check --images-root matches --annotations "
            "(instances_val goes with annotation_val.json)."
        )

    manifest["images_root"] = str(images_root)
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
    logger.info(f"TOTAL selected: {manifest['selected_total']} -> {out}")
    n_pairs = sum(len(r["worker_machine_pairs"]) for r in records)
    logger.info(f"Worker/machine pairs carried on the selection: {n_pairs}")


if __name__ == "__main__":
    main()
