"""
Offline script to build a balanced GRPO training pool.

Combines:
  - The full val split (never touched by SFT gradients, and never augmented)
  - Every violation image (any Rule 1/2/3/4) from the non-augmented train split
  - A matching number of safe images from the non-augmented train split,
    sampled so the WHOLE combined pool ends up ~50/50 safe/violation
    (accounting for val's own, non-balanced safe/violation mix)

Deliberately reads from datasets/processed (the pre-augmentation base), not
datasets/augmented — GRPO trains for a single epoch (see configs/grpo.yaml),
so any pixel-augmented or index-duplicated near-identical image would sit in
the same rollout pool and produce redundant, correlated reward groups instead
of independent training signal. SFT keeps using the augmented dataset as
before; this script only ever affects GRPO's pool.

Usage:
    python data/build_grpo_pool.py
"""
import json
import random

from datasets import load_from_disk, concatenate_datasets

from core.config import load_base_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)

SEED = 42


def is_violation(sample: dict) -> bool:
    return any(sample.get(f"rule_{i}_violation") is not None for i in (1, 2, 3, 4))


def split_violation_safe(hf_dataset):
    """Returns (violation_indices, safe_indices) for a HF Dataset split."""
    violation_idx, safe_idx = [], []
    for i, sample in enumerate(hf_dataset):
        if is_violation(sample):
            violation_idx.append(i)
        else:
            safe_idx.append(i)
    return violation_idx, safe_idx


def count_by_rule(hf_dataset) -> dict:
    """Per-rule violation counts + safe count for a HF Dataset split.

    Rule counts are NOT mutually exclusive (an image can violate more than
    one rule at once), so rule_1+rule_2+rule_3+rule_4 can exceed (total -
    safe) — that's expected, not a bug. See docs/stats/rule_cooc_matrix_train.csv
    for how often rules co-occur.
    """
    counts = {"rule_1": 0, "rule_2": 0, "rule_3": 0, "rule_4": 0, "safe": 0}
    for sample in hf_dataset:
        if is_violation(sample):
            for i in (1, 2, 3, 4):
                if sample.get(f"rule_{i}_violation") is not None:
                    counts[f"rule_{i}"] += 1
        else:
            counts["safe"] += 1
    counts["total"] = len(hf_dataset)
    return counts


def main():
    base_cfg = load_base_config()
    input_dir = get_drive_path(base_cfg["dataset"].get("raw_processed_subdir", "datasets/processed"))
    logger.info(f"Loading non-augmented base dataset from {input_dir}")
    ds = load_from_disk(str(input_dir))

    val_split = ds["val"]
    train_split = ds["train"]
    logger.info(f"Base train (non-augmented): {len(train_split)} | val: {len(val_split)}")

    # --- val: keep in full, just count its safe/violation mix ---
    val_violation_idx, val_safe_idx = split_violation_safe(val_split)
    n_val_violation = len(val_violation_idx)
    n_val_safe = len(val_safe_idx)
    logger.info(f"Val split: {n_val_violation} violation, {n_val_safe} safe (keeping all {len(val_split)})")

    # --- train: take EVERY violation image, no capping across rules ---
    train_violation_idx, train_safe_idx = split_violation_safe(train_split)
    n_train_violation = len(train_violation_idx)
    n_train_safe_available = len(train_safe_idx)
    logger.info(
        f"Train split (non-augmented): {n_train_violation} violation images "
        f"(taking all), {n_train_safe_available} safe images available to draw from"
    )
    train_violation_ds = train_split.select(train_violation_idx)

    # --- balance: total violations are now fixed (val + all train violations); ---
    # --- top up safe images from train until the WHOLE pool is ~50/50 ---
    total_violations = n_val_violation + n_train_violation
    safe_needed_from_train = max(0, total_violations - n_val_safe)

    if safe_needed_from_train > n_train_safe_available:
        logger.warning(
            f"Only {n_train_safe_available} safe train images available, but "
            f"{safe_needed_from_train} would be needed for an exact 50/50 balance. "
            f"Using all {n_train_safe_available} available instead — pool will lean violation-heavy."
        )
        safe_needed_from_train = n_train_safe_available

    rng = random.Random(SEED)
    sampled_safe_idx = sorted(rng.sample(train_safe_idx, safe_needed_from_train))
    train_safe_ds = train_split.select(sampled_safe_idx)

    # --- combine + shuffle ---
    grpo_pool = concatenate_datasets([val_split, train_violation_ds, train_safe_ds])
    grpo_pool = grpo_pool.shuffle(seed=SEED)

    total_safe = n_val_safe + safe_needed_from_train
    total = len(grpo_pool)

    manifest = {
        "seed": SEED,
        "source_dir": str(input_dir),
        "val_total": len(val_split),
        "val_violation": n_val_violation,
        "val_safe": n_val_safe,
        "train_violation_taken": n_train_violation,
        "train_safe_available": n_train_safe_available,
        "train_safe_taken": safe_needed_from_train,
        "pool_total": total,
        "pool_violation": total_violations,
        "pool_safe": total_safe,
        "pool_violation_pct": round(100 * total_violations / total, 2),
        "pool_safe_pct": round(100 * total_safe / total, 2),
        # Per-rule breakdown (not mutually exclusive — see count_by_rule docstring)
        "val_by_rule": count_by_rule(val_split),
        "train_available_by_rule": count_by_rule(train_split),
        "train_violations_taken_by_rule": count_by_rule(train_violation_ds),
        "pool_by_rule": count_by_rule(grpo_pool),
    }
    logger.info(f"GRPO pool composition:\n{json.dumps(manifest, indent=2)}")

    output_dir = get_drive_path(base_cfg["dataset"].get("grpo_pool_subdir", "datasets/grpo_pool"))
    ensure_dir(output_dir)
    logger.info(f"Saving GRPO pool ({total} samples) to {output_dir}")
    grpo_pool.save_to_disk(str(output_dir), max_shard_size="100MB")

    manifest_path = output_dir / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved build manifest to {manifest_path}")
    logger.info("Done. Point GRPO at this pool via data.loader.load_grpo_pool().")


if __name__ == "__main__":
    main()
