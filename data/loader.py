"""
Loads the ConstructionSite 10k dataset from HuggingFace.

Provides two loading functions:
  - load_construction_dataset(): Returns the raw DatasetDict (train + test).
  - load_dataset_splits(): Returns train/val/test with a validation split
    carved from training data (250 samples, seed=42).
"""
from datasets import load_dataset, load_from_disk, DatasetDict, ClassLabel
from pathlib import Path
from core.config import load_base_config
from core.constants import VALIDATION_SPLIT_SIZE, VALIDATION_SPLIT_SEED
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

from typing import Any, Dict

logger = get_logger(__name__)


def _compute_stratum(sample: Dict[str, Any]) -> int:
    """Computes a stratum ID for a sample based on the rarest class present.
    
    This ensures that extremely rare classes (like Rule 2 and Rule 4) are 
    evenly distributed between train and validation splits.
    Priority is given from rarest to most common.
    """
    if sample.get("rule_4_violation") is not None:
        return 0
    if sample.get("rule_2_violation") is not None:
        return 1
    if sample.get("rule_3_violation") is not None:
        return 2
    if sample.get("rule_1_violation") is not None:
        return 3
    if len(sample.get("worker_with_white_hard_hat", [])) > 0:
        return 4
    if len(sample.get("rebar", [])) > 0:
        return 5
    if len(sample.get("excavator", [])) > 0:
        return 6
    return 7


def create_stratified_val_split(hf_dataset, val_size: float = 0.1, seed: int = 42):
    """
    Creates a stratified train/val split from a HuggingFace dataset split.
    Uses the rarest class present in each sample to create strata.
    """
    logger.info("Computing strata for balanced train/val split...")

    cols_needed = [c for c in hf_dataset.column_names if c != "image"]

    def add_stratum(*columns):
        batch = dict(zip(cols_needed, columns))
        num_rows = len(columns[0])
        strata = []
        for i in range(num_rows):
            sample = {k: v[i] for k, v in batch.items()}
            strata.append(_compute_stratum(sample))
        return {"stratum": strata}

    stratified_ds = hf_dataset.map(
        add_stratum,
        batched=True,
        desc="Adding strata",
        input_columns=cols_needed,
    )

    num_strata = 8  # matches _compute_stratum()'s return range (0-7)
    stratified_ds = stratified_ds.cast_column(
        "stratum", ClassLabel(names=[str(i) for i in range(num_strata)])
    )

    # Perform the stratified split
    splits = stratified_ds.train_test_split(
        test_size=val_size,
        stratify_by_column="stratum",
        seed=seed
    )

    train_split = splits["train"].remove_columns(["stratum"])
    val_split = splits["test"].remove_columns(["stratum"])

    logger.info(f"Stratified split complete: Train={len(train_split)}, Val={len(val_split)}")
    return train_split, val_split


def load_construction_dataset() -> DatasetDict:
    """Loads the raw dataset from HuggingFace with native train/test split.

    Returns:
        DatasetDict with "train" (7009) and "test" (3004) splits.
    """
    base_cfg = load_base_config()
    hf_repo = base_cfg["dataset"]["hf_repo"]
    cache_dir = get_drive_path(base_cfg["dataset"]["raw_cache_subdir"])
    ensure_dir(cache_dir)

    logger.info(f"Loading dataset '{hf_repo}' with cache_dir={cache_dir}")
    dataset = load_dataset(hf_repo, cache_dir=str(cache_dir))

    if "train" in dataset:
        logger.info(f"Train split size: {len(dataset['train'])}")
    if "test" in dataset:
        logger.info(f"Test split size: {len(dataset['test'])}")

    return dataset


def load_cleaned_construction_dataset() -> DatasetDict:
    """Loads the CLEANED dataset previously saved via dataset.save_to_disk()
    after manual annotation fixes were applied.
    This is the dataset that should be used for fine-tuning and evaluation.
    """
    base_cfg = load_base_config()
    cleaned_path = get_drive_path(base_cfg["dataset"]["cleaned_subdir"])

    if not Path(cleaned_path).exists():
        raise FileNotFoundError(
            f"No cleaned dataset found at {cleaned_path}. "
            f"Run the data-prep notebook's save_to_disk step first, "
            f"or check 'dataset.cleaned_subdir' in your config."
        )

    logger.info(f"Loading cleaned dataset from disk: {cleaned_path}")
    dataset = load_from_disk(str(cleaned_path))

    if "train" in dataset:
        logger.info(f"Train split size: {len(dataset['train'])}")
    if "test" in dataset:
        logger.info(f"Test split size: {len(dataset['test'])}")

    return dataset



def load_dataset_splits(
    val_size: float = VALIDATION_SPLIT_SIZE,
    seed: int = VALIDATION_SPLIT_SEED,
) -> dict:
    """Loads dataset and carves a validation split from training data.

    Args:
        val_size: Proportion of samples for validation (default: 0.1).
        seed: Random seed for reproducible splitting.

    Returns:
        Dict with keys "train", "val", "test", each a HF Dataset.
    """
    ds = load_cleaned_construction_dataset()

    # Carve validation set from training data using stratified splitting
    train_split, val_split = create_stratified_val_split(ds["train"], val_size=val_size, seed=seed)
    test_split = ds["test"]

    logger.info(
        f"Dataset splits: train={len(train_split)}, "
        f"val={len(val_split)}, test={len(test_split)}"
    )

    return {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }


def load_processed_dataset() -> DatasetDict:
    """Loads the fully processed, stratified, and conversational dataset
    from the processed_subdir. This should be used for all training
    and baseline inference steps.
    """
    base_cfg = load_base_config()
    processed_path = get_drive_path(base_cfg["dataset"].get("processed_subdir", "datasets/processed"))

    if not Path(processed_path).exists():
        raise FileNotFoundError(
            f"No processed dataset found at {processed_path}. "
            f"Please run your data prep notebook to save the dataset first."
        )

    logger.info(f"Loading fully processed dataset from disk: {processed_path}")
    dataset = load_from_disk(str(processed_path))

    for split_name in dataset.keys():
        logger.info(f"Loaded processed '{split_name}' split: {len(dataset[split_name])} samples")

    return dataset


if __name__ == "__main__":
    splits = load_processed_dataset()
    for name, split in splits.items():
        print(f"{name}: {len(split)} samples")