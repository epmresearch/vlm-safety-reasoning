"""
Loads the ConstructionSite 10k dataset from HuggingFace.

Provides two loading functions:
  - load_construction_dataset(): Returns the raw DatasetDict (train + test).
  - load_dataset_splits(): Returns train/val/test with a validation split
    carved from training data (250 samples, seed=42).
"""
from datasets import load_dataset, DatasetDict

from core.config import load_base_config
from core.constants import VALIDATION_SPLIT_SIZE, VALIDATION_SPLIT_SEED
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


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


def load_dataset_splits(
    val_size: int = VALIDATION_SPLIT_SIZE,
    seed: int = VALIDATION_SPLIT_SEED,
) -> dict:
    """Loads dataset and carves a validation split from training data.

    Args:
        val_size: Number of samples for validation (default: 250).
        seed: Random seed for reproducible splitting.

    Returns:
        Dict with keys "train", "val", "test", each a HF Dataset.
    """
    ds = load_construction_dataset()

    # Carve validation set from training data
    train_val = ds["train"].train_test_split(test_size=val_size, seed=seed)
    train_split = train_val["train"]
    val_split = train_val["test"]
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


if __name__ == "__main__":
    splits = load_dataset_splits()
    for name, split in splits.items():
        print(f"{name}: {len(split)} samples")