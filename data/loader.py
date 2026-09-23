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
    if len(sample.get("worker_with_white_hard_hat") or []) > 0:
        return 4
    if len(sample.get("rebar") or []) > 0:
        return 5
    if len(sample.get("excavator") or []) > 0:
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


def load_processed_dataset(subdir: str = None) -> DatasetDict:
    """Loads the fully processed, stratified, and conversational dataset.

    Args:
        subdir: Optional dataset subdirectory override, relative to the data root
            (e.g. "datasets/processed"). Defaults to base.yaml's
            ``dataset.processed_subdir``, which — note the long-standing naming
            trap — points at datasets/AUGMENTED, not datasets/processed.

            The override exists so a task can opt out of the violation-oversampled
            augmented set: object_only and caption_only set
            ``sft_dataset_subdir: datasets/processed`` in their task YAML, because
            the augmentation duplicates images by rare *violation* rule (rule_4
            x16, rule_2 x12, rule_3 x6) and those duplicates carry identical boxes
            and identical captions — no class rebalancing for those tasks, just
            overfitting pressure. unified and violations_only pass nothing and are
            unaffected.
    """
    base_cfg = load_base_config()
    resolved = subdir or base_cfg["dataset"].get("processed_subdir", "datasets/processed")
    processed_path = get_drive_path(resolved)

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


GRPO_POOL_SUBDIR_DEFAULT = "datasets/grpo_pool"


def resolve_grpo_pool_subdir(subdir: str = None) -> str:
    """The GRPO pool directory that will actually be used, as a relative subdir.

    Precedence: explicit ``subdir`` -> ``base.yaml``'s ``dataset.grpo_pool_subdir`` ->
    :data:`GRPO_POOL_SUBDIR_DEFAULT`.

    Split out of :func:`load_grpo_pool` for two reasons: the trainer records the
    EFFECTIVE value in ``run_manifest.json`` (so a run's data provenance is recoverable
    from disk without re-deriving the precedence chain), and the precedence itself is
    then unit-testable without a dataset on disk.
    """
    if subdir:
        return subdir          # short-circuit: do not parse base.yaml just to discard it
    base_cfg = load_base_config()
    return base_cfg["dataset"].get("grpo_pool_subdir", GRPO_POOL_SUBDIR_DEFAULT)


def load_grpo_pool(subdir: str = None):
    """Loads the pre-built, balanced GRPO training pool.

    Built once, offline, by data/build_grpo_pool.py from the non-augmented
    base dataset (datasets/processed): the full val split + every train
    violation image + a matching count of train safe images, so the pool is
    ~50/50 safe/violation. Deliberately excludes the pixel-augmented /
    oversampled duplicate rows that datasets/augmented (used by SFT) has,
    since GRPO trains for a single epoch (configs/grpo.yaml) and duplicate
    or near-duplicate prompts in one pass produce redundant reward groups.

    Args:
        subdir: Optional pool directory override, relative to the data root
            (e.g. "datasets/grpo_pool_v3"). Resolution order is
            **argument -> base.yaml's dataset.grpo_pool_subdir -> the literal
            default**, so passing nothing reproduces the historical behaviour
            byte-for-byte.

            This parameter exists because the key alone did not work. `base.yaml`
            has carried `dataset.grpo_pool_subdir` all along, but this function
            read it from ``load_base_config()`` only -- never from the merged task
            config -- so a task YAML setting it was **silently ignored**: the key
            existed, looked overridable, and was not. A `violations_think` run
            would have trained on the OLD pool while its manifest claimed the new
            one. Same failure shape as every entry in CLAUDE.md's ghost-variable
            table; mirrors ``load_processed_dataset``'s long-standing pattern.

            Callers that pass it: ``models/grpo_trainer.py`` (from
            ``cfg.get("grpo_pool_subdir")``, itself overridable by
            ``--grpo_pool_subdir``) and ``scripts/validate_rewards.py``
            (``--pool-stats``). ``scripts/preflight_grpo.py`` deliberately does NOT --
            it validates reward assembly and prompt length against
            ``load_processed_dataset()``, never the pool, so a missing or malformed
            pool is not caught there; it surfaces as this function's
            ``FileNotFoundError`` at the start of GRPO.

    Returns:
        A flat (non-split) HF Dataset — every row in it is already selected
        for GRPO training, no further filtering/oversampling needed.
    """
    resolved = resolve_grpo_pool_subdir(subdir)
    pool_path = get_drive_path(resolved)

    if not Path(pool_path).exists():
        raise FileNotFoundError(
            f"No GRPO pool found at {pool_path}. Run `python data/build_grpo_pool.py` first."
        )

    logger.info(f"Loading GRPO training pool from disk: {pool_path}")
    dataset = load_from_disk(str(pool_path))
    logger.info(f"Loaded GRPO pool: {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    splits = load_processed_dataset()
    for name, split in splits.items():
        print(f"{name}: {len(split)} samples")