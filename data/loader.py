"""
Loads the ConstructionSite 10k dataset from HuggingFace, using its native train/test split AS-IS. No re-splitting, no modification.
"""
from datasets import load_dataset, DatasetDict

from core.config import load_base_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def load_construction_dataset() -> DatasetDict:
    base_cfg = load_base_config()
    hf_repo = base_cfg["dataset"]["hf_repo"]
    cache_dir = get_drive_path(base_cfg["dataset"]["raw_cache_subdir"])
    ensure_dir(cache_dir)

    logger.info(f"Loading dataset '{hf_repo}' with cache_dir={cache_dir}")
    dataset = load_dataset(hf_repo, cache_dir=str(cache_dir))

    # Sanity check: confirm native split sizes match documentation (7009/3004).
    if "train" in dataset:
        logger.info(f"Train split size: {len(dataset['train'])}")
    if "test" in dataset:
        logger.info(f"Test split size: {len(dataset['test'])}")

    return dataset


if __name__ == "__main__":
    ds = load_construction_dataset()
    print(ds)