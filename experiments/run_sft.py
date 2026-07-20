"""
Entry point: runs SFT training for the unified task.
Usage: python experiments/run_sft.py --tier 2b --variant unified-sft-v1
"""
import argparse

from core.config import load_config
from core.logging import get_logger
from data.loader import load_processed_dataset
from data.preprocessor import build_unified_sft_dataset
from data.samplers import get_resolutions
from models.sft_trainer import run_sft_unified

logger = get_logger(__name__)


def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier)
    parser.add_argument("--variant", default="unified-sft-v1")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logger.info("Loading fully processed and sorted dataset splits...")
    splits = load_processed_dataset()

    if "resolution" in splits["train"].column_names:
        train_resolutions = splits["train"]["resolution"]
    else:
        train_resolutions = get_resolutions(splits["train"])

    logger.info("Preprocessing datasets for unified SFT...")
    train_ds = build_unified_sft_dataset(splits["train"])
    val_ds = build_unified_sft_dataset(splits["val"])

    if train_resolutions is not None and len(train_resolutions) != len(train_ds):
        logger.warning(
            f"Resolution count ({len(train_resolutions)}) != training samples ({len(train_ds)}). "
            f"Disabling resolution bucketing for safety."
        )
        train_resolutions = None

    logger.info(f"Starting SFT for tier: {args.tier}, variant: {args.variant}...")
    checkpoint_dir = run_sft_unified(
        tier=args.tier,
        variant=args.variant,
        train_dataset=list(train_ds),
        val_dataset=list(val_ds),
        train_resolutions=train_resolutions,
        resume=not args.no_resume,
    )

    logger.info(f"SFT run complete. Best/final checkpoint at {checkpoint_dir}")


if __name__ == "__main__":
    main()