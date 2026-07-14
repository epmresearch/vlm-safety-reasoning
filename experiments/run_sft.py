"""
Entry point: runs SFT training for the unified task.
Usage: python experiments/run_sft.py --tier 2b --variant unified-sft-v1
"""
import argparse

from core.constants import DEFAULT_MODEL_TIER
from core.logging import get_logger
from data.loader import load_dataset_splits
from data.preprocessor import build_unified_sft_dataset
from models.sft_trainer import run_sft_unified

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=DEFAULT_MODEL_TIER, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--variant", default="unified-sft-v1", help="Variant name for SFT")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    args = parser.parse_args()

    # Load dataset
    logger.info("Loading dataset splits...")
    splits = load_dataset_splits()
    
    # Preprocess
    logger.info("Preprocessing datasets for unified SFT...")
    train_ds = build_unified_sft_dataset(splits["train"])
    val_ds = build_unified_sft_dataset(splits["val"])

    logger.info(f"Starting SFT for tier: {args.tier}, variant: {args.variant}...")
    checkpoint_dir = run_sft_unified(
        tier=args.tier,
        variant=args.variant,
        train_dataset=list(train_ds),
        val_dataset=list(val_ds),
        resume=not args.no_resume,
    )
    
    logger.info(f"SFT run complete. Checkpoint saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()