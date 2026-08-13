"""
Entry point: runs GRPO/GSPO training on top of an SFT checkpoint.
Usage: python experiments/run_grpo.py --tier 2b --variant unified-grpo-v1
"""
import argparse
import time
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., WANDB_API_KEY, HF_TOKEN)
load_dotenv()

from core.config import load_config
from core.logging import get_logger, attach_file_logger
from core.io import get_drive_path, ensure_dir
from models.grpo_trainer import run_grpo

logger = get_logger(__name__)


def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--variant", default="unified-grpo-v1", help="Variant name for GRPO")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for debugging")
    parser.add_argument("--sft_variant", default="unified-sft-v1", help="SFT variant name to load as starting adapter")
    args = parser.parse_args()

    # Set up unique txt log file in the logs directory
    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_grpo_{args.tier}_{args.variant}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    logger.info(f"Starting GRPO run for tier: {args.tier}, variant: {args.variant}, sft_variant: {args.sft_variant}")

    # Build the path to the best SFT adapter for this tier
    adapter_path = str(get_drive_path("checkpoints", f"qwen3vl-{args.tier}", args.sft_variant, "best"))
    logger.info(f"Using explicitly specified SFT adapter path: {adapter_path}")

    # Run the GRPO training
    checkpoint_dir = run_grpo(
        task="unified",  # The task config name used for training
        model_id=args.tier,
        variant_name=args.variant,
        max_samples=args.max_samples,
        adapter_path=adapter_path,
    )

    logger.info(f"GRPO run complete. Final checkpoint saved at {checkpoint_dir}")


if __name__ == "__main__":
    main()