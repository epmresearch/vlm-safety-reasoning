"""
Entry point: runs GRPO/GSPO training on top of an SFT checkpoint, then evaluates.
Usage: python experiments/run_grpo.py --tier 2b --variant unified-grpo-v1
"""
import argparse

from core.constants import DEFAULT_MODEL_TIER
from core.logging import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=DEFAULT_MODEL_TIER, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--variant", default="unified-grpo-v1", help="Variant name for GRPO")
    args = parser.parse_args()

    # NOTE: GRPO integration for the unified approach is a WIP for future phases.
    # We will update this module when models/grpo_trainer.py is refactored for the unified prompt.
    logger.info(f"GRPO run stub. Tier: {args.tier}, Variant: {args.variant}")
    logger.info("Detailed GRPO implementation is deferred to a future phase.")

if __name__ == "__main__":
    main()