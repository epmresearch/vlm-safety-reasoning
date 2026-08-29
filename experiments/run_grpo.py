"""
Entry point: runs GRPO/GSPO training on top of an SFT checkpoint.
Usage: python experiments/run_grpo.py --tier 2b --variant unified-grpo-v4
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
    parser.add_argument("--variant", default="unified-grpo-v4", help="Variant name for GRPO")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for debugging")
    parser.add_argument("--sft_variant", default="unified-sft-v4", help="SFT variant name to load as starting adapter")
    parser.add_argument("--adapter_path", default=None, help="Explicit full path to adapter. With --base_model_override, this continues that adapter on top of the merged base (epoch chaining); without it, this is a raw-base continuation (see --allow_unmerged_reference).")
    parser.add_argument("--base_model_override", default=None, help="If set, loads THIS path as the base model instead of the HF model (use with merged SFT model for correct KL reference)")
    parser.add_argument("--allow_unmerged_reference", action="store_true", help="Bypass the merged-base-model safety check and proceed without --base_model_override. NOT recommended: TRL's KL reference will be the raw pretrained base, not your SFT policy (the original reference-model bug). Use only for an intentional ablation.")
    parser.add_argument("--task", default="unified", help="Task name: 'unified' or 'violations_only'")
    args = parser.parse_args()

    # Set up unique txt log file in the logs directory
    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_grpo_{args.tier}_{args.variant}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    logger.info(f"Starting GRPO run for tier: {args.tier}, variant: {args.variant}, sft_variant: {args.sft_variant}")

    # If a merged base model is provided, we load that as the base — TRL's KL reference
    # (computed via disable_adapter()) then correctly points at the SFT policy.
    # adapter_path may still be set alongside it, to continue a prior GRPO adapter
    # (epoch chaining) rather than starting a fresh LoRA.
    if args.base_model_override:
        adapter_path = args.adapter_path  # None -> fresh LoRA; set -> continue this adapter
        if adapter_path:
            logger.info(f"Using MERGED base model: {args.base_model_override} with continuation adapter: {adapter_path}")
        else:
            logger.info(f"Using MERGED base model: {args.base_model_override} (fresh LoRA, adapter_path=None for correct KL reference)")
    elif args.allow_unmerged_reference:
        if args.adapter_path:
            adapter_path = args.adapter_path
            logger.warning(f"--allow_unmerged_reference set: using explicit adapter path WITHOUT a merged base: {adapter_path}. KL reference will be the RAW base model, not your SFT policy.")
        else:
            adapter_path = str(get_drive_path("checkpoints", f"qwen3vl-{args.tier}", args.sft_variant, "final"))
            logger.warning(f"--allow_unmerged_reference set: using SFT variant adapter path WITHOUT a merged base: {adapter_path}. KL reference will be the RAW base model, not your SFT policy.")
    else:
        raise SystemExit(
            "Refusing to start GRPO without --base_model_override: without a merged SFT base, "
            "TRL's KL reference model is the raw pretrained base, not your SFT policy (this was the "
            "original reference-model bug — see docs/Diagnosis/VO V4/grpo_image_missing.md). "
            "Pass --base_model_override <merged model path> (see scripts/merge_sft_adapter.py), "
            "or pass --allow_unmerged_reference to proceed anyway (e.g. for an intentional ablation)."
        )

    # Run the GRPO training
    checkpoint_dir = run_grpo(
        task=args.task,
        model_id=args.tier,
        variant_name=args.variant,
        max_samples=args.max_samples,
        adapter_path=adapter_path,
        base_model_override=args.base_model_override,
    )

    logger.info(f"GRPO run complete. Final checkpoint saved at {checkpoint_dir}")


if __name__ == "__main__":
    main()