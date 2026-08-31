"""
Entry point: runs SFT training for the unified task.
Usage: python experiments/run_sft.py --tier 2b --variant unified-sft-v4
"""
import unsloth
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., WANDB_API_KEY, HF_TOKEN)
load_dotenv()

from core.config import load_config
from core.logging import get_logger
from data.loader import load_processed_dataset
from data.preprocessor import build_unified_sft_dataset
from data.samplers import get_resolutions
from models.sft_trainer import run_sft_unified

logger = get_logger(__name__)


def main():
    # Parse just the task arg first to load config
    parser_task = argparse.ArgumentParser(add_help=False)
    parser_task.add_argument("--task", default="unified")
    args_task, _ = parser_task.parse_known_args()

    config = load_config(task=args_task.task)
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--variant", default="unified-sft-v4")
    parser.add_argument("--task", default="unified", help="Task name: 'unified' or 'violations_only'")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from core.logging import attach_file_logger
    from core.io import get_drive_path, ensure_dir
    import time
    
    # Set up unique txt log file in the logs directory
    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{args.tier}_{args.variant}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    logger.info("Loading fully processed and sorted dataset splits...")
    splits = load_processed_dataset()
    
    from core.config import load_config
    from core.io import get_drive_path, ensure_dir
    import json
    from data.oversampling import build_oversampled_indices, build_rare_mask
    # Full merge chain (base -> model_registry -> sft -> tasks/<task>), matching GRPO and
    # the precedence documented in CLAUDE.md. This previously used
    # load_training_config("sft"), which read configs/sft.yaml alone — so a task YAML
    # could never override an SFT hyperparameter, silently and without warning.
    sft_cfg = load_config(task=args.task, training_kind="sft")

    # Dynamic tier-based learning rate scaling for SFT
    if args.tier == "4b":
        sft_cfg["learning_rate"] = min(sft_cfg.get("learning_rate", 1.0e-4), 5.0e-5)
        logger.info(f"Scaled SFT learning rate to {sft_cfg['learning_rate']} for tier 4b")
    elif args.tier == "8b":
        sft_cfg["learning_rate"] = min(sft_cfg.get("learning_rate", 1.0e-4), 2.0e-5)
        logger.info(f"Scaled SFT learning rate to {sft_cfg['learning_rate']} for tier 8b")

    logger.info("Building oversampled dataset...")
    oversample_indices, oversample_manifest = build_oversampled_indices(
        splits["train"],
        rule24_multiplier=sft_cfg.get("oversample_rule24_multiplier", 4),
        rule3_multiplier=sft_cfg.get("oversample_rule3_multiplier", 2),
    )
    
    # Save the manifest for reproducibility
    manifest_dir = ensure_dir(get_drive_path("datasets", "stats"))
    manifest_path = manifest_dir / f"oversample_manifest_{args.tier}_{args.variant}.json"
    with open(manifest_path, "w") as f:
        json.dump(oversample_manifest, f, indent=2)
    logger.info(f"Saved oversample manifest to {manifest_path}")

    train_raw_oversampled = splits["train"].select(oversample_indices)
    
    logger.info("Building rare mask for stratified sampling...")
    rare_mask = build_rare_mask(train_raw_oversampled)

    if "resolution" in train_raw_oversampled.column_names:
        train_resolutions = train_raw_oversampled["resolution"]
    else:
        logger.info("No 'resolution' column found in training dataset. Attempting to compute resolutions...")
        train_resolutions = get_resolutions(train_raw_oversampled)

    logger.info("Preprocessing datasets for SFT...")
    from data.preprocessor import build_sft_dataset
    train_ds = build_sft_dataset(train_raw_oversampled, task=args.task)
    val_ds = build_sft_dataset(splits["val"], task=args.task)

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
        rare_mask=rare_mask,
        train_resolutions=train_resolutions,
        resume=not args.no_resume,
        task=args.task,
    )

    logger.info(f"SFT run complete. Best/final checkpoint at {checkpoint_dir}")


if __name__ == "__main__":
    main()