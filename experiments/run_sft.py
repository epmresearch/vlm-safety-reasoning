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
from core.constants import VALID_TASKS
from core.logging import get_logger
from core.tasks import CAP_VIOLATIONS, task_has
from data.loader import load_processed_dataset
from data.samplers import get_resolutions
from models.sft_trainer import run_sft_unified

logger = get_logger(__name__)


def main():
    # Parse just the task arg first to load config
    parser_task = argparse.ArgumentParser(add_help=False)
    parser_task.add_argument("--task", default="unified", choices=VALID_TASKS)
    args_task, _ = parser_task.parse_known_args()

    config = load_config(task=args_task.task)
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    # Required, not defaulted: the old default ("unified-sft-v4") silently wrote a
    # stale, unversioned variant if a caller forgot the flag.
    parser.add_argument("--variant", required=True, help="Variant name, e.g. oo-sft-8b-v1")
    parser.add_argument(
        "--task", default="unified", choices=VALID_TASKS,
        help="Task to train. Selects the prompt, the SFT target format, and the "
             "input dataset subdir.",
    )
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

    from core.config import load_config
    from core.io import get_drive_path, ensure_dir
    import json
    from data.oversampling import (build_oversampled_indices,
                                   build_rare_mask_for_task)
    # Full merge chain (base -> model_registry -> sft -> tasks/<task>), matching GRPO and
    # the precedence documented in CLAUDE.md. This previously used
    # load_training_config("sft"), which read configs/sft.yaml alone — so a task YAML
    # could never override an SFT hyperparameter, silently and without warning.
    sft_cfg = load_config(task=args.task, training_kind="sft")

    # Loaded AFTER the config so a task YAML can redirect the SFT input split.
    sft_subdir = sft_cfg.get("sft_dataset_subdir")
    if sft_subdir:
        logger.info(
            f"Task '{args.task}' overrides the SFT input dataset to {sft_subdir!r} "
            "(see configs/tasks/%s.yaml)." % args.task
        )
    logger.info("Loading fully processed and sorted dataset splits...")
    splits = load_processed_dataset(subdir=sft_subdir)

    # NO tier-based learning-rate scaling. The LR is whatever the merged config says
    # (configs/sft.yaml: 1.0e-4) at every tier, deliberately.
    #
    # There used to be a clamp here (4b -> 5e-5, 8b -> 2e-5). It was dead code for the
    # whole life of the project, because run_sft.py built sft_cfg and then did not pass
    # it to run_sft_unified() -- so every tier actually trained at 1.0e-4 while the log
    # line claimed otherwise. Fixing that plumbing bug would have activated the clamp
    # for the first time, which is a silent, uncontrolled change to the 4b/8b runs, so
    # the clamp is removed instead. Three reasons:
    #
    #   1. It confounds the tier comparison. This repo runs 3 tiers to say something
    #      about model scale. If 8b trains at 1/5 the LR of 2b and scores worse, there
    #      is no way to attribute that to scale rather than to undertraining.
    #   2. The step budget cannot absorb it. At 1e-4 eval loss was still improving to
    #      ~step 250 of 512, i.e. convergence used about half the budget. At 2e-5 the
    #      same descent needs on the order of 5x more steps than exist, so 8b would be
    #      stopped mid-descent and reported as the model's ceiling.
    #   3. The instinct is a full-fine-tuning one. LoRA is far less LR-sensitive to
    #      model scale -- QLoRA tapered only 2x (2e-4 -> 1e-4) across a 9x parameter
    #      range. A 5x taper across 4x is out of line with that, and 1e-4 is already
    #      empirically stable at 8b in this exact setup (eval loss ~0.055).
    #
    # If a future run needs a per-tier LR, put it in the tier's block in
    # model_registry.yaml so it is declared configuration rather than a hidden
    # override, and keep the taper within ~2x.

    # Rare-rule oversampling and the rare mask that drives the stratified sampler are
    # both defined purely by which *violation* rules a sample triggers. For a task
    # that never predicts violations they rebalance nothing, so they are gated on the
    # capability rather than applied blindly.
    trains_violations = task_has(args.task, CAP_VIOLATIONS)

    if trains_violations:
        logger.info("Building oversampled dataset...")
        oversample_indices, oversample_manifest = build_oversampled_indices(
            splits["train"],
            rule24_multiplier=sft_cfg.get("oversample_rule24_multiplier", 4),
            rule3_multiplier=sft_cfg.get("oversample_rule3_multiplier", 2),
        )
        manifest_dir = ensure_dir(get_drive_path("datasets", "stats"))
        manifest_path = manifest_dir / f"oversample_manifest_{args.tier}_{args.variant}.json"
        with open(manifest_path, "w") as f:
            json.dump(oversample_manifest, f, indent=2)
        logger.info(f"Saved oversample manifest to {manifest_path}")
        train_raw_oversampled = splits["train"].select(oversample_indices)
    else:
        logger.info(
            f"Task '{args.task}' does not predict rule violations - skipping rare-rule "
            "oversampling (it is defined purely by which violation rules a sample trips)."
        )
        train_raw_oversampled = splits["train"]

    # Stratified sampling is a SEPARATE question from oversampling, and applies to any
    # task with a rare target -- not just violation tasks. build_rare_mask_for_task
    # picks the axis from the task's capabilities: rules 2/3/4 for violation tasks
    # (unchanged), rebar + white-hard-hat workers for object_only. It returns None for
    # caption_only, which has no rare class, and the sampler is then skipped.
    #
    # Cheap in the wrong direction: stratifying when it is not needed costs nothing --
    # every index still appears exactly once per epoch and only the ORDER changes --
    # whereas not stratifying when it is needed leaves batches with no gradient for a
    # class. Measure the real incidence with
    #   python scripts/validate_rewards.py --sft-stats --task object_only
    logger.info("Building rare mask for stratified sampling...")
    rare_mask = build_rare_mask_for_task(train_raw_oversampled, args.task)
    if rare_mask is None:
        logger.info(
            f"Task '{args.task}' has no rare-class axis - using plain per-epoch shuffle."
        )
    else:
        n_rare = sum(rare_mask)
        logger.info(
            f"Rare mask for '{args.task}': {n_rare}/{len(rare_mask)} rows "
            f"({n_rare / max(len(rare_mask), 1):.1%}) marked rare."
        )

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

    logger.info(
        f"SFT input: {len(train_ds)} train / {len(val_ds)} val samples "
        f"(task={args.task}, subdir={sft_subdir or 'default (base.yaml processed_subdir)'})"
    )
    logger.info(f"Starting SFT for tier: {args.tier}, variant: {args.variant}...")
    checkpoint_dir = run_sft_unified(
        tier=args.tier,
        variant=args.variant,
        train_dataset=list(train_ds),
        val_dataset=list(val_ds),
        rare_mask=rare_mask,
        train_resolutions=train_resolutions,
        # REQUIRED. Without this the trainer falls back to
        # load_training_config("sft") — configs/sft.yaml alone — which silently
        # discarded (a) the tier learning-rate clamps computed above and (b) any
        # SFT hyperparameter a task YAML tries to override. Every SFT run before
        # this fix trained at 1.0e-4 regardless of tier, while the log line above
        # claimed otherwise.
        sft_cfg=sft_cfg,
        resume=not args.no_resume,
        task=args.task,
    )

    logger.info(f"SFT run complete. Best/final checkpoint at {checkpoint_dir}")


if __name__ == "__main__":
    main()