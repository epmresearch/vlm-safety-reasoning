"""
Supervised fine-tuning using Unsloth's FastVisionModel + TRL's SFTTrainer.

Implements the unified single-prompt approach: ONE model trained on ALL tasks
(caption + object detection + safety violations) in a single JSON output.

Key features:
  - UnslothVisionDataCollator with train_on_responses_only
  - SFTConfig with vision-specific parameters
  - Checkpoint/resume support via training_state.json
  - W&B integration with run resumption
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import load_base_config, load_training_config
from core.constants import MODEL_TIERS, DEFAULT_MODEL_TIER
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from models.model_loader import load_model_for_training, get_model_info, get_batch_config

logger = get_logger(__name__)


def _get_checkpoint_dir(model_short_name: str, variant: str) -> Path:
    """Returns the checkpoint directory path on Google Drive."""
    return get_drive_path("checkpoints", model_short_name, variant)


def _load_training_state(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Loads training_state.json if it exists."""
    state_path = checkpoint_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return None


def _save_training_state(checkpoint_dir: Path, state: Dict[str, Any]) -> None:
    """Saves training_state.json."""
    state_path = checkpoint_dir / "training_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info(f"Saved training state to {state_path}")


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """Finds the latest checkpoint-N directory."""
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def run_sft_unified(
    tier: str = DEFAULT_MODEL_TIER,
    variant: str = "unified-sft-v1",
    train_dataset: Optional[List[Dict[str, Any]]] = None,
    val_dataset: Optional[List[Dict[str, Any]]] = None,
    sft_cfg: Optional[Dict[str, Any]] = None,
    resume: bool = True,
    run_name: Optional[str] = None,
) -> str:
    """Runs unified SFT training with full checkpointing and W&B integration.

    Args:
        tier: Model tier ("2b", "4b", "8b").
        variant: Checkpoint variant name (e.g., "unified-sft-v1").
        train_dataset: List of Unsloth conversation dicts for training.
        val_dataset: List of Unsloth conversation dicts for validation.
        sft_cfg: SFT config dict. If None, loads from configs/sft.yaml.
        resume: Whether to resume from latest checkpoint.
        run_name: W&B run name. If None, auto-generated.

    Returns:
        Path to the final checkpoint directory.
    """
    # Load configs
    if sft_cfg is None:
        sft_cfg = load_training_config("sft")
    base_cfg = load_base_config()
    model_info = get_model_info(tier)
    batch_cfg = get_batch_config(tier)

    # Checkpoint directory
    checkpoint_dir = _get_checkpoint_dir(model_info["short_name"], variant)
    ensure_dir(checkpoint_dir)

    # Check for resume
    resume_checkpoint = None
    wandb_run_id = None
    if resume:
        state = _load_training_state(checkpoint_dir)
        if state:
            resume_checkpoint = _find_latest_checkpoint(checkpoint_dir)
            wandb_run_id = state.get("wandb_run_id")
            logger.info(
                f"Resuming from checkpoint: {resume_checkpoint}, "
                f"W&B run: {wandb_run_id}"
            )

    # Load model
    model, tokenizer, model_info = load_model_for_training(
        model_name=model_info["hf_path"],
        tier=tier,
        sft_cfg=sft_cfg
    )

    # Build SFTConfig
    from trl import SFTTrainer, SFTConfig

    if run_name is None:
        run_name = f"{model_info['short_name']}-{variant}"

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=batch_cfg["gradient_accumulation_steps"],
        num_train_epochs=sft_cfg.get("num_train_epochs", 3),
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_steps=sft_cfg.get("save_steps", 200),
        eval_steps=sft_cfg.get("eval_steps", 200),
        eval_strategy=sft_cfg.get("eval_strategy", "steps"),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        load_best_model_at_end=sft_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=sft_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=sft_cfg.get("greater_is_better", False),
        bf16=sft_cfg.get("bf16", True),
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
        seed=base_cfg.get("seed", 42),
        report_to=["wandb"],
        run_name=run_name,
        # CRITICAL for vision training:
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Build data collator with train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator

    data_collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Initialize W&B
    import wandb
    wandb_kwargs = {
        "project": base_cfg["wandb_project"],
        "entity": base_cfg.get("wandb_entity"),
        "name": run_name,
        "config": {**sft_cfg, **batch_cfg, "model": model_info["hf_path"]},
    }
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    run = wandb.init(**wandb_kwargs)

    # Build trainer
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "args": training_args,
    }
    if val_dataset:
        trainer_kwargs["eval_dataset"] = val_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    logger.info(
        f"Starting SFT: model={model_info['hf_path']}, variant={variant}, "
        f"n_train={len(train_dataset) if train_dataset else 0}, "
        f"n_val={len(val_dataset) if val_dataset else 0}, "
        f"effective_batch={batch_cfg['per_device_train_batch_size'] * batch_cfg['gradient_accumulation_steps']}"
    )

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Save final adapter
    final_dir = checkpoint_dir / "final"
    ensure_dir(final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved final adapter to {final_dir}")

    # Save training state
    from datetime import datetime, timezone
    _save_training_state(checkpoint_dir, {
        "model_name": model_info["hf_path"],
        "tier": tier,
        "variant": variant,
        "completed_epochs": sft_cfg.get("num_train_epochs", 3),
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "wandb_run_id": run.id if run else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    # Finish W&B
    wandb.finish()

    return str(final_dir)