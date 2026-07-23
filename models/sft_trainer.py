"""
Supervised fine-tuning using Unsloth's FastVisionModel + TRL's SFTTrainer.

Unified single-prompt approach: ONE model trained on ALL tasks (caption +
object detection + safety violations) via a single JSON output.

Robustness features:
  - Stratified rare-class batch sampling (data/samplers.py) — default now
    that pixel capping handles OOM safety instead of resolution bucketing.
  - image_min_pixels/max_pixels capping to neutralize 4K/14MP outliers
  - wandb run id persisted at train START (not end) -> resume actually resumes
  - SaveBestModelCallback / ManifestUpdateCallback / GPUMemoryLoggingCallback
  - EarlyStoppingCallback (optional)
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import load_base_config, load_training_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from core.callbacks import SaveBestModelCallback, ManifestUpdateCallback, GPUMemoryLoggingCallback, _ensure_preprocessor_config
from models.model_loader import load_model_for_training, get_model_info, get_batch_config, log_gpu_memory

logger = get_logger(__name__)


def _get_checkpoint_dir(model_short_name: str, variant: str) -> Path:
    return get_drive_path("checkpoints", model_short_name, variant)


def _load_training_state(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = checkpoint_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return None


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def _build_stratified_trainer_class(sampler_batch_size: int, rare_mask: List[bool], cfg: Dict[str, Any]):
    """Returns an SFTTrainer subclass whose train dataloader uses StratifiedRareClassSampler."""
    from trl import SFTTrainer
    from torch.utils.data import DataLoader
    from data.samplers import StratifiedRareClassSampler

    class StratifiedVisionSFTTrainer(SFTTrainer):
        def get_train_dataloader(self) -> DataLoader:
            sampler = StratifiedRareClassSampler(
                rare_mask=rare_mask,
                shuffle=True,
                seed=cfg.get("bucket_shuffle_seed", 42),
            )
            sampler.set_epoch(int(self.state.epoch or 0))
            dataloader_params = {
                "batch_size": self.args.train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "sampler": sampler,
                "drop_last": self.args.dataloader_drop_last,
            }
            return DataLoader(self.train_dataset, **dataloader_params)

    return StratifiedVisionSFTTrainer


def _build_bucketed_trainer_class(sampler_batch_size: int, resolutions: List[float], cfg: Dict[str, Any]):
    """Legacy: SFTTrainer subclass using ResolutionBucketSampler. Kept for
    backward compatibility / ablation use only — not the default anymore."""
    from trl import SFTTrainer
    from torch.utils.data import DataLoader
    from data.samplers import ResolutionBucketSampler

    class BucketedVisionSFTTrainer(SFTTrainer):
        def get_train_dataloader(self) -> DataLoader:
            sampler = ResolutionBucketSampler(
                resolutions=resolutions,
                batch_size=sampler_batch_size,
                shuffle=True,
                seed=cfg.get("bucket_shuffle_seed", 42),
            )
            sampler.set_epoch(int(self.state.epoch or 0))
            dataloader_params = {
                "batch_size": self.args.train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "sampler": sampler,
                "drop_last": self.args.dataloader_drop_last,
            }
            return DataLoader(self.train_dataset, **dataloader_params)

    return BucketedVisionSFTTrainer


def run_sft_unified(
    tier: Optional[str] = None,
    variant: str = "unified-sft-v1",
    train_dataset: Optional[List[Dict[str, Any]]] = None,
    val_dataset: Optional[List[Dict[str, Any]]] = None,
    rare_mask: Optional[List[bool]] = None,
    train_resolutions: Optional[List[float]] = None,
    sft_cfg: Optional[Dict[str, Any]] = None,
    resume: bool = True,
    run_name: Optional[str] = None,
    start_adapter_path: Optional[str] = None,   # <-- NEW
) -> str:
    """Runs unified SFT training with full checkpointing, resume, and best-model saving.

    Args:
        tier: Model tier ("2b", "4b", "8b").
        variant: Checkpoint variant name — use a NEW variant per ablation.
        train_dataset / val_dataset: Unsloth conversation dicts.
        rare_mask: Boolean list, same order/length as train_dataset, True for
            rows containing a Rule 2/3/4 violation. Required if
            `use_stratified_rare_sampling` is enabled (the new default).
        train_resolutions: Per-sample pixel counts. Only used if
            `use_resolution_bucketing` is explicitly enabled (legacy path).
        sft_cfg: SFT config dict. If None, loads configs/sft.yaml.
        resume: Whether to resume from the latest checkpoint + wandb run.
        run_name: W&B run name. Defaults to "{short_name}-{variant}".

    Returns:
        Path to the final ("best") adapter directory.
    """
    if sft_cfg is None:
        sft_cfg = load_training_config("sft")
    base_cfg = load_base_config()
    model_info = get_model_info(tier)
    batch_cfg = get_batch_config(tier)

    checkpoint_dir = _get_checkpoint_dir(model_info["short_name"], variant)
    ensure_dir(checkpoint_dir)
    best_dir = checkpoint_dir / sft_cfg.get("best_model_subdir", "best")

    # --- Resume detection ---
    resume_checkpoint = None
    wandb_run_id = None
    prior_state = _load_training_state(checkpoint_dir) if resume else None
    if prior_state:
        if prior_state.get("hf_path") not in (None, model_info["hf_path"]):
            raise ValueError(
                f"Checkpoint dir {checkpoint_dir} was trained with model "
                f"{prior_state.get('hf_path')}, not {model_info['hf_path']}. "
                f"Use a different --variant name for this config."
            )
        resume_checkpoint = _find_latest_checkpoint(checkpoint_dir)
        wandb_run_id = prior_state.get("wandb_run_id")
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}, W&B run: {wandb_run_id}")

    # --- Load model ---
    model, tokenizer, model_info = load_model_for_training(
        model_name=model_info["hf_path"], tier=tier, sft_cfg=sft_cfg, adapter_path=start_adapter_path,
    )
    log_gpu_memory("after model load")

    # --- Build SFTConfig ---
    from trl import SFTConfig

    if run_name is None:
        run_name = f"{model_info['short_name']}-{variant}"

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=batch_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=batch_cfg["gradient_accumulation_steps"],
        num_train_epochs=sft_cfg.get("num_train_epochs", 3),
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        optim=sft_cfg.get("optim", "adamw_8bit"),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_steps=sft_cfg.get("save_steps", 100),
        eval_steps=sft_cfg.get("eval_steps", 100),
        eval_accumulation_steps=batch_cfg.get("eval_accumulation_steps"),
        eval_strategy=sft_cfg.get("eval_strategy", "steps"),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        load_best_model_at_end=sft_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=sft_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=sft_cfg.get("greater_is_better", False),
        bf16=sft_cfg.get("bf16", True),
        seed=base_cfg.get("seed", 42),
        report_to=["wandb"],
        run_name=run_name,
        auto_find_batch_size=sft_cfg.get("auto_find_batch_size", False),
        dataloader_drop_last=sft_cfg.get("dataloader_drop_last", False),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # --- Data collator ---
    from unsloth.trainer import UnslothVisionDataCollator
    data_collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # --- W&B: init BEFORE training, persist run_id IMMEDIATELY ---
    import wandb
    from core.wandb_utils import init_run
    run = init_run(
        study_name="sft",
        run_name=run_name,
        config={**sft_cfg, **batch_cfg, "model": model_info["hf_path"], "variant": variant},
        run_id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

    static_manifest_fields = {
        "hf_path": model_info["hf_path"],
        "tier": tier,
        "variant": variant,
        "wandb_run_id": run.id if run else None,
    }
    ensure_dir(checkpoint_dir)
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump({**static_manifest_fields, "status": "starting"}, f, indent=2)
        
    # Dump the full merged configuration for local reproducibility
    from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT
    from core.config import load_task_config
    try:
        task_cfg = load_task_config("unified")
    except FileNotFoundError:
        task_cfg = {}
        
    # Capture Git metadata
    import subprocess
    git_commit = "unknown"
    git_is_dirty = False
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        git_is_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT).decode("utf-8").strip())
    except Exception:
        pass
        
    full_config = {
        "base_cfg": base_cfg,
        "sft_cfg": sft_cfg,
        "batch_cfg": batch_cfg,
        "task_cfg": task_cfg,
        "model_info": model_info,
        "tier": tier,
        "variant": variant,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
        "prompts": {
            "SYSTEM_PROMPT": SYSTEM_PROMPT,
            "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT
        }
    }
    with open(checkpoint_dir / "run_config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    # --- Callbacks ---
    callbacks = [
        SaveBestModelCallback(
            best_dir=str(best_dir),
            metric_name=sft_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=sft_cfg.get("greater_is_better", False),
            improvement_threshold=sft_cfg.get("best_model_threshold", 0.0),
            base_model_name=model_info["hf_path"],
        ),
        ManifestUpdateCallback(checkpoint_dir=str(checkpoint_dir), static_fields=static_manifest_fields),
        GPUMemoryLoggingCallback(every_n_steps=sft_cfg.get("log_gpu_memory_every_n_steps", 20)),
    ]
    patience = sft_cfg.get("early_stopping_patience")
    if patience:
        from transformers import EarlyStoppingCallback
        threshold = sft_cfg.get("early_stopping_threshold", 0.0)
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold
        ))

    # --- Trainer selection ---
    use_stratified = sft_cfg.get("use_stratified_rare_sampling", True) and rare_mask is not None
    use_bucketing = sft_cfg.get("use_resolution_bucketing", False) and train_resolutions is not None

    if use_stratified and len(rare_mask) != len(train_dataset):
        logger.warning(
            f"rare_mask length ({len(rare_mask)}) != train_dataset length "
            f"({len(train_dataset)}). Disabling stratified sampling for safety."
        )
        use_stratified = False

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "args": training_args,
        "callbacks": callbacks,
        "max_seq_length": sft_cfg.get("max_seq_length", 2048),
    }
    if val_dataset:
        trainer_kwargs["eval_dataset"] = val_dataset

    if use_stratified:
        logger.info("Stratified rare-class sampling ENABLED for train dataloader.")
        TrainerClass = _build_stratified_trainer_class(
            sampler_batch_size=batch_cfg["per_device_train_batch_size"],
            rare_mask=rare_mask,
            cfg=sft_cfg,
        )
    elif use_bucketing:
        logger.info("Resolution bucketing ENABLED (legacy path) for train dataloader.")
        TrainerClass = _build_bucketed_trainer_class(
            sampler_batch_size=batch_cfg["per_device_train_batch_size"],
            resolutions=train_resolutions,
            cfg=sft_cfg,
        )
    else:
        logger.info("No custom sampler (falling back to default shuffling).")
        from trl import SFTTrainer as TrainerClass

    trainer = TrainerClass(**trainer_kwargs)

    logger.info(
        f"Starting SFT: model={model_info['hf_path']}, variant={variant}, "
        f"n_train={len(train_dataset) if train_dataset else 0}, "
        f"n_val={len(val_dataset) if val_dataset else 0}, "
        f"effective_batch={batch_cfg['per_device_train_batch_size'] * batch_cfg['gradient_accumulation_steps']}"
    )
    log_gpu_memory("before trainer.train()")

    try:
        if resume_checkpoint:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()
        final_status = "completed"
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state before exiting.")
        final_status = "interrupted"
    except RuntimeError as e:
        logger.error(f"Training crashed (likely OOM or CUDA error): {e}")
        final_status = f"crashed: {e}"
        raise
    finally:
        final_dir = checkpoint_dir / "final"
        try:
            ensure_dir(final_dir)
            model.save_pretrained(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))
            _ensure_preprocessor_config(str(final_dir), model_info["hf_path"])
            logger.info(f"Saved final adapter snapshot to {final_dir}")
        except Exception as e:
            logger.warning(f"Could not save final snapshot: {e}")

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump({
                **static_manifest_fields,
                "status": final_status,
                "global_step": trainer.state.global_step,
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
            }, f, indent=2, default=str)

        wandb.finish()

    return str(best_dir) if best_dir.exists() else str(checkpoint_dir / "final")