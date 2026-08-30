"""
GRPO / GSPO-style reinforcement learning on top of an SFT checkpoint,
using TRL's GRPOTrainer with the 6-component reward system from rewards/.

The model produces unified JSON output (caption + detected_objects +
safety_violations) and the composite reward scores it across six axes:
    1. Format validity  (schema compliance)
    2. Caption quality   (semantic + lexical + length calibration)
    3. Object grounding  (mask-union IoU with TN fix)
    4. Violation ID       (F-beta, β=2, recall-weighted)
    5. Violation grounding (TP-conditioned mask-union IoU)
    6. Reasoning quality   (TP-conditioned semantic + lexical + length)

Supports two modes:
    - TRL native multi-reward:  reward_funcs=[f1,...,f6], reward_weights=[w1,...,w6]
    - Fallback single-function: reward_funcs=[unified_fn] (for older TRL versions)
"""
import os
from typing import Callable, Dict, List, Optional

from core.config import load_config, load_task_config
from core.io import ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from core.callbacks import PersistentCheckpointCallback, GPUMemoryLoggingCallback, ConsoleLogCallback
from models.model_loader import get_model_info, load_model_for_training
from core.io import get_drive_path
from data.loader import load_processed_dataset
from data.preprocessor import to_grpo_prompt, build_grpo_dataset

from rewards.unified_reward import (
    get_reward_funcs_and_weights,
    build_grpo_reward_fn,
    REWARD_COMPONENTS,
)

logger = get_logger(__name__)


def _check_trl_supports_reward_weights() -> bool:
    """Check if installed TRL version supports reward_weights in GRPOConfig."""
    try:
        from trl import GRPOConfig
        import inspect
        sig = inspect.signature(GRPOConfig)
        return "reward_weights" in sig.parameters
    except Exception:
        return False


def run_grpo(
    task: str,
    model_id: str,
    variant_name: str = "grpo_v1",
    max_samples: Optional[int] = None,
    adapter_path: Optional[str] = None,
    base_model_override: Optional[str] = None,
) -> str:
    """Run GRPO training for the unified safety inspection task.

    Args:
        task: Task name (e.g. "unified").
        model_id: Model registry ID to fine-tune.
        variant_name: Name for the output variant.
        max_samples: Optional cap on dataset size (for debugging).
        adapter_path: Optional explicit path to the SFT adapter to load. 
                      Overrides the default in the model registry.
        base_model_override: If set, loads THIS local path as the base model
                             instead of downloading from HuggingFace. Use this
                             with a merged SFT model so that TRL's KL reference
                             correctly points to the SFT policy, not the raw base.

    Returns:
        Path to the saved checkpoint directory.
    """
    cfg = load_config(task=task, training_kind="grpo")
    sft_cfg = load_config(task=task, training_kind="sft")
    task_cfg = load_task_config(task)
    entry = get_model_info(model_id)
    
    # Use merged SFT model path if provided, otherwise use HuggingFace model path
    hf_path = base_model_override if base_model_override else entry["hf_path"]
    if base_model_override:
        logger.info(f"Using MERGED SFT base model from disk: {base_model_override}")
        logger.info("TRL reference model will correctly point to SFT policy (not raw base)")
    
    # Use the explicitly provided adapter_path if available, otherwise fallback to registry
    lora_path = adapter_path if adapter_path is not None else entry.get("lora_path")

    from unsloth import FastVisionModel, PatchFastRL
    PatchFastRL("GRPO", FastVisionModel)
    from trl import GRPOTrainer, GRPOConfig

    class _PerImageGRPOTrainer(GRPOTrainer):
        """Works around a confirmed TRL bug: when _tokenize_prompts batches
        multiple different conversations (each with its own image) into a
        single apply_chat_template(..., tokenize=True) call, only the FIRST
        image in the batch gets expanded into real vision tokens — every
        other prompt collapses to a single unexpanded <|image_pad|>
        placeholder (confirmed directly: a 2-image batch test returned
        [795, 239] tokens instead of two properly-expanded lengths, and the
        real training run's logged prompts showed exactly one image_pad
        token for every sample). This override tokenizes each conversation
        one at a time instead, then merges the per-prompt multimodal fields
        exactly the way TRL's own per-environment branch already does
        (see the environment_factories branch in vanilla _tokenize_prompts).
        """

        def _tokenize_prompts(self, prompts: list):
            import torch as _torch
            from trl.data_utils import prepare_multimodal_messages

            # This pipeline's prompts are always conversational and always
            # contain an image, so no need to branch on self._is_vlm (which
            # Unsloth's rewritten trainer doesn't reliably expose here).
            prompts = [prepare_multimodal_messages(prompt) for prompt in prompts]

            images = []
            has_images = False
            for prompt in prompts:
                prompt_images = []
                for message in prompt:
                    if isinstance(message["content"], list):
                        for part in message["content"]:
                            if part["type"] == "image":
                                prompt_images.append(part["image"])
                                has_images = True
                images.append(prompt_images if prompt_images else None)
            images = images if has_images else None

            prompt_ids = []
            multimodal_fields = {}
            for prompt in prompts:
                tokenized = self.processing_class.apply_chat_template(
                    conversation=[prompt],
                    tools=self.tools or None,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
                prompt_ids.append(tokenized["input_ids"][0])
                for k, v in tokenized.items():
                    if k not in ("input_ids", "attention_mask"):
                        multimodal_fields.setdefault(k, []).append(v)

            multimodal_fields = {
                k: _torch.cat(v) if isinstance(v[0], _torch.Tensor) else [row for prompt_v in v for row in prompt_v]
                for k, v in multimodal_fields.items()
            }
            return prompt_ids, images, multimodal_fields

    logger.info(f"Loading model for GRPO: base={hf_path}, adapter={lora_path}")
    
    # Override SFT context window with explicitly defined GRPO context window
    if "max_seq_length" in cfg:
        sft_cfg["max_seq_length"] = cfg["max_seq_length"]
    
    # Override gradient checkpointing so we can disable Unsloth offloading in GRPO
    if "use_gradient_checkpointing" in cfg:
        sft_cfg["use_gradient_checkpointing"] = cfg["use_gradient_checkpointing"]
        
    if "load_in_4bit" in cfg:
        sft_cfg["load_in_4bit"] = cfg["load_in_4bit"]

    model, tokenizer, _ = load_model_for_training(
        model_name=hf_path,
        tier=model_id,
        sft_cfg=sft_cfg,
        adapter_path=lora_path,
        # hf_path may be a local merged checkpoint (base_model_override); Unsloth's VLM
        # processor auto-detection degrades to text-only for a local qwen3_vl path, so
        # always load the tokenizer/processor from the original HF repo instead.
        tokenizer_name=entry["hf_path"],
    )

    # Left padding is required for GRPO batched generation rollouts
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Force stochastic sampling at the model level (TRL ignores do_sample kwarg)
    # -----------------------------------------------------------------------
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        max_new_tokens=task_cfg.get("max_completion_length", cfg.get("max_completion_length", 1024)),
        pad_token_id=tokenizer.pad_token_id,
    )
    logger.info(f"Forced generation_config on model: {model.generation_config}")

    # -----------------------------------------------------------------------
    # Build reward configuration
    # -----------------------------------------------------------------------
    use_native_weights = _check_trl_supports_reward_weights()
    logger.info(f"TRL native reward_weights support: {use_native_weights}")

    from rewards.unified_reward import get_reward_funcs_for_task
    _funcs, _weights = get_reward_funcs_for_task(task)
    # Log the component registry
    for func, weight in zip(_funcs, _weights):
        logger.info(f"  Reward component: {func.__name__} (weight={weight:.2f})")

    # -----------------------------------------------------------------------
    # Load the GRPO training pool: prefer the pre-built, balanced pool
    # (data/build_grpo_pool.py) if it's been built; otherwise fall back to
    # the raw train split + oversampling so this still works standalone
    # while the pool feature is still being finished/tested separately.
    # -----------------------------------------------------------------------
    try:
        from data.loader import load_grpo_pool
        train_split = load_grpo_pool()
        logger.info(f"GRPO pool loaded: {len(train_split)} samples (already balanced)")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"GRPO pool not available ({e}); falling back to raw train split + oversampling")
        from data.loader import load_processed_dataset
        from data.oversampling import build_oversampled_indices
        raw_dataset = load_processed_dataset()
        train_split = raw_dataset["train"]
        oversampled_indices, oversample_manifest = build_oversampled_indices(
            train_split,
            rule24_multiplier=cfg.get("oversample_rule24_multiplier", 1),
            rule3_multiplier=cfg.get("oversample_rule3_multiplier", 1),
        )
        train_split = train_split.select(oversampled_indices)
        logger.info(f"Oversample manifest: {oversample_manifest}")
        logger.info(
            f"Oversampled training set: {len(raw_dataset['train'])} → "
            f"{len(train_split)} samples"
        )

    from data.preprocessor import build_grpo_dataset_for_task
    train_data = build_grpo_dataset_for_task(train_split, task=task, max_samples=max_samples)
    logger.info(f"GRPO prompt dataset built: {len(train_data)} samples (Python list)")

    if not train_data:
        raise ValueError(
            "GRPO training dataset is empty. Cannot train on zero samples."
        )

    # -----------------------------------------------------------------------
    # Output directory & checkpoint recovery
    # -----------------------------------------------------------------------
    short_name = entry.get("short_name", f"qwen3vl-{model_id}")
    output_dir = str(get_drive_path("checkpoints", short_name, variant_name))
    ensure_dir(output_dir)

    # Save a run manifest with all configs
    import json
    manifest = {
        "grpo_cfg": cfg,
        "sft_cfg": sft_cfg,
        "task_cfg": task_cfg,
        "model_id": model_id,
        "variant": variant_name
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    from transformers.trainer_utils import get_last_checkpoint

    resume_from_checkpoint = False
    if os.path.exists(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            resume_from_checkpoint = True
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # -----------------------------------------------------------------------
    # WandB tracking
    # -----------------------------------------------------------------------
    run = init_run(study_name=f"grpo-{task}", run_name=variant_name, config=cfg)

    # -----------------------------------------------------------------------
    # GRPOConfig + Trainer setup
    # -----------------------------------------------------------------------
    grpo_config_kwargs = dict(
        output_dir=output_dir,
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg.get("max_prompt_length", 2048),
        max_completion_length=task_cfg.get("max_completion_length", cfg.get("max_completion_length", 1024)),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit", 3),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        beta=cfg["beta"],
        temperature=0.9,
        top_p=0.95,
        generation_kwargs={"do_sample": True},
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "adamw_8bit"),
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=4,
    )

    if use_native_weights:
        # ------------------------------------------------------------------
        # Mode 1: TRL native multi-reward (preferred — per-component logging)
        # ------------------------------------------------------------------
        from rewards.unified_reward import get_reward_funcs_for_task
        reward_funcs, reward_weights = get_reward_funcs_for_task(task)
        grpo_config_kwargs["reward_weights"] = reward_weights

        grpo_config = GRPOConfig(**grpo_config_kwargs)

        trainer = _PerImageGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_data,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
            callbacks=[
                PersistentCheckpointCallback(persistent_freq=100),
                GPUMemoryLoggingCallback(every_n_steps=10),
                ConsoleLogCallback()
            ],
        )
        logger.info(
            f"Using TRL native multi-reward mode: "
            f"{len(reward_funcs)} functions, weights={reward_weights}"
        )
    else:
        # ------------------------------------------------------------------
        # Mode 2: Fallback single composite reward (older TRL versions)
        # ------------------------------------------------------------------
        reward_fn = build_grpo_reward_fn(task=task)

        grpo_config = GRPOConfig(**grpo_config_kwargs)

        trainer = _PerImageGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_data,
            reward_funcs=[reward_fn],
            processing_class=tokenizer,
            callbacks=[
                PersistentCheckpointCallback(persistent_freq=100),
                GPUMemoryLoggingCallback(every_n_steps=10),
                ConsoleLogCallback()
            ],
        )
        logger.info(
            "Using single composite reward mode (TRL reward_weights not available)"
        )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info(
        f"Starting GRPO training: task={task}, model_id={model_id}, "
        f"variant={variant_name}, dataset_size={len(train_data)}, "
        f"num_generations={cfg['num_generations']}, beta={cfg['beta']}"
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        final_dir = os.path.join(output_dir, "final")
        ensure_dir(final_dir)
        logger.info(f"Saving final adapter to {final_dir}")
        trainer.save_model(final_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        finish_run(run)

    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--variant_name", default="grpo_v1")
    parser.add_argument("--adapter_path", default=None,
                        help="Optional explicit path to SFT adapter")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size for debugging")
    args = parser.parse_args()
    run_grpo(args.task, args.model_id, args.variant_name, args.max_samples, adapter_path=args.adapter_path)