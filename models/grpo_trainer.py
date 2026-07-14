"""
GRPO / GSPO-style reinforcement learning on top of an SFT checkpoint,
using TRL's GRPOTrainer with unified reward functions from rewards/.

The model produces unified JSON output (caption + detected_objects +
safety_violations) and the composite reward scores it across all axes.
"""
from typing import Callable, Dict, List

from core.config import load_config, load_task_config
from core.io import ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from models.registry import get_model_entry, register_finetuned_variant, checkpoint_path
from data.loader import load_construction_dataset
from data.preprocessor import to_grpo_prompt

from rewards.unified_reward import compute_reward, DEFAULT_WEIGHTS

logger = get_logger(__name__)


def _build_grpo_reward_fn(
    weights: Dict[str, float],
) -> Callable[[List[str], List[Dict]], List[float]]:
    """Build a single reward function for GRPOTrainer.

    TRL's GRPOTrainer expects reward functions with signature:
        (completions: List[str], **kwargs) -> List[float]

    where kwargs includes ground_truth (list of dicts aligned with
    completions). We wrap our unified compute_reward to match.

    Args:
        weights: Per-component reward weights.

    Returns:
        A callable matching the GRPOTrainer reward function interface.
    """

    def unified_reward_fn(
        completions: List[str],
        ground_truth: List[Dict] = None,
        **kwargs,
    ) -> List[float]:
        if ground_truth is None:
            return [0.0] * len(completions)

        scores = []
        for completion, gt in zip(completions, ground_truth):
            score = compute_reward(completion, gt, weights=weights)
            scores.append(score)
        return scores

    unified_reward_fn.__name__ = "reward_unified"
    return unified_reward_fn


def run_grpo(task: str, model_id: str, variant_name: str = "grpo_v1") -> str:
    """Run GRPO training for the unified safety inspection task.

    Args:
        task: Task name (e.g. "full_unified").
        model_id: Model registry ID to fine-tune.
        variant_name: Name for the output variant.

    Returns:
        Path to the saved checkpoint directory.
    """
    cfg = load_config(task=task, training_kind="grpo")
    task_cfg = load_task_config(task)
    entry = get_model_entry(model_id)
    hf_path = entry["hf_path"]
    lora_path = entry.get("lora_path")  # typically the SFT adapter

    from unsloth import FastVisionModel
    from trl import GRPOTrainer, GRPOConfig

    logger.info(f"Loading model for GRPO: base={hf_path}, adapter={lora_path}")
    model, tokenizer = FastVisionModel.from_pretrained(hf_path, load_in_4bit=True)
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    # Build reward weights from task config, falling back to defaults
    config_weights = task_cfg.get("reward_weights", {})
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(config_weights)

    logger.info(f"Reward weights: {weights}")

    logger.info("Building GRPO prompt dataset...")
    raw_dataset = load_construction_dataset()
    grpo_prompts = [to_grpo_prompt(raw, task, task_cfg) for raw in raw_dataset["train"]]
    train_data = [
        {
            "prompt": [m.dict() for m in p.prompt_messages],
            "image_id": p.image_id,
            "ground_truth": p.ground_truth,
        }
        for p in grpo_prompts
    ]

    reward_fn = _build_grpo_reward_fn(weights)

    output_dir = checkpoint_path(task, variant_name)
    ensure_dir(output_dir)

    run = init_run(study_name=f"grpo-{task}", run_name=variant_name, config=cfg)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        beta=cfg["beta"],
        report_to=["wandb"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_data,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
    )

    logger.info(
        f"Starting GRPO training: task={task}, model_id={model_id}, "
        f"variant={variant_name}"
    )
    trainer.train()

    logger.info(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)

    new_model_id = f"{task}-{variant_name}"
    register_finetuned_variant(model_id, new_model_id, output_dir)

    finish_run(run)
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--variant_name", default="grpo_v1")
    args = parser.parse_args()
    run_grpo(args.task, args.model_id, args.variant_name)