"""
GRPO / GSPO-style reinforcement learning on top of an SFT checkpoint,
using TRL's GRPOTrainer with reward functions from rewards/.
"""
from typing import Callable, List

from core.config import load_config, load_task_config
from core.io import ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from models.registry import get_model_entry, register_finetuned_variant, checkpoint_path
from data.loader import load_construction_dataset
from data.preprocessor import to_grpo_prompt

from rewards.json_validity import reward_json_validity
from rewards.rule_violation_accuracy import reward_rule_violation_accuracy
from rewards.grounding_iou import reward_grounding_iou
from rewards.caption_quality import reward_caption_quality
from rewards.attribute_accuracy import reward_attribute_accuracy

logger = get_logger(__name__)

REWARD_FN_MAP = {
    "json_validity": reward_json_validity,
    "rule_violation_accuracy": reward_rule_violation_accuracy,
    "grounding_iou": reward_grounding_iou,
    "caption_quality": reward_caption_quality,
    "attribute_accuracy": reward_attribute_accuracy,
}


def build_reward_functions(task_cfg: dict) -> List[Callable]:
    weights = task_cfg["reward_weights"]
    fns = []
    for name, weight in weights.items():
        base_fn = REWARD_FN_MAP[name]

        def weighted_fn(completions, base_fn=base_fn, weight=weight, **kwargs):
            scores = base_fn(completions, **kwargs)
            return [s * weight for s in scores]

        weighted_fn.__name__ = f"reward_{name}"
        fns.append(weighted_fn)
    return fns


def run_grpo(task: str, model_id: str, variant_name: str = "grpo_v1") -> str:
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

    reward_funcs = build_reward_functions(task_cfg)

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
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
    )

    logger.info(f"Starting GRPO training: task={task}, model_id={model_id}, variant={variant_name}")
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