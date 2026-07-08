"""
Supervised fine-tuning using Unsloth's FastVisionModel + TRL's SFTTrainer.
Saves adapter to Drive and pushes to the HF Hub.
"""
import os
from typing import Any, Dict, List

from core.config import load_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from models.registry import get_model_entry, register_finetuned_variant, checkpoint_path
from data.loader import load_construction_dataset
from data.preprocessor import build_sft_dataset
from core.config import load_task_config

logger = get_logger(__name__)


def run_sft(task: str, model_id: str, variant_name: str = "sft_v1") -> str:
    """
    Returns the local path to the saved adapter.
    """
    cfg = load_config(task=task, training_kind="sft")
    task_cfg = load_task_config(task)
    entry = get_model_entry(model_id)
    hf_path = entry["hf_path"]

    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig

    logger.info(f"Loading base model for SFT: {hf_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        hf_path,
        load_in_4bit=cfg["load_in_4bit"],
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    logger.info("Loading and preprocessing dataset for SFT...")
    raw_dataset = load_construction_dataset()
    sft_samples = build_sft_dataset(raw_dataset["train"], task, task_cfg)
    train_data = [
        {"messages": [m.dict() for m in s.messages], "image_id": s.image_id}
        for s in sft_samples
    ]

    output_dir = checkpoint_path(task, variant_name)
    ensure_dir(output_dir)

    run = init_run(study_name=f"sft-{task}", run_name=variant_name, config=cfg)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        max_seq_length=cfg["max_seq_length"],
        report_to=["wandb"],
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        args=sft_config,
    )

    logger.info(f"Starting SFT training: task={task}, model_id={model_id}, variant={variant_name}")
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
    parser.add_argument("--variant_name", default="sft_v1")
    args = parser.parse_args()
    run_sft(args.task, args.model_id, args.variant_name)