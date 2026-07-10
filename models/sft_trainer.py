"""
Supervised fine-tuning using Unsloth's FastVisionModel + TRL's SFTTrainer.

run_sft_multitask() is the Project 1 path: ONE model per size variant
(small/medium/large), trained on the COMBINED data of all three tasks.

run_sft() (single-task) is kept for potential future use — NOT part of the
current Project 1 pipeline.
"""
from typing import Any, Dict

from core.config import load_config, load_base_config, load_training_config, load_task_config
from core.io import ensure_dir
from core.logging import get_logger
from core.wandb_utils import init_run, finish_run
from models.registry import (
    get_model_entry, register_finetuned_variant, checkpoint_path, checkpoint_path_multitask,
)
from data.loader import load_construction_dataset
from data.preprocessor import build_sft_dataset, build_multitask_sft_dataset
from data.dataset_cache import save_sft_samples_jsonl, load_sft_samples_jsonl

logger = get_logger(__name__)


def _make_sft_config(sft_cfg: Dict[str, Any], output_dir: str):
    from trl import SFTConfig
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["lr"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        weight_decay=sft_cfg["weight_decay"],
        logging_steps=sft_cfg["logging_steps"],
        save_steps=sft_cfg["save_steps"],
        max_seq_length=sft_cfg["max_seq_length"],
        report_to=["wandb"],
        bf16=True,
    )


def run_sft_multitask(
    model_id: str,
    task_cfgs: Dict[str, Dict[str, Any]],
    variant_name: str = "sft_v1",
    use_cache: bool = True,
    cache_filename: str = "multitask_train.jsonl",
) -> str:
    base_cfg = load_base_config()
    sft_cfg = load_training_config("sft")
    entry = get_model_entry(model_id)
    hf_path = entry["hf_path"]

    from unsloth import FastVisionModel

    logger.info(f"Loading base model for multi-task SFT: {hf_path}")
    model, tokenizer = FastVisionModel.from_pretrained(hf_path, load_in_4bit=sft_cfg["load_in_4bit"])
    model = FastVisionModel.get_peft_model(
        model, r=sft_cfg["lora"]["r"], lora_alpha=sft_cfg["lora"]["alpha"],
        lora_dropout=sft_cfg["lora"]["dropout"], target_modules=sft_cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    # Use the SAME cached dataset across all size variants — required for a
    # fair small vs medium vs large comparison (identical training data/order).
    if use_cache:
        try:
            sft_samples = load_sft_samples_jsonl(cache_filename)
        except FileNotFoundError:
            logger.info("No cached multi-task dataset found — building it now (first run only).")
            raw_dataset = load_construction_dataset()
            sft_samples = build_multitask_sft_dataset(raw_dataset["train"], task_cfgs, seed=base_cfg["seed"])
            save_sft_samples_jsonl(sft_samples, cache_filename)
    else:
        raw_dataset = load_construction_dataset()
        sft_samples = build_multitask_sft_dataset(raw_dataset["train"], task_cfgs, seed=base_cfg["seed"])
        save_sft_samples_jsonl(sft_samples, cache_filename)

    train_data = [
        {"messages": [m.dict() for m in s.messages], "image_id": s.image_id, "task": s.task}
        for s in sft_samples
    ]

    size_name = entry.get("size") or model_id.split("-")[0]
    output_dir = checkpoint_path_multitask(size_name, variant_name)
    ensure_dir(output_dir)
    run = init_run(study_name="sft-multitask", run_name=f"{size_name}-{variant_name}", config=sft_cfg)

    from trl import SFTTrainer
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_data,
                          args=_make_sft_config(sft_cfg, output_dir))

    logger.info(f"Starting multi-task SFT: model_id={model_id}, variant={variant_name}, "
                f"n_samples={len(train_data)}")
    trainer.train()
    trainer.save_model(output_dir)

    new_model_id = f"{size_name}-{variant_name}"
    register_finetuned_variant(model_id, new_model_id, output_dir)
    finish_run(run)
    return output_dir


# --- Single-task SFT (kept for future use; NOT used in current Project 1 flow) ---

def run_sft(task: str, model_id: str, variant_name: str = "sft_v1") -> str:
    cfg = load_config(task=task, training_kind="sft")
    task_cfg = load_task_config(task)
    entry = get_model_entry(model_id)
    hf_path = entry["hf_path"]

    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(hf_path, load_in_4bit=cfg["load_in_4bit"])
    model = FastVisionModel.get_peft_model(
        model, r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    raw_dataset = load_construction_dataset()
    sft_samples = build_sft_dataset(raw_dataset["train"], task, task_cfg)
    train_data = [{"messages": [m.dict() for m in s.messages], "image_id": s.image_id} for s in sft_samples]

    output_dir = checkpoint_path(task, variant_name)
    ensure_dir(output_dir)
    run = init_run(study_name=f"sft-{task}", run_name=variant_name, config=cfg)

    from trl import SFTTrainer
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_data,
                          args=_make_sft_config(cfg, output_dir))
    trainer.train()
    trainer.save_model(output_dir)

    new_model_id = f"{task}-{variant_name}"
    register_finetuned_variant(model_id, new_model_id, output_dir)
    finish_run(run)
    return output_dir