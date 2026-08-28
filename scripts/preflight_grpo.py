import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_config, load_task_config
from models.model_loader import get_model_info, load_model_for_training
from data.loader import load_processed_dataset
from data.preprocessor import build_grpo_dataset_for_task

def run_preflight(task="violations_only", model_id="2b"):
    print(">>> 1. Loading configs...")
    cfg = load_config(task=task, training_kind="grpo")
    sft_cfg = load_config(task=task, training_kind="sft")
    task_cfg = load_task_config(task)
    entry = get_model_info(model_id)
    hf_path = entry["hf_path"]
    
    print(">>> 2. Loading model (this may take a minute)...")
    from unsloth import FastVisionModel, PatchFastRL
    PatchFastRL("GRPO", FastVisionModel)
    
    # We load the model normally just like the trainer does
    model, tokenizer, _ = load_model_for_training(
        model_name=hf_path,
        tier=model_id,
        sft_cfg=sft_cfg,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(">>> 3. Loading a tiny dataset slice...")
    raw_dataset = load_processed_dataset()
    # Just take 4 samples for a quick check
    train_split = raw_dataset["train"].select(range(4))
    train_data = build_grpo_dataset_for_task(train_split, task=task)
    
    print(">>> 4. Initializing GRPOTrainer (No training will happen)...")
    from trl import GRPOTrainer, GRPOConfig
    
    # Mock reward function just to satisfy the trainer
    def mock_reward(prompts, completions, **kwargs):
        return [1.0] * len(prompts)
        
    grpo_config = GRPOConfig(
        output_dir="./tmp_preflight",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none" # Disable wandb for this test
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_data,
        reward_funcs=[mock_reward],
        processing_class=tokenizer,
    )
    
    print(">>> 5. Fetching the first PyTorch batch from the dataloader...")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    
    print("\n================ PREFLIGHT RESULTS ================")
    print("Batch keys and tensor shapes:")
    for k, v in batch.items():
        shape = v.shape if hasattr(v, 'shape') else type(v)
        print(f"  {k}: {shape}")
        
    print("\n--- SANITY CHECKS ---")
    if "pixel_values" in batch or "pixel_values_2d" in batch:
        print("✅ SUCCESS: Image tensors (pixel_values) are successfully reaching the PyTorch batch!")
    else:
        print("❌ FAIL: IMAGES NOT REACHING THE MODEL — ABORT (No pixel_values in batch)")
        
    if "prompt_input_ids" in batch: # TRL renames the prompt column
        prompt_len = batch["prompt_input_ids"].shape[-1]
    elif "input_ids" in batch:
        prompt_len = batch["input_ids"].shape[-1]
    else:
        prompt_len = 0
        
    print(f"Prompt sequence length: {prompt_len} tokens")
    if prompt_len > 500:
        print("✅ SUCCESS: Prompt length is > 500, which mathematically proves vision tokens exist!")
    else:
        print("❌ FAIL: Prompt length is too short. No image tokens in prompt — ABORT")
    print("===================================================")

if __name__ == "__main__":
    run_preflight()
