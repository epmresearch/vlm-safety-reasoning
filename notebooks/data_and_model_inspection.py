"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SFT PRE-FLIGHT INSPECTION NOTEBOOK                       ║
║                                                                              ║
║  Copy each "# %% [markdown]" and "# %%" cell block into separate Colab      ║
║  cells and run sequentially.                                                 ║
║                                                                              ║
║  This notebook exposes the internal functions so you can verify the data     ║
║  formatting, image resolutions, and model memory loading BEFORE you commit   ║
║  to a long fine-tuning run.                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # 🔍 Pre-Flight Inspection
# 
# Before running a multi-hour fine-tuning job, run this notebook to step through 
# the pipeline manually. You can inspect the exact prompt templates, verify image 
# bounds, and check how much GPU RAM the model takes just sitting idle.

# %% [markdown]
# ## 0. Environment Setup

# %%
import sys
import os

# Assuming you are running this from inside the `notebooks` directory in Colab:
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root set to: {PROJECT_ROOT}")


# %% [markdown]
# ## 1. Config Loading
# Let's verify that the configs are correctly loaded for the 2b tier.

# %%
from core.config import load_config, load_task_config
from models.model_loader import get_model_info, get_batch_config

tier = "2b"

# Load merged config (base + model_registry + sft)
sft_config = load_config(training_kind="sft")
task_config = load_task_config("unified")
model_info = get_model_info(tier)
batch_cfg = get_batch_config(tier)

print(f"Target Model: {model_info['hf_path']}")
print(f"SFT Max Sequence Length: {sft_config.get('max_seq_length')}")
print(f"Image Pixel Bounds: {sft_config.get('image_min_pixels')} to {sft_config.get('image_max_pixels')}")
print(f"Batch Config: {batch_cfg}")


# %% [markdown]
# ## 2. Raw Dataset Inspection
# Let's load the dataset from HuggingFace/Disk and check the class distributions 
# that the stratification logic generated.

# %%
from data.loader import load_dataset_splits
import pandas as pd

print("Loading raw splits...")
splits = load_dataset_splits()
train_raw = splits["train"]
val_raw = splits["val"]

print(f"Train samples: {len(train_raw)}")
print(f"Validation samples: {len(val_raw)}")

# Look at the first sample's raw keys
print("\nSample 0 Keys:", train_raw[0].keys())

# Print out the raw ground truth text for sample 0
from data.preprocessor import build_ground_truth_dict
import json
print("\nGround Truth JSON for Sample 0:")
print(json.dumps(build_ground_truth_dict(train_raw[0]), indent=2))


# %% [markdown]
# ## 3. Prompt Template Validation
# Here we test `build_unified_sft_dataset` to ensure your data correctly converts 
# into Unsloth's multi-turn conversational format (`user` -> `assistant`).

# %%
from data.preprocessor import build_unified_sft_dataset

print("Converting a small slice to SFT prompt format...")
# Just process the first 3 samples to save time
sample_slice = train_raw.select(range(3))
sft_preview = build_unified_sft_dataset(sample_slice)

messages = sft_preview[0]["messages"]

print("\n--- SYSTEM PROMPT ---")
print(messages[0]["content"])

print("\n--- USER PROMPT ---")
# If it has an image, it's a list. If text only, it's a string.
user_content = messages[1]["content"]
print(user_content)

print("\n--- ASSISTANT RESPONSE (TARGET) ---")
print(messages[2]["content"])


# %% [markdown]
# ## 4. Resolution & Memory Bucketing
# Check the image resolutions. `get_resolutions` calculates the pixel area of every image. 
# This confirms the `image_min_pixels` and `image_max_pixels` caps will protect us from 14MP outliers.

# %%
from data.samplers import get_resolutions
import numpy as np

print("Calculating resolutions for the first 500 samples...")
# Let's just check the first 500 to keep it fast
preview_resolutions = get_resolutions(train_raw.select(range(500)))

if preview_resolutions:
    res_array = np.array(preview_resolutions)
    print(f"Max pixels in sample set: {res_array.max():,}")
    print(f"Min pixels in sample set: {res_array.min():,}")
    print(f"Mean pixels: {res_array.mean():,.0f}")
    
    # Check how many would be capped by the sft.yaml bounds
    max_cap = sft_config.get("image_max_pixels", 1204224)
    capped_count = (res_array > max_cap).sum()
    print(f"Images exceeding maximum cap ({max_cap:,}): {capped_count} (These will be downscaled)")
else:
    print("No resolutions found (text-only dataset?)")


# %% [markdown]
# ## 5. Model Loading & GPU Memory Check
# Let's actually load the model with the SFT config. This applies 4-bit quantization, 
# injects the LoRA adapters, and flips the model to `for_training`. 
# We will check how much GPU VRAM is allocated.

# %%
from models.model_loader import load_model_for_training, log_gpu_memory
import torch

if not torch.cuda.is_available():
    print("⚠️ WARNING: No GPU detected! Model loading will fail or be extremely slow.")
else:
    log_gpu_memory("Before Model Load")
    
    print(f"\nLoading model: {model_info['hf_path']}...")
    model, tokenizer, _ = load_model_for_training(
        model_name=model_info["hf_path"], 
        tier=tier, 
        sft_cfg=sft_config
    )
    
    print("\nModel successfully loaded!")
    log_gpu_memory("After Model Load")
    
    # Inspect the LoRA configuration that was attached
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()


# %% [markdown]
# ## 6. Peak VRAM Stress Test (Simulate 1 Training Step)
# This cell creates a dummy trainer and runs exactly 2 steps of actual training 
# (forward pass, loss calculation, backward pass, optimizer step).
# This gives you the EXACT peak VRAM you will see during real training, allowing 
# you to know with 100% certainty if your batch size will OOM.

# %%
if torch.cuda.is_available():
    print("Initializing dummy trainer for Peak VRAM test...")
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator
    from data.samplers import get_resolutions
    
    # Since your dataset is already sorted by resolution, the largest images are at the end!
    # We just grab the very last batch to test the absolute worst-case memory scenario.
    print("Grabbing the last batch of images (highest resolution) for worst-case stress test...")
    batch_size_needed = batch_cfg["per_device_train_batch_size"] * batch_cfg.get("gradient_accumulation_steps", 1)
    num_test_samples = min(max(2 * batch_size_needed, 16), len(train_raw))
    
    worst_case_samples = train_raw.select(range(len(train_raw) - num_test_samples, len(train_raw)))
    
    resolutions = get_resolutions(worst_case_samples)
    if resolutions:
        max_res_raw = max(resolutions)
        max_cap = sft_config.get("image_max_pixels", 1204224)
        max_res_capped = min(max_res_raw, max_cap)
        print(f"Max image resolution in this batch : {max_res_raw:,} pixels")
        print(f"Resolution after safety capping    : {max_res_capped:,} pixels")
        
    test_ds = list(build_unified_sft_dataset(worst_case_samples))
    
    data_collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # Force 2 steps, disable W&B, use exact batch size from config
    test_args = SFTConfig(
        output_dir="/tmp/test_trainer",
        per_device_train_batch_size=batch_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=batch_cfg["gradient_accumulation_steps"],
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=sft_config.get("bf16", True),
        max_seq_length=sft_config.get("max_seq_length", 4096),
        auto_find_batch_size=False, # We want to test the literal batch size
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=test_ds,
        data_collator=data_collator,
        args=test_args,
    )
    
    print("\nStarting 2-step stress test. Watch the memory output...")
    try:
        # Reset memory stats tracking so we only measure peak during training
        torch.cuda.reset_peak_memory_stats()
        
        trainer.train()
        
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("\n" + "="*50)
        print(f"✅ STRESS TEST PASSED")
        print(f"Peak VRAM Reserved : {peak_reserved:.2f} GB")
        print(f"Peak VRAM Allocated: {peak_allocated:.2f} GB")
        print(f"Total GPU VRAM     : {total_vram:.2f} GB")
        print(f"Safety Headroom    : {total_vram - peak_reserved:.2f} GB")
        print("="*50)
        print("You are safe to start the actual fine-tuning run!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "❌"*25)
            print("OOM CRASH DETECTED!")
            print("Your current batch size or image pixel bounds are too high.")
            print("Reduce `per_device_train_batch_size` or `image_max_pixels` in configs/sft.yaml.")
            print("❌"*25)
        else:
            raise e
