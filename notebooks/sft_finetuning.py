"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SFT FINE-TUNING NOTEBOOK TEMPLATE                        ║
║                                                                              ║
║  Copy each "# %% [markdown]" and "# %%" cell block into separate Colab      ║
║  cells and run sequentially.                                                 ║
║                                                                              ║
║  This notebook breaks down the SFT pipeline so you have full control over    ║
║  data loading, preprocessing, and the training loop while retaining          ║
║  auto-resume and OOM protections.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # 🚀 Vision-Language Model SFT Fine-Tuning
# 
# This notebook fine-tunes the VLM using PEFT/LoRA. It is designed to be highly 
# resilient in the Colab environment. If your session disconnects or crashes, 
# simply run this notebook again, and it will pick up exactly where it left off.

# %% [markdown]
# ## 0. Environment Setup

# %%
import sys
import os

# If you cloned into a specific folder in Colab, adjust this path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root set to: {PROJECT_ROOT}")

# Create directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)


# %% [markdown]
# ## 1. Authentication
# Log into Weights & Biases (for run tracking and resumption).

# %%
import wandb
from huggingface_hub import login

# login() # Prompt for HuggingFace Token (Only needed if the base model is gated)
wandb.login() # Prompt for W&B Token. CRITICAL for resuming runs cleanly.


# %% [markdown]
# ## 2. Load Raw Datasets
# We first load the raw dataset splits. This gives you a chance to inspect or 
# filter the data before handing it over to the trainer.

# %%
from data.loader import load_processed_dataset
from data.samplers import get_resolutions

print("Loading pre-processed, stratified, and sorted dataset splits...")
splits = load_processed_dataset()
train_raw = splits["train"]
val_raw = splits["val"]

print(f"Loaded {len(train_raw)} training samples and {len(val_raw)} validation samples.")

# Pull the pre-calculated resolutions (from your earlier notebook)
if "resolution" in train_raw.column_names:
    print("Using pre-calculated 'resolution' column for memory safety bucketing...")
    train_resolutions = train_raw["resolution"]
else:
    print("Calculating image resolutions for memory safety bucketing...")
    train_resolutions = get_resolutions(train_raw)


# %% [markdown]
# ## 3. Preprocess for Unsloth
# We now convert the raw JSON logic into the exact `<|im_start|>` conversational 
# templates required by the Qwen VLM.

# %%
from data.preprocessor import build_unified_sft_dataset

print("Formatting dataset for Unified SFT...")
train_ds = build_unified_sft_dataset(train_raw)
val_ds = build_unified_sft_dataset(val_raw)

# Safety check for bucketing
if train_resolutions is not None and len(train_resolutions) != len(train_ds):
    print(f"⚠️ Resolution count ({len(train_resolutions)}) != training samples ({len(train_ds)}).")
    print("Disabling resolution bucketing for safety.")
    train_resolutions = None

# Convert to standard Python lists for the trainer
train_ds = list(train_ds)
val_ds = list(val_ds)
print("Preprocessing complete!")


# %% [markdown]
# ## 4. Start or Resume Fine-Tuning
# This triggers the actual training loop. It will automatically:
# 1. Check your Drive for existing checkpoints.
# 2. If a checkpoint exists, seamlessly resume the training state AND the W&B charts.
# 3. Apply memory bounds (from `configs/sft.yaml`) to prevent OOM on massive images.
# 4. Save the absolute best adapter periodically to a `best/` directory on your Drive.

# %%
from models.sft_trainer import run_sft_unified

tier = "2b"
variant = "unified-sft-v1"  # Change this to e.g. "unified-sft-lora32" for ablations
resume_training = True      # Set False only if you want to overwrite and restart from scratch

print(f"Starting SFT for tier: {tier}, variant: {variant}...")

checkpoint_dir = run_sft_unified(
    tier=tier,
    variant=variant,
    train_dataset=train_ds,
    val_dataset=val_ds,
    train_resolutions=train_resolutions,
    resume=resume_training,
)

print(f"🎉 SFT run complete. Best/final checkpoint saved at {checkpoint_dir}")
