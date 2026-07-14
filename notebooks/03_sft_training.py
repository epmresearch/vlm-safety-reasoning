# ============================================================
# Cell 1: Setup Google Drive & Environment
# ============================================================
from google.colab import drive
import os

drive.mount('/content/drive')

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_TOKEN"
PROJECT_ROOT = "/content/drive/MyDrive/vlm-finetuning-project1/vlm-safety-reasoning"

# ============================================================
# Cell 2: Install Dependencies
# ============================================================
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft wandb pycocotools bert-score
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ============================================================
# Cell 3: Data Preprocessing
# ============================================================
from data.loader import load_dataset_splits
from data.preprocessor import build_unified_sft_dataset

print("Loading raw dataset splits...")
splits = load_dataset_splits()

print("Building unified SFT dataset...")
train_ds = build_unified_sft_dataset(splits["train"])
val_ds = build_unified_sft_dataset(splits["val"])

print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# ============================================================
# Cell 4: Run SFT Training
# ============================================================
from models.sft_trainer import run_sft_unified
from core.constants import DEFAULT_MODEL_TIER

VARIANT_NAME = "unified-sft-v1"

print(f"Starting SFT for {DEFAULT_MODEL_TIER} | Variant: {VARIANT_NAME}")
checkpoint_dir = run_sft_unified(
    tier=DEFAULT_MODEL_TIER,
    variant=VARIANT_NAME,
    train_dataset=list(train_ds),
    val_dataset=list(val_ds),
    resume=True
)
print(f"Training finished! Checkpoint saved to: {checkpoint_dir}")
