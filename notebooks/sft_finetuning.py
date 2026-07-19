"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SFT FINE-TUNING NOTEBOOK TEMPLATE                        ║
║                                                                              ║
║  Copy each "# %% [markdown]" and "# %%" cell block into separate Colab      ║
║  cells and run sequentially.                                                 ║
║                                                                              ║
║  This notebook leverages the robust SFT pipeline you built, meaning it:      ║
║  - Automatically handles Out-of-Memory (OOM) via batch size scaling          ║
║  - Immediately saves the best model to Drive during training                 ║
║  - Perfectly resumes from checkpoints and stitches W&B charts back together  ║
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
# Ensure your working directory is the root of the repository so imports work correctly.

# %%
import sys
import os

# If you cloned into a specific folder in Colab, adjust this path.
# Assuming you are running this from inside the `notebooks` directory:
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root set to: {PROJECT_ROOT}")

# Create directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)


# %% [markdown]
# ## 1. Authentication
# Log into Weights & Biases (for run tracking and resumption) and HuggingFace 
# (if you are using gated base models).

# %%
import wandb
from huggingface_hub import login

# Prompt for HuggingFace Token (Only needed if the base model is gated, like Qwen/Llama)
# login() 

# Prompt for W&B Token. This is CRITICAL for resuming runs cleanly.
wandb.login()


# %% [markdown]
# ## 2. Start or Resume Fine-Tuning
# 
# This cell executes the robust SFT pipeline. Under the hood, it does the following:
# 1. **Loads Configurations**: Reads `configs/sft.yaml` and `configs/model_registry.yaml` to configure hyperparameters (e.g., 4-bit loading, LoRA r=16, learning rate).
# 2. **Preprocesses Data**: Formats the dataset into Unsloth's exact `<|im_start|>` chat template and applies resolution bucketing to save memory.
# 3. **Auto-Resume**: Checks your Google Drive (`checkpoints/full_unified/unified-sft-v1`) for a `training_state.json`. If found, it automatically:
#     - Loads the latest model checkpoint.
#     - Re-attaches to the exact W&B run ID so your charts don't break.
# 4. **Safe Checkpointing**: Uses custom callbacks to instantly copy the model to a `best/` directory whenever evaluation improves, protecting your best weights from late-stage Colab crashes.
# 
# **To change models:** Change `tier="2b"` to `"4b"` or `"8b"`.  
# **To start a fresh ablation:** Change `variant="unified-sft-v1"` to something new like `"unified-sft-lora32"`.

# %%
from experiments.run_sft import main as run_sft_main
from unittest.mock import patch

# These arguments match the CLI: `python experiments/run_sft.py --tier 2b --variant unified-sft-v1`
args = [
    "run_sft.py", 
    "--tier", "2b", 
    "--variant", "unified-sft-v1"
    # "--no-resume" # Uncomment this ONLY if you want to explicitly overwrite and restart from scratch
]

print(f"Executing: {' '.join(args)}")

# Execute the pipeline
with patch("sys.argv", args):
    run_sft_main()

# %% [markdown]
# ## 3. (Optional) Run Evaluation on the Fine-Tuned Model
# 
# Once the training successfully finishes (or if you interrupt it and want to test the `best` checkpoint), 
# you can immediately evaluate it to get your BERTScore, Grounding IoU, and Violation F1 metrics.

# %%
# from experiments.compare_results import main as compare_main
# 
# # This will generate the Base vs SFT vs GRPO table
# test_args = ["compare_results.py", "--tier", "2b"]
# with patch("sys.argv", test_args):
#     compare_main()
