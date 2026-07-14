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
# Cell 3: Note on GRPO Phase
# ============================================================
print("Note: GRPO/GSPO training for the unified task is scheduled for a future phase.")
print("The reward functions and RL trainer pipeline need to be adapted to the unified prompt.")
# When implemented, you would load the SFT checkpoint here and launch the GRPO trainer.
