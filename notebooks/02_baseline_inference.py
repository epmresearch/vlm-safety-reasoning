# ============================================================
# Cell 1: Setup Google Drive & Environment
# ============================================================
from google.colab import drive
import os

drive.mount('/content/drive')

# Put your tokens here
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
# Cell 3: Load Base Model and Dataset
# ============================================================
from core.constants import DEFAULT_MODEL_TIER
from data.loader import load_dataset_splits
from models.model_loader import load_model_for_inference

print("Loading dataset splits...")
splits = load_dataset_splits()
test_data = splits["test"]

print(f"Loading baseline model (tier: {DEFAULT_MODEL_TIER})...")
model, tokenizer, model_info = load_model_for_inference(tier=DEFAULT_MODEL_TIER)

# ============================================================
# Cell 4: Run Baseline Inference
# ============================================================
from models.inference import run_inference
import json
from core.io import get_drive_path, ensure_dir

# Use a small subset for quick testing (e.g. max_samples=10). Set to None for full evaluation.
MAX_SAMPLES = 50 

print(f"Running baseline inference on {MAX_SAMPLES} samples...")
results = run_inference(
    model=model,
    tokenizer=tokenizer,
    dataset=test_data,
    max_samples=MAX_SAMPLES
)

# ============================================================
# Cell 5: Evaluate Baseline Outputs
# ============================================================
from evaluation.evaluator import run_full_evaluation

raw_predictions = [res["raw_output"] for res in results]
# Convert test_data (HuggingFace Dataset) to list of dicts for evaluation
references = list(test_data.select(range(MAX_SAMPLES))) if MAX_SAMPLES else list(test_data)

print("Running unified evaluation pipeline...")
eval_results = run_full_evaluation(raw_predictions, references)

# Save results
output_dir = ensure_dir(get_drive_path("results", model_info["short_name"], "baseline"))
with open(output_dir / "metrics.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"Evaluation complete. Results saved to {output_dir}/metrics.json")
print("Metrics Summary:")
print(json.dumps(eval_results["structural"], indent=2))
