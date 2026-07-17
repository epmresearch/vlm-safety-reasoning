# ============================================================
# Cell 1: Setup and Loading
# ============================================================
import os
import json
from pathlib import Path

# Important: ensure your Drive is mounted in Colab before running this!
# from google.colab import drive
# drive.mount('/content/drive')

from data.loader import load_processed_dataset
from models.model_loader import load_model_for_inference
from models.inference import run_inference_batched
from evaluation.evaluator import run_full_evaluation
from core.run_manifest import save_run_manifest
from core.config import load_config

# 1. Define where to save results on your Drive
DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/Research_Results/baseline_2b")
ensure_dir(DRIVE_RESULTS_DIR)
JSONL_OUTPUT_PATH = str(DRIVE_RESULTS_DIR / "predictions.jsonl")

# 2. Load Core Configuration & Define Overrides
base_config = load_config(task="unified")

run_config = {
    "experiment": "baseline_inference",
    "model_tier": "2b",
    "batch_size": 32,  # Easy to override here!
    "max_new_tokens": base_config.get("max_new_tokens", 1000),
    "notes": "Colab batched auto-resume run",
    # Injecting the entire YAML state so your receipt tracks everything perfectly
    "full_yaml_state": base_config 
}
save_run_manifest(str(DRIVE_RESULTS_DIR), run_config)

# 3. Load the Dataset
print("Loading dataset...")
splits = load_processed_dataset()
test_data = splits["test"]

# 4. Load the Model (Only do this once!)
print("Loading model into VRAM...")
model, tokenizer, info = load_model_for_inference(tier="2b")
print("Setup complete. You are ready to run inference.")


# ============================================================
# Cell 2: Auto-Resume Batched Inference
# ============================================================
print(f"Starting batched inference. Results will stream to: {JSONL_OUTPUT_PATH}")

# If Colab disconnects, just re-run this exact cell! 
# It will read the JSONL file, skip the images it already processed, and pick up where it left off.
results = run_inference_batched(
    model=model,
    tokenizer=tokenizer,
    dataset=test_data,
    batch_size=run_config["batch_size"],
    max_new_tokens=run_config["max_new_tokens"],
    output_path=JSONL_OUTPUT_PATH
)

print("Inference Phase Complete!")


# ============================================================
# Cell 3: Fast Evaluation (Run anytime!)
# ============================================================
# You can run this cell even if Inference is only half done!
print(f"Loading raw predictions from {JSONL_OUTPUT_PATH}...")

raw_predictions = []
references = []

if not os.path.exists(JSONL_OUTPUT_PATH):
    print("No predictions found. Run Cell 2 first!")
else:
    with open(JSONL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    raw_predictions.append(record["raw_output"])
                    references.append(record["sample"])
                except Exception as e:
                    print(f"Skipping malformed line: {e}")

    print(f"Loaded {len(raw_predictions)} predictions.")
    
    # Run the evaluation pipeline (This takes 5-10 seconds, no GPU needed)
    print("Running evaluation metrics...")
    eval_results = run_full_evaluation(raw_predictions, references)
    
    # Save the metrics back to your Drive
    metrics_path = DRIVE_RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["metrics"], f, indent=2)
        
    # Save the Diagnostic Log for schema failures
    failures_path = DRIVE_RESULTS_DIR / "schema_failures.jsonl"
    with open(failures_path, "w", encoding="utf-8") as f:
        for failure in eval_results.get("failures", []):
            f.write(json.dumps(failure) + "\n")
            
    print(f"Evaluation complete! Metrics saved to: {metrics_path}")
    print(f"Diagnostic log saved to: {failures_path} ({len(eval_results.get('failures', []))} failures logged)")
    
    # Print a quick summary of the IoU
    print("\n--- QUICK METRIC SUMMARY ---")
    for key, val in eval_results["metrics"].items():
        if "grounding_iou" in key and "mean" in key:
            print(f"{key}: {val:.4f}")
