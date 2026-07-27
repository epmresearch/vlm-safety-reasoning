# ==============================================================================
# NOTEBOOK 1: GENERATION (Repetition Penalty Ablation)
# ==============================================================================

# ------------------
# CELL 1.0: Setup, Clone Repo, and Install Dependencies
import os
import subprocess
import shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DRIVE_ROOT = "/content/drive/MyDrive/vlm-finetuning-project1"
REPO_DIR = "vlm-safety-reasoning"
ENV_PATH = f"{DRIVE_ROOT}/secrets/.env"

def load_secrets(env_path: str) -> dict:
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Secrets file not found at: {env_path}")
    secrets = {}
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            secrets[key] = value.strip(" \"'\r")
            os.environ[key] = secrets[key]
    return secrets

print(">>> Loading secrets...")
secrets = load_secrets(ENV_PATH)

print(">>> Configuring Git identity...")
subprocess.run(["git", "config", "--global", "user.email", secrets["GIT_EMAIL"]], check=True)
subprocess.run(["git", "config", "--global", "user.name", secrets["GIT_NAME"]], check=True)

AUTH_REPO_URL = f"https://{secrets['GITHUB_USERNAME']}:{secrets['GITHUB_TOKEN']}@github.com/epmresearch/vlm-safety-reasoning.git"

if os.path.exists(REPO_DIR):
    os.chdir(REPO_DIR)
    subprocess.run(["git", "remote", "set-url", "origin", AUTH_REPO_URL], check=True)
    subprocess.run(["git", "pull", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", AUTH_REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)

shutil.copy(ENV_PATH, ".env")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print(">>> Setup complete. CWD:", os.getcwd())

import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# ------------------
# CELL 1.1: Load Data and Stratified Sampling
import random
from datasets import load_from_disk

dataset_path = f"{DRIVE_ROOT}/datasets/processed"
print("Loading dataset...")
ds = load_from_disk(dataset_path)
test_ds = ds["test"]

# Stratified Sampling: Ensure all rules and safe cases are represented
rule_buckets = {f"rule_{i}": [] for i in range(1, 5)}
rule_buckets["safe"] = []

for idx, sample in enumerate(test_ds):
    rules_present = [f"rule_{i}" for i in range(1, 5) if sample.get(f"rule_{i}_violation")]
    if not rules_present:
        rule_buckets["safe"].append(idx)
    else:
        # Assign to the rarest rule present to ensure rare rules get filled
        for r in ["rule_4", "rule_3", "rule_2", "rule_1"]:
            if r in rules_present:
                rule_buckets[r].append(idx)
                break

selected_indices = []
random.seed(42)
for bucket, indices in rule_buckets.items():
    # Target ~40-50 per bucket to get roughly 250 total
    target = min(45, len(indices))
    selected = random.sample(indices, target)
    selected_indices.extend(selected)

subset_ds = test_ds.select(selected_indices)
print(f"Stratified subset size: {len(subset_ds)} images.")
for bucket, indices in rule_buckets.items():
    print(f"  {bucket}: {min(45, len(indices))} samples")

# ------------------
# CELL 1.2: Load Model
# We use the codebase's model_loader to properly handle PEFT adapters and pixel bounds!
from models.model_loader import load_model_for_inference
import torch

# IMPORTANT: Update adapter_path to your best checkpoint!
base_model_name = "unsloth/Qwen2-VL-2B-Instruct"
adapter_path = f"{DRIVE_ROOT}/checkpoints/qwen3vl-2b/unified-sft-v1/best"

print(f"Loading model with adapter from {adapter_path}...")
model, tokenizer, _ = load_model_for_inference(
    model_name=base_model_name,
    adapter_path=adapter_path
)

# ------------------
# CELL 1.3: Custom Inference Function (repetition_penalty = 1.0)
# We redefine this here because the codebase models/inference.py hardcodes repetition_penalty=1.15
import time
import json
import gc
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT

def generate_batch_rep_1(model, tokenizer, pil_images, max_new_tokens=1000):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": UNIFIED_INSPECTION_PROMPT},
            ]},
        ]
        for img in pil_images
    ]

    texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch_messages]

    all_image_inputs = []
    for conv in batch_messages:
        img_in, _ = process_vision_info(conv)
        if img_in:
            all_image_inputs.extend(img_in if isinstance(img_in, list) else [img_in])

    inputs = tokenizer(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.0,  # <--- REPETITION PENALTY DISABLED!
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)

# ------------------
# CELL 1.4: Run Inference and Save
output_path = f"{DRIVE_ROOT}/results/ablation_rep_1.0.jsonl"
batch_size = 8

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

results = []
for start in tqdm(range(0, len(subset_ds), batch_size), desc="Inferencing rep=1.0"):
    batch = subset_ds.select(range(start, min(start + batch_size, len(subset_ds))))
    pil_images = batch["image"]
    
    start_time = time.time()
    outputs = generate_batch_rep_1(model, tokenizer, pil_images)
    elapsed = time.time() - start_time
    
    for i, sample in enumerate(batch):
        res = {
            "image_id": sample["image_id"],
            "raw_output": outputs[i],
            "latency_seconds": elapsed / len(pil_images)
        }
        results.append(res)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
            
    gc.collect()
    torch.cuda.empty_cache()

print(f"Saved {len(results)} predictions to {output_path}")


# ==============================================================================
# NOTEBOOK 2: REPAIR & METRICS CALCULATION
# ==============================================================================

# ------------------
# CELL 2.1: Setup and Load Data
import json
import numpy as np
import re
from datasets import load_from_disk
# IMPORTING FROM CODEBASE
from core.constants import RULES
from evaluation.metrics_captioning import compute_all_caption_metrics
from preprocessing.structural_repair import repair_and_validate

# Make sure DRIVE_ROOT is defined if running Notebook 2 in a fresh session
try:
    DRIVE_ROOT
except NameError:
    DRIVE_ROOT = "/content/drive/MyDrive/vlm-finetuning-project1"

dataset_path = f"{DRIVE_ROOT}/datasets/processed"
output_path = f"{DRIVE_ROOT}/results/ablation_rep_1.0.jsonl"

print("Loading GT subset...")
ds = load_from_disk(dataset_path)
test_ds = ds["test"]

predictions_raw = {}
with open(output_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            predictions_raw[str(record["image_id"])] = record["raw_output"]

# Filter GT to only the ones we ran inference on
gt_subset = [s for s in test_ds if str(s["image_id"]) in predictions_raw]
print(f"Loaded {len(gt_subset)} samples for evaluation.")

# ------------------
# CELL 2.2: Structural Repair
def strip_fences(text):
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

parsed_predictions = {}
count_valid = 0
count_repaired = 0
count_failed = 0
repair_stats = {}

for sample in gt_subset:
    img_id = str(sample["image_id"])
    raw_str = predictions_raw[img_id]
    
    # USING CODEBASE REPAIR LOGIC
    result = repair_and_validate(raw_str)
    
    if result.get("changes"):
        for change in result["changes"]:
            c_type = change.get("type", "unknown_fix")
            repair_stats[c_type] = repair_stats.get(c_type, 0) + 1
            
    if result["status"] == "fixed_valid":
        parsed_predictions[img_id] = result["fixed_parsed"]
        count_repaired += 1
    elif result["status"] == "valid_raw":
        try:
            parsed_predictions[img_id] = json.loads(strip_fences(raw_str))
            count_valid += 1
        except Exception as e:
            print(f"Failed to JSON parse valid_raw for {img_id}: {e}")
            parsed_predictions[img_id] = {}
            count_failed += 1
    else:
        print(f"Failed to parse (status: {result['status']}) for {img_id}")
        parsed_predictions[img_id] = {}
        count_failed += 1

print("\n=== REPAIR REPORT ===")
print(f"Already Valid (No repair needed): {count_valid}")
print(f"Successfully Repaired:            {count_repaired}")
print(f"Failed completely:                {count_failed}")
print(f"Total Successful Parses:          {sum(1 for v in parsed_predictions.values() if v)}")

if repair_stats:
    print("\n--- Specific Repairs Applied ---")
    for r_type, count in repair_stats.items():
        print(f"{r_type}: {count} times")

# ------------------
# CELL 2.3: Data Extraction & Length Percentiles
pred_captions, gt_captions, caption_images = [], [], []
pred_reasons, gt_reasons, reason_images = [], [], []

for sample in gt_subset:
    img_id = str(sample["image_id"])
    pred = parsed_predictions.get(img_id, {})
    
    # 1. Extract Captions
    if pred.get("caption"):
        pred_captions.append(pred["caption"])
        gt_captions.append(sample["image_caption"])
        caption_images.append(sample["image"])
        
    # 2. Extract Reasons
    for r in RULES:
        if pred.get(f"{r}_violation") and sample.get(f"{r}_violation"):
            p_reason = pred[f"{r}_violation"].get("reason", "")
            g_reason = sample[f"{r}_violation"].get("reason", "")
            if p_reason and g_reason:
                pred_reasons.append(p_reason)
                gt_reasons.append(g_reason)
                reason_images.append(sample["image"])

print("=== CAPTION LENGTH METRICS (Repetition Penalty = 1.0) ===")
word_counts_c = [len(c.split()) for c in pred_captions]
if word_counts_c:
    print(f"Mean Words: {np.mean(word_counts_c):.1f}")
    print(f"25th Pct:   {np.percentile(word_counts_c, 25):.1f}")
    print(f"50th Pct:   {np.percentile(word_counts_c, 50):.1f}")
    print(f"75th Pct:   {np.percentile(word_counts_c, 75):.1f}")
    print(f"99th Pct:   {np.percentile(word_counts_c, 99):.1f}")
    print(f"Max Words:  {np.max(word_counts_c)}")

print("\n=== REASON LENGTH METRICS (Repetition Penalty = 1.0) ===")
word_counts_r = [len(r.split()) for r in pred_reasons]
if word_counts_r:
    print(f"Mean Words: {np.mean(word_counts_r):.1f}")
    print(f"25th Pct:   {np.percentile(word_counts_r, 25):.1f}")
    print(f"50th Pct:   {np.percentile(word_counts_r, 50):.1f}")
    print(f"75th Pct:   {np.percentile(word_counts_r, 75):.1f}")
    print(f"99th Pct:   {np.percentile(word_counts_r, 99):.1f}")
    print(f"Max Words:  {np.max(word_counts_r)}")

# ------------------
# CELL 2.4: Codebase NLP Metrics
print("=== NLP METRICS: CAPTIONS ===")
cap_metrics = compute_all_caption_metrics(pred_captions, gt_captions, images=caption_images, include_spice=False)
for key, value in cap_metrics.items():
    print(f"{key}: {value:.4f}")

print("\n=== NLP METRICS: REASONS ===")
if pred_reasons:
    res_metrics = compute_all_caption_metrics(pred_reasons, gt_reasons, images=reason_images, include_spice=False)
    for key, value in res_metrics.items():
        print(f"{key}: {value:.4f}")
else:
    print("No matching True Positive reasons found.")

# ------------------
# CELL 2.5: Full Pipeline Metrics (Grounding & Violations)
from evaluation.metrics_grounding import compute_grounding_metrics
from evaluation.metrics_violations import compute_violation_metrics

# Re-align predictions and ground truth into lists of dictionaries
pred_list = []
gt_list = []

for sample in gt_subset:
    img_id = str(sample["image_id"])
    pred = parsed_predictions.get(img_id, {})
    pred_list.append(pred)
    gt_list.append(sample)

print("\n=== SAFETY VIOLATION IDENTIFICATION (Recall/Precision) ===")
violation_metrics = compute_violation_metrics(pred_list, gt_list)
for k, v in violation_metrics.items():
    if "macro" in k or "micro" in k:  # Print the high-level summaries
        print(f"{k}: {v:.4f}")

print("\n=== GROUNDING (BBOX) METRICS (IoU/Recall) ===")
grounding_metrics = compute_grounding_metrics(pred_list, gt_list)
for k, v in grounding_metrics.items():
    if "macro" in k or "micro" in k:
        print(f"{k}: {v:.4f}")

# ------------------
# CELL 2.6: Save All Metrics to JSON
import os
import json
import numpy as np

# Convert the input path (e.g. ablation_rep_1.0.jsonl) into a metrics save path
metrics_save_path = output_path.replace(".jsonl", "_metrics.json").replace(".json", "_metrics.json")

# Helper to safely extract lengths
def get_length_stats(counts):
    if not counts: return {}
    return {
        "mean": float(np.mean(counts)),
        "p25": float(np.percentile(counts, 25)),
        "p50": float(np.percentile(counts, 50)),
        "p75": float(np.percentile(counts, 75)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(np.max(counts))
    }

final_report = {
    "repair_stats": {
        "already_valid": count_valid,
        "repaired": count_repaired,
        "failed": count_failed,
        "specific_fixes": repair_stats
    },
    "length_metrics": {
        "captions": get_length_stats(word_counts_c),
        "reasons": get_length_stats(word_counts_r)
    },
    "nlp_metrics_captions": cap_metrics,
    "nlp_metrics_reasons": res_metrics if 'res_metrics' in locals() else {},
    "violation_metrics": violation_metrics,
    "grounding_metrics": grounding_metrics
}

# Save to disk
os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
with open(metrics_save_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=4)

print(f"\n✅ Successfully saved all aggregated metrics to:\n{metrics_save_path}")
