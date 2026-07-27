# CELL 1: Setup and Data Loading
import json
from datasets import load_from_disk
from collections import defaultdict

# Paths (Update if needed)
sft_path = "/content/drive/MyDrive/vlm-finetuning-project1/results/inference/unified-sft-v1_test/repair_applied/predictions_repaired.jsonl"
dataset_path = "/content/drive/MyDrive/vlm-finetuning-project1/datasets/processed"

print("Loading dataset...")
ds = load_from_disk(dataset_path)
test_ds = ds["test"]  # Assuming evaluation is on the test split

import re

def strip_fences(text):
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

print("Loading predictions...")
predictions = {}
with open(sft_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            img_id = str(record.get("image_id", ""))
            raw_str = record.get("raw_output", "")
            
            try:
                parsed_pred = json.loads(raw_str)
                predictions[img_id] = parsed_pred
            except Exception:
                try:
                    parsed_pred = json.loads(strip_fences(raw_str))
                    predictions[img_id] = parsed_pred
                except Exception:
                    pass # Skip unparseable

print(f"Loaded {len(predictions)} predictions.")

# -----------
# CELL 2: Extract Rule 1 Cases with Metadata
# We only care about images that ACTUALLY have a Rule 1 violation in Ground Truth

rule_1_cases = []

for example in test_ds:
    img_id = str(example.get("image_id", example.get("id", "")))
    
    if img_id not in predictions:
        continue
        
    pred = predictions[img_id]
    
    # Check GT
    has_gt_r1 = bool(example.get("rule_1_violation"))
    
    # Check Pred
    has_pred_r1 = bool(pred.get("rule_1_violation"))
    
    if has_gt_r1:
        # Determine if it was a Hit (True Positive) or Miss (False Negative)
        status = "Hit" if has_pred_r1 else "Miss"
        
        # Extract the 4 metadata fields
        rule_1_cases.append({
            "image_id": img_id,
            "status": status,
            "illumination": example.get("illumination", "unknown"),
            "camera_distance": example.get("camera_distance", "unknown"),
            "view": example.get("view", "unknown"),
            "quality_of_info": example.get("quality_of_info", "unknown")
        })

print(f"Total Rule 1 Ground Truth Cases Found: {len(rule_1_cases)}")

# -----------
# CELL 3: Stratified Analysis Logic

metadata_fields = ["illumination", "camera_distance", "view", "quality_of_info"]
stratified_stats = {field: defaultdict(lambda: {"hits": 0, "misses": 0, "total": 0}) for field in metadata_fields}

for case in rule_1_cases:
    for field in metadata_fields:
        val = str(case[field]).strip()
        if not val:
            val = "unknown"
            
        stratified_stats[field][val]["total"] += 1
        if case["status"] == "Hit":
            stratified_stats[field][val]["hits"] += 1
        else:
            stratified_stats[field][val]["misses"] += 1

# -----------
# CELL 4: Print Stratified Report

print("=== RULE 1 PERCEPTUAL DIFFICULTY (STRATIFIED ANALYSIS) ===\n")

for field in metadata_fields:
    print(f"--- Breakdown by: {field.upper()} ---")
    
    # Sort buckets by total cases (descending)
    sorted_buckets = sorted(stratified_stats[field].items(), key=lambda x: x[1]["total"], reverse=True)
    
    for bucket_name, stats in sorted_buckets:
        total = stats["total"]
        hits = stats["hits"]
        misses = stats["misses"]
        
        miss_rate = (misses / total) * 100 if total > 0 else 0
        hit_rate = (hits / total) * 100 if total > 0 else 0
        
        print(f"  {bucket_name:20} | Total GT: {total:<4} | Hits: {hits:<4} | Misses: {misses:<4} | Miss Rate: {miss_rate:>5.1f}%")
        
    print("-" * 65 + "\n")
