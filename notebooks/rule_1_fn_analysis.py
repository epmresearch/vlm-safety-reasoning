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
                # Try parsing raw (works for the 35 fixed_valid ones without fences)
                parsed_pred = json.loads(raw_str)
                predictions[img_id] = parsed_pred
            except Exception:
                # Try stripping fences (works for the 2969 valid_raw ones WITH fences)
                try:
                    parsed_pred = json.loads(strip_fences(raw_str))
                    predictions[img_id] = parsed_pred
                except Exception:
                    pass # Truly unparseable

print(f"Loaded {len(predictions)} predictions.")

# -----------
# CELL 2: Extracting Rule 1 False Negatives
# A False Negative (FN) means Ground Truth HAS Rule 1, but Model PREDICTED NO Rule 1.

rule_1_fn_cases = []
rule_1_tp_cases = [] # True Positives for comparison

for example in test_ds:
    img_id = str(example.get("image_id", example.get("id", "")))
    
    if img_id not in predictions:
        continue
        
    pred = predictions[img_id]
    
    # Check GT
    gt_r1 = example.get("rule_1_violation")
    has_gt_r1 = bool(gt_r1)
    
    # Check Pred
    pred_r1 = pred.get("rule_1_violation")
    has_pred_r1 = bool(pred_r1)
    
    if has_gt_r1:
        # Extract the ground truth reason string
        gt_reason = gt_r1.get("reason", "") if isinstance(gt_r1, dict) else str(gt_r1)
        
        if not has_pred_r1:
            rule_1_fn_cases.append({"image_id": img_id, "reason": gt_reason})
        else:
            rule_1_tp_cases.append({"image_id": img_id, "reason": gt_reason})

print(f"Total Rule 1 Ground Truth Cases: {len(rule_1_fn_cases) + len(rule_1_tp_cases)}")
print(f"Rule 1 True Positives (Hits): {len(rule_1_tp_cases)}")
print(f"Rule 1 False Negatives (Misses): {len(rule_1_fn_cases)}")
print(f"Rule 1 Recall: {len(rule_1_tp_cases) / (len(rule_1_fn_cases) + len(rule_1_tp_cases)):.1%}")

# -----------
# CELL 3: Categorize by PPE Sub-Type via Keywords
# We will define keyword clusters to automatically classify the GT reason strings.

categories = {
    "Hard Hat": ["hat", "helmet", "head"],
    "Footwear": ["shoe", "boot", "foot", "feet", "toe"],
    "Gloves": ["glove", "hand"],
    "High-Vis / Vest": ["vest", "high-vis", "visibility", "reflective", "jacket", "night", "dark"],
    "Face / Eye Protection": ["face", "mask", "goggle", "shield", "glass", "weld", "grind", "drill", "cut", "eye"],
    "Clothing (General)": ["shirt", "pant", "sleeve", "cloth", "short", "bare", "topless"],
}

def categorize_reason(reason_str):
    reason_lower = reason_str.lower()
    matched_cats = []
    
    for cat_name, keywords in categories.items():
        if any(kw in reason_lower for kw in keywords):
            matched_cats.append(cat_name)
            
    if not matched_cats:
        return ["Other / Unspecified"]
    return matched_cats

# Analyze False Negatives (Where the model failed)
fn_category_counts = defaultdict(int)
fn_examples_by_cat = defaultdict(list)

for case in rule_1_fn_cases:
    cats = categorize_reason(case["reason"])
    for c in cats:
        fn_category_counts[c] += 1
        if len(fn_examples_by_cat[c]) < 3: # Save up to 3 examples for display
            fn_examples_by_cat[c].append(case)
            
# Analyze True Positives (Where the model succeeded, for comparison)
tp_category_counts = defaultdict(int)
for case in rule_1_tp_cases:
    cats = categorize_reason(case["reason"])
    for c in cats:
        tp_category_counts[c] += 1

# -----------
# CELL 4: Print Statistical Report & Examples

print("=== RULE 1 FALSE NEGATIVE ANALYSIS (Where the model failed) ===")
total_fns = len(rule_1_fn_cases)

# Sort categories by highest failure count
sorted_fn_cats = sorted(fn_category_counts.items(), key=lambda x: x[1], reverse=True)

for cat, count in sorted_fn_cats:
    pct = (count / total_fns) * 100
    print(f"\n[{cat}]: {count} Misses ({pct:.1f}% of total Rule 1 misses)")
    
    # Print examples
    print("  Examples of GT Reasons the model missed:")
    for ex in fn_examples_by_cat[cat]:
        print(f"    - (Img {ex['image_id']}): {ex['reason']}")
        
print("\n" + "="*50 + "\n")
print("=== COMPARISON: MISS RATE BY PPE SUB-TYPE ===")
# Compute recall per sub-category (How many Hard Hat cases did we hit vs miss?)
all_categories = set(fn_category_counts.keys()).union(set(tp_category_counts.keys()))

for cat in sorted(all_categories):
    tp = tp_category_counts.get(cat, 0)
    fn = fn_category_counts.get(cat, 0)
    total = tp + fn
    
    recall = (tp / total) * 100 if total > 0 else 0
    miss_rate = (fn / total) * 100 if total > 0 else 0
    
    print(f"{cat:25} | Total GT: {total:<4} | Hits: {tp:<4} | Misses: {fn:<4} | Miss Rate: {miss_rate:>5.1f}%")
