"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 EVALUATION METRICS DEMO NOTEBOOK                            ║
║                                                                              ║
║  Copy each "# %% [markdown]" and "# %%" cell block into separate Colab      ║
║  cells and run sequentially.                                                 ║
║                                                                              ║
║  This notebook demonstrates exactly how your predictions are scored against  ║
║  the ground truth, complete with expected mathematical results so you can    ║
║  verify the evaluation pipeline is perfectly accurate.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # 📊 Evaluation Metrics Demo & Verification
# 
# This notebook breaks down the evaluation pipeline. We feed it manually crafted 
# model predictions and ground truths, and then verify that the calculated 
# Precision, Recall, F1, and IoU match the expected mathematical results.

# %% [markdown]
# ## 0. Environment Setup

# %%
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root set to: {PROJECT_ROOT}")


# %% [markdown]
# ## 1. Structural & Parsing Metrics
# 
# The first step is testing if the model generates valid, parseable JSON that adheres 
# to your schema. We will feed one perfect response and one hallucinated string.
# 
# **Expected Result:**
# - JSON Validity: 50% (1/2)
# - Schema Adherence: 50% (1/2)

# %%
from evaluation.metrics_structural import compute_structural_metrics

raw_predictions = [
    # Sample 1: Perfect fenced JSON (Note: JSON uses "null", not Python's "None")
    '```json\n{"caption": "A safe site.", "rule_1_violation": null}\n```',
    
    # Sample 2: Model hallucinates a conversational response
    'I cannot see any violations in this image. It is safe.'
]

print("Computing Structural Metrics...")
structural_results = compute_structural_metrics(raw_predictions)

print("\n--- Structural Results ---")
for key, val in structural_results.items():
    print(f"{key}: {val}")

# Verifying calculations
assert structural_results["structural_json_validity_rate"] == 0.5
assert structural_results["structural_schema_adherence_rate"] == 0.5
print("\n✅ Verification Passed: Structural math is correct.")


# %% [markdown]
# ## 2. Object Grounding (Intersection over Union - IoU)
# 
# Now let's test how bounding boxes are scored. The system automatically normalizes 
# coordinates and calculates the overlap area.
# 
# **Scenario:**
# - **Image 1:** Exact perfect match. (IoU = 1.0)
# - **Image 2:** Model predicts a box `[0, 0, 100, 100]` (Area=10000). Ground truth is `[0, 0, 50, 100]` (Area=5000). Overlap is 5000. Union is 10000. 
#   **Expected IoU = 5000 / 10000 = 0.5**

# %%
from evaluation.metrics_grounding import compute_grounding_metrics

# Ground Truth references (The pipeline provides these in [0, 1] scale)
refs = [
    {"excavator": [[0.01, 0.01, 0.02, 0.02]]},  # Image 1
    {"excavator": [[0.0, 0.0, 0.05, 0.10]]}     # Image 2
]

# Parsed Model Predictions (The VLM outputs these in [0, 1000] scale)
preds = [
    {"excavator": [[10, 10, 20, 20]]},  # Image 1: Perfect Match (10/1000 = 0.01)
    {"excavator": [[0, 0, 100, 100]]}   # Image 2: Oversized box (Double width)
]

print("Computing Grounding IoU...")
grounding_results = compute_grounding_metrics(preds, refs)

iou_img1 = 1.0
iou_img2 = 0.5
expected_macro_iou = (iou_img1 + iou_img2) / 2.0

print("\n--- Grounding Results ---")
print(f"Excavator Macro IoU: {grounding_results['grounding_iou_all_macro_excavator']:.2f}")

# Verifying calculations
assert abs(grounding_results["grounding_iou_all_macro_excavator"] - expected_macro_iou) < 1e-5
print(f"\n✅ Verification Passed: Grounding IoU is correctly averaging to {expected_macro_iou:.2f}.")


# %% [markdown]
# ## 3. Safety Violation Metrics (Precision, Recall, F1)
# 
# This checks how well the model identifies rule violations using standard classification math.
# 
# **Scenario for Rule 1 (Hard Hat):**
# - Image 1: GT has violation. Model correctly detects it. (True Positive)
# - Image 2: GT has violation. Model misses it. (False Negative)
# - Image 3: GT is safe. Model hallucinates a violation. (False Positive)
# 
# **Math for Rule 1:**
# - Precision = TP / (TP + FP) = 1 / (1 + 1) = **0.50**
# - Recall = TP / (TP + FN) = 1 / (1 + 1) = **0.50**
# - F1 = 2 * (P * R) / (P + R) = **0.50**

# %%
from evaluation.metrics_violations import compute_violation_metrics

# GT Data
refs = [
    {"rule_1_violation": {"reason": "missing hat", "bounding_box": [0,0,1,1]}}, # Img 1: Unsafe
    {"rule_1_violation": {"reason": "missing hat", "bounding_box": [0,0,1,1]}}, # Img 2: Unsafe
    {"rule_1_violation": None}                                                  # Img 3: Safe
]

# Model Predictions
preds = [
    {"rule_1_violation": {"reason": "no hat", "bounding_box": [0,0,1,1]}},      # Img 1: Found it! (TP)
    {"rule_1_violation": None},                                                 # Img 2: Missed it! (FN)
    {"rule_1_violation": {"reason": "hallucination", "bounding_box": [0,0,1,1]}}# Img 3: Hallucinated! (FP)
]

print("Computing Violation Identification Metrics...")
violation_results = compute_violation_metrics(preds, refs)

print("\n--- Rule 1 Violation Results ---")
print(f"Precision: {violation_results['violation_identification_precision_rule_1']:.2f}")
print(f"Recall   : {violation_results['violation_identification_recall_rule_1']:.2f}")
print(f"F1 Score : {violation_results['violation_identification_f1_rule_1']:.2f}")

# Verifying calculations
assert violation_results['violation_identification_precision_rule_1'] == 0.5
assert violation_results['violation_identification_recall_rule_1'] == 0.5
assert violation_results['violation_identification_f1_rule_1'] == 0.5
print("\n✅ Verification Passed: Violation Precision/Recall/F1 math is correct.")


# %% [markdown]
# ## 4. Full Pipeline Integration
# 
# You can test a complete end-to-end evaluation pass simply by calling `run_full_evaluation`.
# Note: If you do not have Java installed (`!apt-get install -y default-jre`), the 
# official CIDEr/METEOR/SPICE captioning metrics will gracefully skip and return empty dicts.

# %%
from evaluation.evaluator import run_full_evaluation

raw = [
    '```json\n{"caption": "A safe construction site", "rule_1_violation": None, "excavator": []}\n```'
]
refs = [
    {"image_id": "test_01", "caption": "A safe site", "rule_1_violation": None, "excavator": []}
]

print("Running Full Pipeline...")
full_results = run_full_evaluation(raw, refs)

print("\nPipeline successfully completed! Keys calculated:")
for k, v in full_results["metrics"].items():
    if isinstance(v, float):
        print(f" - {k}: {v:.3f}")
    else:
        print(f" - {k}: {v}")
