import pytest
import math

from evaluation.metrics_violations import compute_violation_metrics

def test_empty_inputs():
    """Test that empty/invalid inputs raise ValueError (fail-fast, N4 fix)."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_violation_metrics([], [])
    with pytest.raises(ValueError, match="non-empty"):
        compute_violation_metrics(None, None)
    with pytest.raises(ValueError, match="length mismatch"):
        compute_violation_metrics([{"rule_1_violation": {}}, {}], [{}])  # Length mismatch

def test_rule_0_metrics():
    """Test that Rule 0 (no violations) is correctly identified."""
    # Image 1: Safe (GT=None). Model predicts Safe (Pred=None) -> Rule 0 TP
    # Image 2: Safe (GT=None). Model hallucinates Rule 1 -> Rule 0 FN, Rule 1 FP
    # Image 3: Unsafe (GT=Rule 2). Model predicts Safe -> Rule 0 FP, Rule 2 FN
    
    refs = [
        {}, # Img 1: Safe
        {}, # Img 2: Safe
        {"rule_2_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}} # Img 3: Unsafe
    ]
    preds = [
        {}, # Img 1: Safe
        {"rule_1_violation": {"reason": "x", "bounding_box": [[0,0,1000,1000]]}}, # Img 2: Hallucination
        {} # Img 3: Missed
    ]
    
    res = compute_violation_metrics(preds, refs)
    
    # --- Rule 0 Analysis ---
    # Img 1: TP
    # Img 2: FN (True image is safe, model said unsafe)
    # Img 3: FP (True image is unsafe, model said safe)
    # Rule 0 Precision = 1 TP / (1 TP + 1 FP) = 0.5
    # Rule 0 Recall = 1 TP / (1 TP + 1 FN) = 0.5
    
    assert res["violation_identification_precision_rule_0"] == 0.5
    assert res["violation_identification_recall_rule_0"] == 0.5
    assert res["violation_identification_f1_rule_0"] == 0.5

def test_standard_rule_metrics():
    """Test standard TP, FP, FN calculation for Rules 1-4."""
    # Img 1: GT has Rule 1. Model predicts Rule 1. (Rule 1 TP)
    # Img 2: GT has Rule 2. Model predicts Rule 3. (Rule 2 FN, Rule 3 FP)
    
    refs = [
        {"rule_1_violation": {"bounding_box": [[0,0,1,1]]}},
        {"rule_2_violation": {"bounding_box": [[0,0,1,1]]}}
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0,0,1000,1000]]}},
        {"rule_3_violation": {"bounding_box": [[0,0,1000,1000]]}}
    ]
    
    res = compute_violation_metrics(preds, refs)
    
    # Rule 1: 1 TP, 0 FP, 0 FN -> P=1.0, R=1.0
    assert res["violation_identification_precision_rule_1"] == 1.0
    assert res["violation_identification_recall_rule_1"] == 1.0
    
    # Rule 2: 0 TP, 0 FP, 1 FN -> P=0.0, R=0.0
    assert res["violation_identification_precision_rule_2"] == 0.0
    assert res["violation_identification_recall_rule_2"] == 0.0
    
    # Rule 3: 0 TP, 1 FP, 0 FN -> P=0.0, R=0.0
    assert res["violation_identification_precision_rule_3"] == 0.0
    assert res["violation_identification_recall_rule_3"] == 0.0

def test_global_micro_metrics():
    """Test that pooled (micro) precision/recall accurately aggregates all rules."""
    refs = [
        {"rule_1_violation": {"bounding_box": [[0,0,1,1]]}},
        {"rule_2_violation": {"bounding_box": [[0,0,1,1]]}}
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0,0,1000,1000]]}},
        {"rule_3_violation": {"bounding_box": [[0,0,1000,1000]]}}
    ]
    
    # Global overlaps:
    # Img 1: 1 TP (Rule 1)
    # Img 2: 1 FP (Rule 3), 1 FN (Rule 2)
    # Totals: 1 TP, 1 FP, 1 FN
    # Micro P = 1 / (1 + 1) = 0.5
    # Micro R = 1 / (1 + 1) = 0.5
    
    res = compute_violation_metrics(preds, refs)
    
    assert res["violation_identification_precision_micro"] == 0.5
    assert res["violation_identification_recall_micro"] == 0.5
    assert res["violation_identification_f1_micro"] == 0.5


def test_macro_vs_micro_divergence():
    """Test that true macro and micro averages diverge under class imbalance.

    Hand-constructed scenario:
      - 3 images violate Rule 1. Model correctly identifies all 3.
      - 1 image violates Rule 2. Model misses it entirely.
      - No Rule 3 or Rule 4 violations anywhere; model doesn't predict them either.

    Per-rule breakdown:
      Rule 1: TP=3, FP=0, FN=0 → P=1.0, R=1.0, F1=1.0
      Rule 2: TP=0, FP=0, FN=1 → P=0.0, R=0.0, F1=0.0
      Rule 3: TP=0, FP=0, FN=0 → P=0.0, R=0.0, F1=0.0  (no data)
      Rule 4: TP=0, FP=0, FN=0 → P=0.0, R=0.0, F1=0.0  (no data)

    Micro (pooled TP/FP/FN across all images and rules):
      Total: TP=3, FP=0, FN=1
      Micro P = 3/3 = 1.0
      Micro R = 3/4 = 0.75
      Micro F1 = 2*1.0*0.75/(1.0+0.75) = 6/7 ≈ 0.857

    Macro (mean of per-rule metrics, Rules 1–4):
      Macro P = (1.0 + 0.0 + 0.0 + 0.0) / 4 = 0.25
      Macro R = (1.0 + 0.0 + 0.0 + 0.0) / 4 = 0.25
      Macro F1 = (1.0 + 0.0 + 0.0 + 0.0) / 4 = 0.25

    This demonstrates how micro (0.857) hides poor Rule 2-4 performance
    while macro (0.25) exposes it — exactly the class-imbalance issue
    documented in the audit.
    """
    refs = [
        {"rule_1_violation": {"bounding_box": [[0,0,1,1]]}},
        {"rule_1_violation": {"bounding_box": [[0,0,1,1]]}},
        {"rule_1_violation": {"bounding_box": [[0,0,1,1]]}},
        {"rule_2_violation": {"bounding_box": [[0,0,1,1]]}},
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0,0,1000,1000]]}},
        {"rule_1_violation": {"bounding_box": [[0,0,1000,1000]]}},
        {"rule_1_violation": {"bounding_box": [[0,0,1000,1000]]}},
        {},  # Model misses Rule 2
    ]

    res = compute_violation_metrics(preds, refs)

    # Micro (pooled): TP=3, FP=0, FN=1
    assert res["violation_identification_precision_micro"] == 1.0
    assert res["violation_identification_recall_micro"] == 0.75
    assert abs(res["violation_identification_f1_micro"] - 6/7) < 1e-9

    # Macro (mean of per-rule, Rules 1–4 only, excludes rule_0):
    assert res["violation_identification_precision_macro"] == 0.25
    assert res["violation_identification_recall_macro"] == 0.25
    assert res["violation_identification_f1_macro"] == 0.25



def test_grounding_iou_separation():
    """Test that grounding IoU is computed correctly and separated by rule."""
    # Img 1: Rule 1 TP. Perfect IoU (1.0).
    # Img 2: Rule 1 TP. Zero IoU (0.0).
    # Img 3: Rule 4 TP. Partial IoU (0.5).
    
    refs = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}},
        {"rule_1_violation": {"bounding_box": [[0, 0, 0.5, 0.5]]}},
        {"rule_4_violation": {"bounding_box": [[0, 0, 1, 1]]}}
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 1000]]}}, # Perfect match
        {"rule_1_violation": {"bounding_box": [[500, 500, 1000, 1000]]}}, # No overlap
        {"rule_4_violation": {"bounding_box": [[0, 0, 1000, 500]]}} # Top half overlap -> IoU 0.5
    ]
    
    res = compute_violation_metrics(preds, refs)
    
    # Rule 1 IoU: Two instances (1.0 and 0.0) -> Average 0.5
    assert res["violation_grounding_iou_rule_1_tn0"] == 0.5
    
    # Rule 4 IoU: One instance (0.5) -> Average 0.5
    assert res["violation_grounding_iou_rule_4_tn0"] == 0.5
    
    # Rule 2 IoU: No instances -> 0.0
    assert res["violation_grounding_iou_rule_2_tn0"] == 0.0
    
    # Global IoU Macro (True Macro): Average of (0.5, 0.0, 0.0, 0.5) = 0.25
    assert res["violation_grounding_iou_macro_tn0"] == 0.25

def test_flat_box_handling():
    """Test that flat boxes are correctly parsed to compute IoU (the bug we fixed)."""
    # If the bug was not fixed, `clean_boxes` would throw out the flat arrays
    # resulting in an empty list for both GT and Pred.
    # An empty vs empty box comparison yields a default IoU of 1.0 (True Negative bounding box logic).
    # By using a partial overlap box, we can prove the fix works:
    # - If broken: Returns 1.0
    # - If fixed: Returns exactly 0.5
    refs = [
        {"rule_1_violation": {"bounding_box": [0, 0, 1, 1]}}
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [0, 0, 1000, 500]}}
    ]
    
    res = compute_violation_metrics(preds, refs)
    
    assert res["violation_grounding_iou_rule_1_tn0"] == 0.5

def test_violation_tn_three_way_split():
    """Test that TN cases produce diverging IoU metrics for the 3 conventions."""
    # We need a mix of cases for Rule 1 (all are True Positives for the *violation* itself):
    # 1. Normal overlap (IoU = 0.5)
    # 2. TN box (both empty)
    # 3. FN box (GT exists, Pred empty -> IoU = 0.0)
    
    refs = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}},      # Normal
        {"rule_1_violation": {"bounding_box": []}},                  # TN
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}},      # FN
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 500]]}}, # Normal -> 0.5
        {"rule_1_violation": {"bounding_box": []}},                  # TN
        {"rule_1_violation": {"bounding_box": []}},                  # FN -> 0.0
    ]
    
    res = compute_violation_metrics(preds, refs)
    
    # Math for the 3 variants:
    # We have 3 total items for Rule 1.
    # Item 1 (Normal): IoU = 0.5 (for all variants)
    # Item 2 (TN): tn0=0.0, tn1=1.0, excl=skipped
    # Item 3 (FN): IoU = 0.0 (for all variants)
    
    # tn0: (0.5 + 0.0 + 0.0) / 3 = 0.1666...
    assert abs(res["violation_grounding_iou_rule_1_tn0"] - (0.5 / 3)) < 1e-6
    
    # tn1: (0.5 + 1.0 + 0.0) / 3 = 0.5
    assert abs(res["violation_grounding_iou_rule_1_tn1"] - 0.5) < 1e-6
    
    # excl: skips TN. remaining: 0.5, 0.0 -> (0.5 + 0.0) / 2 = 0.25
    assert abs(res["violation_grounding_iou_rule_1_excl"] - 0.25) < 1e-6
    
    # TN count should be 1
    assert res["violation_grounding_tn_count_rule_1"] == 1

