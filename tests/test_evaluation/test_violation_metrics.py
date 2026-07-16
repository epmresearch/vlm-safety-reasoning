import pytest
import math

from evaluation.metrics_violations import compute_violation_metrics

def test_empty_inputs():
    """Test that empty inputs return an empty dictionary without crashing."""
    assert compute_violation_metrics([], []) == {}
    assert compute_violation_metrics(None, None) == {}
    assert compute_violation_metrics([{"rule_1_violation": {}}], []) == {} # Length mismatch

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
    
    assert res["violation_rule_0_precision"] == 0.5
    assert res["violation_rule_0_recall"] == 0.5
    assert res["violation_rule_0_f1"] == 0.5

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
    assert res["violation_rule_1_precision"] == 1.0
    assert res["violation_rule_1_recall"] == 1.0
    
    # Rule 2: 0 TP, 0 FP, 1 FN -> P=0.0, R=0.0
    assert res["violation_rule_2_precision"] == 0.0
    assert res["violation_rule_2_recall"] == 0.0
    
    # Rule 3: 0 TP, 1 FP, 0 FN -> P=0.0, R=0.0
    assert res["violation_rule_3_precision"] == 0.0
    assert res["violation_rule_3_recall"] == 0.0

def test_global_macro_metrics():
    """Test that global precision/recall accurately aggregates all rules."""
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
    # Global P = 1 / (1 + 1) = 0.5
    # Global R = 1 / (1 + 1) = 0.5
    
    res = compute_violation_metrics(preds, refs)
    
    assert res["violation_macro_precision"] == 0.5
    assert res["violation_macro_recall"] == 0.5
    assert res["violation_macro_f1"] == 0.5

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
    assert res["violation_iou_rule_1"] == 0.5
    
    # Rule 4 IoU: One instance (0.5) -> Average 0.5
    assert res["violation_iou_rule_4"] == 0.5
    
    # Rule 2 IoU: No instances -> 0.0
    assert res["violation_iou_rule_2"] == 0.0
    
    # Global IoU: Three instances (1.0, 0.0, 0.5) -> Average 0.5
    assert res["violation_grounding_iou"] == 0.5

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
    
    assert res["violation_iou_rule_1"] == 0.5
