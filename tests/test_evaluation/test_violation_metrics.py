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
    assert res["violation_grounding_mask_iou_rule_1_tn0"] == 0.5
    
    # Rule 4 IoU: One instance (0.5) -> Average 0.5
    assert res["violation_grounding_mask_iou_rule_4_tn0"] == 0.5
    
    # Rule 2 IoU: No instances -> the per-rule key still reports 0.0, because
    # downstream CSV/chart code expects a value for every rule.
    assert res["violation_grounding_mask_iou_rule_2_tn0"] == 0.0
    assert res["violation_grounding_scored_count_rule_2"] == 0

    # ...but the MACRO averages only the rules that were actually measured.
    # Rules 2 and 3 have no true positives, so there was no box to score:
    #   correct : (0.5 + 0.5) / 2 = 0.5   over n_rules = 2
    #   old bug : (0.5 + 0.0 + 0.0 + 0.5) / 4 = 0.25
    assert res["violation_grounding_mask_iou_macro_tn0"] == 0.5
    assert res["violation_grounding_mask_iou_macro_tn0_n_rules"] == 2
    assert res["violation_grounding_greedy_iou_macro_tn0_n_rules"] == 2
    assert res["violation_grounding_scored_count_rule_1"] == 2
    assert res["violation_grounding_scored_count_rule_4"] == 1

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
    
    assert res["violation_grounding_mask_iou_rule_1_tn0"] == 0.5

def test_violation_tn_three_way_split():
    """A rule flagged on both sides but localized on neither scores 0.0, not 1.0.

    Two semantics to keep in mind:

    1. A rule enters the grounding computation only when _is_violation_present() is True
       for BOTH pred and GT (i.e. it is in common_rules). Since **null is the only safe
       signal**, item 3 below uses an explicit null to be a genuine false negative — a
       keyed-but-empty object like {"bounding_box": []} would count as a detection.
    2. The old `_tn1` keys scored this case 1.0 (full grounding credit for not
       localizing) and have been removed; only the `_tn0` convention remains.
    """
    refs = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}},                 # Normal
        {"rule_1_violation": {"bounding_box": [], "reason": "unsafe area"}},    # TN (reason-only)
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}},                 # FN
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 500]]}},            # -> IoU 0.5
        {"rule_1_violation": {"bounding_box": [], "reason": "unsafe area"}},    # TN
        {"rule_1_violation": None},                                             # Safe -> FN
    ]

    res = compute_violation_metrics(preds, refs)

    # Only items 1 and 2 reach the grounding computation: item 3's prediction is null,
    # so rule_1 is not in common_rules for it.
    # Item 1: IoU = 0.5.  Item 2 (flagged by both, localized by neither): 0.0.
    # -> (0.5 + 0.0) / 2 = 0.25
    assert abs(res["violation_grounding_mask_iou_rule_1_tn0"] - 0.25) < 1e-6

    # TN count should be 1
    assert res["violation_grounding_tn_count_rule_1"] == 1

    # The tn1 convention (which scored item 2 as 1.0) is gone.
    assert "violation_grounding_mask_iou_rule_1_tn1" not in res
    assert "violation_grounding_mask_iou_macro_tn1" not in res
    assert not any(k.endswith("_tn1") for k in res if k.startswith("violation_grounding"))

    # Item 3 remains an identification-level false negative (2 TP, 1 FN -> recall 2/3).
    assert abs(res["violation_identification_recall_rule_1"] - (2 / 3)) < 1e-6


def test_contentless_violation_object_counts_as_a_detection():
    """A keyed-but-empty violation object is a detection, not a safe call.

    This is the shape structural_repair.py:959 produces from a bare `true`. Scoring it
    as safe would invert the model's own answer — and on a safe image would credit an
    unsubstantiated alarm as a rule_0 true positive.
    """
    # GT safe, model emits a contentless violation object -> false alarm, not rule_0 TP.
    res = compute_violation_metrics(
        [{"rule_1_violation": {"bounding_box": [], "reason": ""}}],
        [{"rule_1_violation": None}],
    )
    assert res["violation_identification_recall_rule_0"] == 0.0   # the false alarm is counted
    assert res["violation_identification_precision_rule_1"] == 0.0

    # GT has the violation, model asserts it without evidence -> identification TP,
    # but zero grounding credit because it never said where.
    res = compute_violation_metrics(
        [{"rule_1_violation": {"bounding_box": [], "reason": ""}}],
        [{"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]]}}],
    )
    assert res["violation_identification_recall_rule_1"] == 1.0
    assert res["violation_grounding_mask_iou_rule_1_tn0"] == 0.0

    # An explicit null on a safe image is still the correct safe call.
    res = compute_violation_metrics(
        [{"rule_1_violation": None}],
        [{"rule_1_violation": None}],
    )
    assert res["violation_identification_recall_rule_0"] == 1.0



# ---------------------------------------------------------------------------
# Image-level base rates and the packed per-image outcome vector
#
# violation_pred_positive_rate is the fraction of images the model flagged at
# all. Against violation_gt_positive_rate it is the single most diagnostic
# number for this task and it was absent from every metric file before: a
# zero-shot baseline flagging 64% of images against a 13.7% base rate has a
# recall that looks excellent and means nothing.
# ---------------------------------------------------------------------------

def test_image_level_positive_rates_and_counts():
    refs = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}},   # violation
        {"rule_2_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}},   # violation
        {"rule_1_violation": None},                                              # safe
        {"rule_1_violation": None},                                              # safe
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 1000]], "reason": "p"}},  # TP
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 1000]], "reason": "p"}},  # wrong rule
        {"rule_1_violation": {"bounding_box": [[0, 0, 1000, 1000]], "reason": "p"}},  # false alarm
        {"rule_1_violation": None},                                                   # correct safe
    ]
    res = compute_violation_metrics(preds, refs)

    assert res["violation_gt_positive_image_count"] == 2
    assert res["violation_pred_positive_image_count"] == 3
    assert res["violation_gt_positive_rate"] == 0.5
    assert res["violation_pred_positive_rate"] == 0.75

    # Absolute counts behind each per-rule ratio.
    assert res["violation_gt_count_rule_1"] == 1
    assert res["violation_pred_count_rule_1"] == 3
    assert res["violation_tp_count_rule_1"] == 1
    assert res["violation_gt_count_rule_2"] == 1
    assert res["violation_pred_count_rule_2"] == 0
    assert res["violation_tp_count_rule_2"] == 0


def test_per_image_outcome_vector_round_trips_to_the_same_confusion_matrix():
    """The packed vector must be a lossless record of the identification result.

    compare_all.py runs its paired bootstrap off this string alone, so if it
    ever disagreed with the aggregate metrics the confidence intervals would be
    describing a different model than the table above them.
    """
    import base64
    from core.constants import RULES

    refs = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"},
         "rule_3_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}},
        {"rule_2_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}},
        {"rule_1_violation": None},
        {"rule_4_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}},
    ]
    preds = [
        {"rule_1_violation": {"bounding_box": [[0, 0, 500, 500]], "reason": "p"}},
        {"rule_2_violation": {"bounding_box": [[0, 0, 500, 500]], "reason": "p"},
         "rule_4_violation": {"bounding_box": [[0, 0, 500, 500]], "reason": "p"}},
        {"rule_3_violation": {"bounding_box": [[0, 0, 500, 500]], "reason": "p"}},
        None,
    ]
    res = compute_violation_metrics(preds, refs)
    raw = base64.b64decode(res["violation_per_image_outcomes_b64"])
    assert len(raw) == len(preds)

    tp = fp = fn = 0
    pred_pos = 0
    for byte in raw:
        pred_mask, gt_mask = byte >> 4, byte & 0x0F
        if pred_mask:
            pred_pos += 1
        for i in range(len(RULES)):
            p, g = (pred_mask >> i) & 1, (gt_mask >> i) & 1
            if p and g:
                tp += 1
            elif p:
                fp += 1
            elif g:
                fn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    assert abs(prec - res["violation_identification_precision_micro"]) < 1e-12
    assert abs(rec - res["violation_identification_recall_micro"]) < 1e-12
    assert abs(f1 - res["violation_identification_f1_micro"]) < 1e-12
    assert pred_pos == res["violation_pred_positive_image_count"]


def test_identification_macro_still_counts_a_real_zero():
    """The macro fix applies to TP-CONDITIONED families only.

    An identification metric has a well-defined denominator even at zero true
    positives -- rule_3 has ground-truth positives whether or not the model
    finds any -- so an F1 of 0.0 there is a real, earned zero and must keep its
    full weight. Only reasoning and grounding, which have no denominator without
    a true positive, skip unmeasured rules.
    """
    refs = [{"rule_1_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"},
             "rule_3_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "r"}}]
    preds = [{"rule_1_violation": {"bounding_box": [[0, 0, 1000, 1000]], "reason": "p"}}]
    res = compute_violation_metrics(preds, refs)

    assert res["violation_identification_f1_rule_1"] == 1.0
    assert res["violation_identification_f1_rule_3"] == 0.0   # missed it entirely
    # macro divides by 4, INCLUDING the earned zeros -- unchanged behaviour.
    assert res["violation_identification_f1_macro"] == 1.0 / 4
    # ...while grounding, which could only be measured for rule_1, divides by 1.
    assert res["violation_grounding_mask_iou_macro_tn0_n_rules"] == 1
