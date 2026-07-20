"""
Metrics for safety violation evaluation.
"""
from typing import Dict, List, Any, Set
import pandas as pd

from core.constants import RULES
from data.box_utils import greedy_multibox_iou, scale_1000_to_01, clean_boxes, normalize_boxes
from core.logging import get_logger

logger = get_logger(__name__)

def compute_violation_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes rule identification (F1, Precision, Recall) and grounding IoU for safety violations.
    predictions: List of flat output dictionaries.
    references: List of flat output dictionaries.
    """
    if not predictions or not references:
        raise ValueError(
            "compute_violation_metrics requires non-empty predictions and references lists."
        )
    if len(predictions) != len(references):
        raise ValueError(
            f"compute_violation_metrics: length mismatch — "
            f"{len(predictions)} predictions vs {len(references)} references."
        )

    # We add rule_0 (no violation) for explicit tracking
    ALL_RULES = RULES + ["rule_0"]
    global_tp, global_fp, global_fn = 0, 0, 0
    rule_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ALL_RULES}
    
    # Trackers for the three TN variants and TN counts
    rule_iou_tn0 = {r: [] for r in RULES}
    rule_iou_tn1 = {r: [] for r in RULES}
    rule_iou_excl = {r: [] for r in RULES}
    rule_tn_count = {r: 0 for r in RULES}
    
    for pred_dict, gt_dict in zip(predictions, references):
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}
        
        pred_rules = set()
        pred_by_rule = {}
        gt_rules = set()
        gt_by_rule = {}
        
        for r in RULES:
            p_v = pred_dict.get(f"{r}_violation")
            if p_v:
                pred_rules.add(r)
                pred_by_rule[r] = p_v
                
            g_v = gt_dict.get(f"{r}_violation")
            if g_v:
                gt_rules.add(r)
                gt_by_rule[r] = g_v
        
        # Rule 0 tracking
        if not gt_rules and not pred_rules:
            rule_counts["rule_0"]["tp"] += 1
        elif not gt_rules and pred_rules:
            rule_counts["rule_0"]["fn"] += 1
        elif gt_rules and not pred_rules:
            rule_counts["rule_0"]["fp"] += 1
            
        # Global counts
        tp = len(pred_rules & gt_rules)
        fp = len(pred_rules - gt_rules)
        fn = len(gt_rules - pred_rules)
        
        global_tp += tp
        global_fp += fp
        global_fn += fn
        
        # Per-rule counts
        for r in RULES:
            in_pred, in_gt = r in pred_rules, r in gt_rules
            if in_pred and in_gt:
                rule_counts[r]["tp"] += 1
            elif in_pred and not in_gt:
                rule_counts[r]["fp"] += 1
            elif not in_pred and in_gt:
                rule_counts[r]["fn"] += 1
                
        # Grounding IoU for correctly identified rules
        common_rules = pred_rules & gt_rules
        for r in common_rules:
            pred_boxes_1000 = pred_by_rule[r].get("bounding_box", [])
            gt_boxes_01 = gt_by_rule[r].get("bounding_box", [])
            
            # Normalize first to handle flat lists
            pred_boxes_1000 = normalize_boxes(pred_boxes_1000)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)
            
            # Then clean
            pred_boxes_1000 = clean_boxes(pred_boxes_1000)
            gt_boxes_01 = clean_boxes(gt_boxes_01)
            
            # Scale pred to [0, 1]
            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]
            
            if not pred_boxes_01 and not gt_boxes_01:
                # True Negative
                rule_iou_tn0[r].append(0.0)
                rule_iou_tn1[r].append(1.0)
                # _excl omits it entirely
                rule_tn_count[r] += 1
            elif not pred_boxes_01 or not gt_boxes_01:
                # False Positive or False Negative
                rule_iou_tn0[r].append(0.0)
                rule_iou_tn1[r].append(0.0)
                rule_iou_excl[r].append(0.0)
            else:
                iou, _, _ = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)
                rule_iou_tn0[r].append(iou)
                rule_iou_tn1[r].append(iou)
                rule_iou_excl[r].append(iou)

    metrics = {}
    
    # Global (pooled) precision/recall/F1 — this is MICRO-averaging:
    # pool all TP/FP/FN across images and rules, then compute one ratio.
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics["violation_identification_precision_micro"] = precision
    metrics["violation_identification_recall_micro"] = recall
    metrics["violation_identification_f1_micro"] = f1
    
    # Per-rule metrics (computed for ALL_RULES including rule_0)
    # Also collect per-violation-rule values for macro-averaging below.
    rule_precisions, rule_recalls, rule_f1s = [], [], []
    for r in ALL_RULES:
        tp, fp, fn = rule_counts[r]["tp"], rule_counts[r]["fp"], rule_counts[r]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        r_f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        
        metrics[f"violation_identification_precision_{r}"] = p
        metrics[f"violation_identification_recall_{r}"] = rec
        metrics[f"violation_identification_f1_{r}"] = r_f1
        
        # Collect per-rule values for macro, excluding rule_0.
        # rule_0 ("no violation") is semantically different from the four
        # safety-rule classes and the paper (Table 7) treats it separately.
        # Macro-average is over Rules 1–4 only.
        if r in RULES:
            rule_precisions.append(p)
            rule_recalls.append(rec)
            rule_f1s.append(r_f1)
    
    # True MACRO-average: unweighted mean of per-rule precision/recall/F1
    # over Rules 1–4 only (excludes rule_0). This gives each violation
    # rule equal weight regardless of occurrence frequency.
    metrics["violation_identification_precision_macro"] = (
        sum(rule_precisions) / len(rule_precisions) if rule_precisions else 0.0
    )
    metrics["violation_identification_recall_macro"] = (
        sum(rule_recalls) / len(rule_recalls) if rule_recalls else 0.0
    )
    metrics["violation_identification_f1_macro"] = (
        sum(rule_f1s) / len(rule_f1s) if rule_f1s else 0.0
    )
        
    # Grounding IoU per rule and global macro
    tn0_macros, tn1_macros, excl_macros = [], [], []
    for r in RULES:
        metrics[f"violation_grounding_tn_count_{r}"] = rule_tn_count[r]
        
        # _tn0
        tn0_val = sum(rule_iou_tn0[r]) / len(rule_iou_tn0[r]) if rule_iou_tn0[r] else 0.0
        metrics[f"violation_grounding_iou_{r}_tn0"] = tn0_val
        tn0_macros.append(tn0_val)
        
        # _tn1
        tn1_val = sum(rule_iou_tn1[r]) / len(rule_iou_tn1[r]) if rule_iou_tn1[r] else 0.0
        metrics[f"violation_grounding_iou_{r}_tn1"] = tn1_val
        tn1_macros.append(tn1_val)
        
        # _excl
        excl_val = sum(rule_iou_excl[r]) / len(rule_iou_excl[r]) if rule_iou_excl[r] else 0.0
        metrics[f"violation_grounding_iou_{r}_excl"] = excl_val
        excl_macros.append(excl_val)
        
    metrics["violation_grounding_iou_macro_tn0"] = sum(tn0_macros) / len(tn0_macros) if tn0_macros else 0.0
    metrics["violation_grounding_iou_macro_tn1"] = sum(tn1_macros) / len(tn1_macros) if tn1_macros else 0.0
    metrics["violation_grounding_iou_macro_excl"] = sum(excl_macros) / len(excl_macros) if excl_macros else 0.0
    
    return metrics

