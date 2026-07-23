"""
Metrics for safety violation evaluation.
"""
from typing import Dict, List, Any, Set
import pandas as pd

from core.constants import RULES
from data.box_utils import compute_mask_union_iou, greedy_multibox_iou, scale_1000_to_01, clean_boxes, normalize_boxes
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
    
    # Mask Trackers
    rule_iou_tn0_mask = {r: [] for r in RULES}
    rule_inter_total_mask = {r: 0.0 for r in RULES}
    rule_union_total_mask = {r: 0.0 for r in RULES}

    # Greedy Trackers
    rule_iou_tn0_greedy = {r: [] for r in RULES}
    rule_inter_total_greedy = {r: 0.0 for r in RULES}
    rule_union_total_greedy = {r: 0.0 for r in RULES}

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
                
        common_rules = pred_rules & gt_rules
        for r in common_rules:
            pred_boxes_1000 = pred_by_rule[r].get("bounding_box", [])
            gt_boxes_01 = gt_by_rule[r].get("bounding_box", [])

            # Normalize first to handle flat lists
            pred_boxes_1000 = normalize_boxes(pred_boxes_1000)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)

            # Scale pred to [0, 1] FIRST
            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]

            # Then clean both using the exact same [0, 1] scale threshold
            pred_boxes_01 = clean_boxes(pred_boxes_01)
            gt_boxes_01 = clean_boxes(gt_boxes_01)

            # 1. Mask-Union IoU
            mask_result = compute_mask_union_iou(pred_boxes_01, gt_boxes_01)
            mask_iou = mask_result["iou"]

            # 2. Greedy IoU
            greedy_iou_val, greedy_inter, greedy_union = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)

            if not pred_boxes_01 and not gt_boxes_01:
                # True Negative
                rule_iou_tn0_mask[r].append(0.0)
                rule_iou_tn0_greedy[r].append(0.0)
                rule_tn_count[r] += 1
            else:
                rule_iou_tn0_mask[r].append(mask_iou)
                rule_iou_tn0_greedy[r].append(greedy_iou_val)

            rule_inter_total_mask[r] += mask_result["intersection"]
            rule_union_total_mask[r] += mask_result["union"]

            rule_inter_total_greedy[r] += greedy_inter
            rule_union_total_greedy[r] += greedy_union

    metrics = {}
    
    # Global (pooled) precision/recall/F1 — this is MICRO-averaging
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics["violation_identification_precision_micro"] = precision
    metrics["violation_identification_recall_micro"] = recall
    metrics["violation_identification_f1_micro"] = f1
    
    rule_precisions, rule_recalls, rule_f1s = [], [], []
    for r in ALL_RULES:
        tp, fp, fn = rule_counts[r]["tp"], rule_counts[r]["fp"], rule_counts[r]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        r_f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        
        metrics[f"violation_identification_precision_{r}"] = p
        metrics[f"violation_identification_recall_{r}"] = rec
        metrics[f"violation_identification_f1_{r}"] = r_f1
        
        if r in RULES:
            rule_precisions.append(p)
            rule_recalls.append(rec)
            rule_f1s.append(r_f1)
    
    metrics["violation_identification_precision_macro"] = (
        sum(rule_precisions) / len(rule_precisions) if rule_precisions else 0.0
    )
    metrics["violation_identification_recall_macro"] = (
        sum(rule_recalls) / len(rule_recalls) if rule_recalls else 0.0
    )
    metrics["violation_identification_f1_macro"] = (
        sum(rule_f1s) / len(rule_f1s) if rule_f1s else 0.0
    )
        
    # Grounding IoU per rule and global macro/micro
    tn0_macros_mask, tn0_macros_greedy = [], []
    total_inter_mask, total_union_mask = 0.0, 0.0
    total_inter_greedy, total_union_greedy = 0.0, 0.0
    
    for r in RULES:
        metrics[f"violation_grounding_tn_count_{r}"] = rule_tn_count[r]
        
        # Mask
        tn0_val_mask = sum(rule_iou_tn0_mask[r]) / len(rule_iou_tn0_mask[r]) if rule_iou_tn0_mask[r] else 0.0
        metrics[f"violation_grounding_mask_iou_{r}_tn0"] = tn0_val_mask
        tn0_macros_mask.append(tn0_val_mask)
        
        inter_r_mask = rule_inter_total_mask[r]
        union_r_mask = rule_union_total_mask[r]
        metrics[f"violation_grounding_mask_iou_{r}_micro"] = inter_r_mask / union_r_mask if union_r_mask > 0 else 0.0
        total_inter_mask += inter_r_mask
        total_union_mask += union_r_mask

        # Greedy
        tn0_val_greedy = sum(rule_iou_tn0_greedy[r]) / len(rule_iou_tn0_greedy[r]) if rule_iou_tn0_greedy[r] else 0.0
        metrics[f"violation_grounding_greedy_iou_{r}_tn0"] = tn0_val_greedy
        tn0_macros_greedy.append(tn0_val_greedy)
        
        inter_r_greedy = rule_inter_total_greedy[r]
        union_r_greedy = rule_union_total_greedy[r]
        metrics[f"violation_grounding_greedy_iou_{r}_micro"] = inter_r_greedy / union_r_greedy if union_r_greedy > 0 else 0.0
        total_inter_greedy += inter_r_greedy
        total_union_greedy += union_r_greedy
        
    metrics["violation_grounding_mask_iou_macro_tn0"] = sum(tn0_macros_mask) / len(tn0_macros_mask) if tn0_macros_mask else 0.0
    metrics["violation_grounding_mask_iou_micro_mean"] = total_inter_mask / total_union_mask if total_union_mask > 0 else 0.0

    metrics["violation_grounding_greedy_iou_macro_tn0"] = sum(tn0_macros_greedy) / len(tn0_macros_greedy) if tn0_macros_greedy else 0.0
    metrics["violation_grounding_greedy_iou_micro_mean"] = total_inter_greedy / total_union_greedy if total_union_greedy > 0 else 0.0
    
    return metrics