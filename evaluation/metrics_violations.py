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
    if not predictions or not references or len(predictions) != len(references):
        return {}

    global_tp = 0
    global_fp = 0
    global_fn = 0
    
    rule_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in RULES}
    
    iou_scores = []
    
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
            
            pred_boxes_1000 = clean_boxes(pred_boxes_1000)
            gt_boxes_01 = clean_boxes(gt_boxes_01)
            
            # Scale pred to [0, 1]
            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]
            
            pred_boxes_01 = normalize_boxes(pred_boxes_01)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)
            
            if not pred_boxes_01 and not gt_boxes_01:
                iou = 1.0
            elif not pred_boxes_01 or not gt_boxes_01:
                iou = 0.0
            else:
                iou = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)
                
            iou_scores.append(iou)

    metrics = {}
    
    # Global F1
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics["violation_macro_precision"] = precision
    metrics["violation_macro_recall"] = recall
    metrics["violation_macro_f1"] = f1
    
    # Per-rule metrics
    for r in RULES:
        tp, fp, fn = rule_counts[r]["tp"], rule_counts[r]["fp"], rule_counts[r]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        r_f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        
        metrics[f"violation_{r}_precision"] = p
        metrics[f"violation_{r}_recall"] = rec
        metrics[f"violation_{r}_f1"] = r_f1
        
    # Grounding IoU
    metrics["violation_grounding_iou"] = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    
    return metrics
