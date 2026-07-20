"""
Metrics for bounding box grounding evaluation.
"""
from typing import Dict, List, Any
import numpy as np

from data.box_utils import greedy_multibox_iou, scale_1000_to_01, normalize_boxes, clean_boxes
from core.constants import GROUNDING_CLASSES
from core.logging import get_logger

logger = get_logger(__name__)

def compute_grounding_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes class-wise grounding IoU for object detection tasks.
    predictions: list of parsed 'detected_objects' dictionaries.
                 Bounding boxes are expected in [0, 1000] scale.
    references: list of ground truth 'detected_objects' dictionaries.
                Bounding boxes are expected in [0, 1] scale.
    """
    if not predictions or not references:
        raise ValueError(
            "compute_grounding_metrics requires non-empty predictions and references lists."
        )
    if len(predictions) != len(references):
        raise ValueError(
            f"compute_grounding_metrics: length mismatch — "
            f"{len(predictions)} predictions vs {len(references)} references."
        )

    class_ious_all_tn0: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_all_tn1: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_all_excl: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    
    class_ious_exist_tn0: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_exist_tn1: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_exist_excl: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    
    class_inter_total: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_total: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_inter_exist: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_exist: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    
    class_tn: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_fp: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_fn: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_exist_n: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    
    for pred, gt in zip(predictions, references):
        pred_objs = pred or {}
        gt_objs = gt or {}
        
        for cls in GROUNDING_CLASSES:
            pred_boxes_1000 = pred_objs.get(cls, [])
            gt_boxes_01 = gt_objs.get(cls, [])
            
            # Normalize boxes first (handles flat single boxes)
            pred_boxes_1000 = normalize_boxes(pred_boxes_1000)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)
            
            # Then clean (removes invalid boxes)
            pred_boxes_1000 = clean_boxes(pred_boxes_1000)
            gt_boxes_01 = clean_boxes(gt_boxes_01)
            
            # Convert prediction boxes from [0, 1000] to [0, 1] scale for comparison
            pred_boxes_01 = []
            for box in pred_boxes_1000:
                scaled_box = scale_1000_to_01(box)
                pred_boxes_01.append(scaled_box)
            
            # Compute multi-box IoU
            # Compute multi-box IoU (greedy matching returns 0.0 IoU for TN cases by default)
            iou, total_inter, total_union = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)
                
            # Record TN/FP/FN for transparency and apply variants
            if not gt_boxes_01 and not pred_boxes_01:
                class_tn[cls] += 1
                class_ious_all_tn0[cls].append(0.0)
                class_ious_all_tn1[cls].append(1.0)
                # excl ignores it
            elif not gt_boxes_01 and pred_boxes_01:
                class_fp[cls] += 1
                class_ious_all_tn0[cls].append(iou)
                class_ious_all_tn1[cls].append(iou)
                class_ious_all_excl[cls].append(iou)
            elif gt_boxes_01 and not pred_boxes_01:
                class_fn[cls] += 1
                class_ious_all_tn0[cls].append(iou)
                class_ious_all_tn1[cls].append(iou)
                class_ious_all_excl[cls].append(iou)
            else:
                class_ious_all_tn0[cls].append(iou)
                class_ious_all_tn1[cls].append(iou)
                class_ious_all_excl[cls].append(iou)
                
            class_inter_total[cls] += total_inter
            class_union_total[cls] += total_union
            
            # Exist IoU only includes images where the ground truth object exists
            if gt_boxes_01:
                class_exist_n[cls] += 1
                class_ious_exist_tn0[cls].append(iou)
                class_ious_exist_tn1[cls].append(iou)
                class_ious_exist_excl[cls].append(iou)
                class_inter_exist[cls] += total_inter
                class_union_exist[cls] += total_union
            
    # Aggregate metrics
    metrics = {}
    
    total_inter_all_t = 0.0
    total_union_all_t = 0.0
    total_inter_all_e = 0.0
    total_union_all_e = 0.0
    
    all_ious_total_tn0, all_ious_total_tn1, all_ious_total_excl = [], [], []
    all_ious_exist_tn0, all_ious_exist_tn1, all_ious_exist_excl = [], [], []
    
    for cls in GROUNDING_CLASSES:
        # 1. Macro Total variants
        metrics[f"grounding_iou_all_macro_{cls}_tn0"] = sum(class_ious_all_tn0[cls]) / len(class_ious_all_tn0[cls]) if class_ious_all_tn0[cls] else 0.0
        metrics[f"grounding_iou_all_macro_{cls}_tn1"] = sum(class_ious_all_tn1[cls]) / len(class_ious_all_tn1[cls]) if class_ious_all_tn1[cls] else 0.0
        metrics[f"grounding_iou_all_macro_{cls}_excl"] = sum(class_ious_all_excl[cls]) / len(class_ious_all_excl[cls]) if class_ious_all_excl[cls] else 0.0
        
        all_ious_total_tn0.extend(class_ious_all_tn0[cls])
        all_ious_total_tn1.extend(class_ious_all_tn1[cls])
        all_ious_total_excl.extend(class_ious_all_excl[cls])
        
        # 2. Micro Total
        inter_t = class_inter_total[cls]
        union_t = class_union_total[cls]
        metrics[f"grounding_iou_all_micro_{cls}"] = inter_t / union_t if union_t > 0 else 0.0
        total_inter_all_t += inter_t
        total_union_all_t += union_t
        
        # 3. Macro Object Exist variants
        metrics[f"grounding_iou_existing_macro_{cls}_tn0"] = sum(class_ious_exist_tn0[cls]) / len(class_ious_exist_tn0[cls]) if class_ious_exist_tn0[cls] else 0.0
        metrics[f"grounding_iou_existing_macro_{cls}_tn1"] = sum(class_ious_exist_tn1[cls]) / len(class_ious_exist_tn1[cls]) if class_ious_exist_tn1[cls] else 0.0
        metrics[f"grounding_iou_existing_macro_{cls}_excl"] = sum(class_ious_exist_excl[cls]) / len(class_ious_exist_excl[cls]) if class_ious_exist_excl[cls] else 0.0
        
        all_ious_exist_tn0.extend(class_ious_exist_tn0[cls])
        all_ious_exist_tn1.extend(class_ious_exist_tn1[cls])
        all_ious_exist_excl.extend(class_ious_exist_excl[cls])
        
        # 4. Micro Object Exist
        inter_e = class_inter_exist[cls]
        union_e = class_union_exist[cls]
        metrics[f"grounding_iou_existing_micro_{cls}"] = inter_e / union_e if union_e > 0 else 0.0
        total_inter_all_e += inter_e
        total_union_all_e += union_e
        
        # Transparency counters
        metrics[f"grounding_true_negatives_count_{cls}"] = class_tn[cls]
        metrics[f"grounding_false_positives_count_{cls}"] = class_fp[cls]
        metrics[f"grounding_false_negatives_count_{cls}"] = class_fn[cls]
        metrics[f"grounding_existing_ground_truth_count_{cls}"] = class_exist_n[cls]
        
    # N1 Fix + N3 variants
    metrics["grounding_iou_all_pooled_mean_tn0"] = sum(all_ious_total_tn0) / len(all_ious_total_tn0) if all_ious_total_tn0 else 0.0
    metrics["grounding_iou_all_pooled_mean_tn1"] = sum(all_ious_total_tn1) / len(all_ious_total_tn1) if all_ious_total_tn1 else 0.0
    metrics["grounding_iou_all_pooled_mean_excl"] = sum(all_ious_total_excl) / len(all_ious_total_excl) if all_ious_total_excl else 0.0
    
    metrics["grounding_iou_all_macro_mean_tn0"] = sum([metrics[f"grounding_iou_all_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_all_macro_mean_tn1"] = sum([metrics[f"grounding_iou_all_macro_{cls}_tn1"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_all_macro_mean_excl"] = sum([metrics[f"grounding_iou_all_macro_{cls}_excl"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    
    metrics["grounding_iou_all_micro_mean"] = total_inter_all_t / total_union_all_t if total_union_all_t > 0 else 0.0
    
    metrics["grounding_iou_existing_pooled_mean_tn0"] = sum(all_ious_exist_tn0) / len(all_ious_exist_tn0) if all_ious_exist_tn0 else 0.0
    metrics["grounding_iou_existing_pooled_mean_tn1"] = sum(all_ious_exist_tn1) / len(all_ious_exist_tn1) if all_ious_exist_tn1 else 0.0
    metrics["grounding_iou_existing_pooled_mean_excl"] = sum(all_ious_exist_excl) / len(all_ious_exist_excl) if all_ious_exist_excl else 0.0
    
    metrics["grounding_iou_existing_macro_mean_tn0"] = sum([metrics[f"grounding_iou_existing_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_existing_macro_mean_tn1"] = sum([metrics[f"grounding_iou_existing_macro_{cls}_tn1"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_existing_macro_mean_excl"] = sum([metrics[f"grounding_iou_existing_macro_{cls}_excl"] for cls in GROUNDING_CLASSES]) / len(GROUNDING_CLASSES)
    
    metrics["grounding_iou_existing_micro_mean"] = total_inter_all_e / total_union_all_e if total_union_all_e > 0 else 0.0
    
    return metrics
