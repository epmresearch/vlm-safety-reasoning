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
    if not predictions or not references or len(predictions) != len(references):
        return {}

    class_ious_total: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_exist: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    
    class_inter_total: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_total: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_inter_exist: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_exist: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    
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
            
            # Compute multi-box IoU (our updated function handles TN, FP, FN properly)
            iou, total_inter, total_union = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)
                
            # Total IoU includes all cases (including TN=1.0, FP=0.0, FN=0.0)
            class_ious_total[cls].append(iou)
            class_inter_total[cls] += total_inter
            class_union_total[cls] += total_union
            
            # Exist IoU only includes images where the ground truth object exists
            if gt_boxes_01:
                class_ious_exist[cls].append(iou)
                class_inter_exist[cls] += total_inter
                class_union_exist[cls] += total_union
            
    # Aggregate metrics
    metrics = {}
    
    total_inter_all_t = 0.0
    total_union_all_t = 0.0
    total_inter_all_e = 0.0
    total_union_all_e = 0.0
    
    all_ious_total = []
    all_ious_exist = []
    
    for cls in GROUNDING_CLASSES:
        # 1. Macro Total
        ious_t = class_ious_total[cls]
        metrics[f"grounding_iou_total_macro_{cls}"] = sum(ious_t) / len(ious_t) if ious_t else 0.0
        all_ious_total.extend(ious_t)
        
        # 2. Micro Total
        inter_t = class_inter_total[cls]
        union_t = class_union_total[cls]
        metrics[f"grounding_iou_total_micro_{cls}"] = inter_t / union_t if union_t > 0 else 0.0
        total_inter_all_t += inter_t
        total_union_all_t += union_t
        
        # 3. Macro Object Exist
        ious_e = class_ious_exist[cls]
        metrics[f"grounding_iou_exist_macro_{cls}"] = sum(ious_e) / len(ious_e) if ious_e else 0.0
        all_ious_exist.extend(ious_e)
        
        # 4. Micro Object Exist
        inter_e = class_inter_exist[cls]
        union_e = class_union_exist[cls]
        metrics[f"grounding_iou_exist_micro_{cls}"] = inter_e / union_e if union_e > 0 else 0.0
        total_inter_all_e += inter_e
        total_union_all_e += union_e
        
    metrics["grounding_iou_total_macro_mean"] = sum(all_ious_total) / len(all_ious_total) if all_ious_total else 0.0
    metrics["grounding_iou_total_micro_mean"] = total_inter_all_t / total_union_all_t if total_union_all_t > 0 else 0.0
    
    metrics["grounding_iou_exist_macro_mean"] = sum(all_ious_exist) / len(all_ious_exist) if all_ious_exist else 0.0
    metrics["grounding_iou_exist_micro_mean"] = total_inter_all_e / total_union_all_e if total_union_all_e > 0 else 0.0
    
    return metrics
