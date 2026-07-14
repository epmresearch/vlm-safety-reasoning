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

    class_ious: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    
    for pred, gt in zip(predictions, references):
        pred_objs = pred or {}
        gt_objs = gt or {}
        
        for cls in GROUNDING_CLASSES:
            pred_boxes_1000 = pred_objs.get(cls, [])
            gt_boxes_01 = gt_objs.get(cls, [])
            
            # Clean and normalize
            pred_boxes_1000 = clean_boxes(pred_boxes_1000)
            gt_boxes_01 = clean_boxes(gt_boxes_01)
            
            # Convert prediction boxes from [0, 1000] to [0, 1] scale for comparison
            pred_boxes_01 = []
            for box in pred_boxes_1000:
                scaled_box = scale_1000_to_01(box)
                pred_boxes_01.append(scaled_box)
            
            pred_boxes_01 = normalize_boxes(pred_boxes_01)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)
            
            # Compute multi-box IoU
            if not pred_boxes_01 and not gt_boxes_01:
                # Both empty: perfect match
                iou = 1.0
            elif not pred_boxes_01 or not gt_boxes_01:
                # One empty, the other not: zero match
                iou = 0.0
            else:
                iou = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)
                
            class_ious[cls].append(iou)
            
    # Aggregate metrics
    metrics = {}
    for cls in GROUNDING_CLASSES:
        ious = class_ious[cls]
        metrics[f"grounding_iou_{cls}"] = sum(ious) / len(ious) if ious else 0.0
        
    all_ious = [iou for ious in class_ious.values() for iou in ious]
    metrics["grounding_iou_mean"] = sum(all_ious) / len(all_ious) if all_ious else 0.0
    
    return metrics
