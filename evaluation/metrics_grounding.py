"""
Metrics for bounding box grounding evaluation.

Reports both Semantic Coverage (Mask-Union IoU) and Instance Precision (Greedy IoU).
"""
from typing import Dict, List, Any
import numpy as np

from data.box_utils import (
    normalize_boxes, clean_boxes, scale_1000_to_01, compute_mask_union_iou, greedy_multibox_iou
)
from core.constants import GROUNDING_CLASSES
from core.logging import get_logger

logger = get_logger(__name__)


def compute_grounding_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    if not predictions or not references:
        raise ValueError("compute_grounding_metrics requires non-empty predictions and references lists.")
    if len(predictions) != len(references):
        raise ValueError(f"compute_grounding_metrics: length mismatch")

    # MASK TRACKERS
    class_ious_all_tn0_mask: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_exist_tn0_mask: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_inter_total_mask: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_total_mask: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}

    # GREEDY TRACKERS
    class_ious_all_tn0_greedy: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_exist_tn0_greedy: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_inter_total_greedy: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}
    class_union_total_greedy: Dict[str, float] = {cls: 0.0 for cls in GROUNDING_CLASSES}

    class_tn: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_fp: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_fn: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}
    class_exist_n: Dict[str, int] = {cls: 0 for cls in GROUNDING_CLASSES}

    for pred, gt in zip(predictions, references):
        pred_objs = pred or {}
        gt_objs = gt or {}

        for cls in GROUNDING_CLASSES:
            pred_boxes_1000 = normalize_boxes(pred_objs.get(cls, []))
            gt_boxes_01 = normalize_boxes(gt_objs.get(cls, []))

            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]

            pred_boxes_01 = clean_boxes(pred_boxes_01)
            gt_boxes_01 = clean_boxes(gt_boxes_01)

            # 1. Mask-Union IoU
            mask_result = compute_mask_union_iou(pred_boxes_01, gt_boxes_01)
            mask_iou = mask_result["iou"]

            # 2. Greedy IoU
            greedy_iou_val, greedy_inter, greedy_union = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)

            is_tn = not gt_boxes_01 and not pred_boxes_01
            is_fp = not gt_boxes_01 and bool(pred_boxes_01)
            is_fn = bool(gt_boxes_01) and not pred_boxes_01

            if is_tn:
                class_tn[cls] += 1
                class_ious_all_tn0_mask[cls].append(0.0)
                class_ious_all_tn0_greedy[cls].append(0.0)
            else:
                if is_fp:
                    class_fp[cls] += 1
                elif is_fn:
                    class_fn[cls] += 1
                class_ious_all_tn0_mask[cls].append(mask_iou)
                class_ious_all_tn0_greedy[cls].append(greedy_iou_val)

            class_inter_total_mask[cls] += mask_result["intersection"]
            class_union_total_mask[cls] += mask_result["union"]

            class_inter_total_greedy[cls] += greedy_inter
            class_union_total_greedy[cls] += greedy_union

            if gt_boxes_01:
                class_exist_n[cls] += 1
                class_ious_exist_tn0_mask[cls].append(mask_iou)
                class_ious_exist_tn0_greedy[cls].append(greedy_iou_val)

    metrics = {}

    total_inter_all_t_mask, total_union_all_t_mask = 0.0, 0.0
    total_inter_all_t_greedy, total_union_all_t_greedy = 0.0, 0.0

    all_ious_total_tn0_mask, all_ious_total_tn0_greedy = [], []

    for cls in GROUNDING_CLASSES:
        # Mask Metrics
        metrics[f"grounding_mask_iou_all_macro_{cls}_tn0"] = (
            sum(class_ious_all_tn0_mask[cls]) / len(class_ious_all_tn0_mask[cls]) if class_ious_all_tn0_mask[cls] else 0.0
        )
        metrics[f"grounding_mask_iou_all_micro_{cls}"] = class_inter_total_mask[cls] / class_union_total_mask[cls] if class_union_total_mask[cls] > 0 else 0.0

        all_ious_total_tn0_mask.extend(class_ious_all_tn0_mask[cls])
        total_inter_all_t_mask += class_inter_total_mask[cls]
        total_union_all_t_mask += class_union_total_mask[cls]
        
        metrics[f"grounding_mask_iou_exist_macro_{cls}"] = (
            sum(class_ious_exist_tn0_mask[cls]) / len(class_ious_exist_tn0_mask[cls]) if class_ious_exist_tn0_mask[cls] else 0.0
        )

        # Greedy Metrics
        metrics[f"grounding_greedy_iou_all_macro_{cls}_tn0"] = (
            sum(class_ious_all_tn0_greedy[cls]) / len(class_ious_all_tn0_greedy[cls]) if class_ious_all_tn0_greedy[cls] else 0.0
        )
        metrics[f"grounding_greedy_iou_all_micro_{cls}"] = class_inter_total_greedy[cls] / class_union_total_greedy[cls] if class_union_total_greedy[cls] > 0 else 0.0

        all_ious_total_tn0_greedy.extend(class_ious_all_tn0_greedy[cls])
        total_inter_all_t_greedy += class_inter_total_greedy[cls]
        total_union_all_t_greedy += class_union_total_greedy[cls]
        
        metrics[f"grounding_greedy_iou_exist_macro_{cls}"] = (
            sum(class_ious_exist_tn0_greedy[cls]) / len(class_ious_exist_tn0_greedy[cls]) if class_ious_exist_tn0_greedy[cls] else 0.0
        )

        # Counters
        metrics[f"grounding_true_negatives_count_{cls}"] = class_tn[cls]
        metrics[f"grounding_false_positives_count_{cls}"] = class_fp[cls]
        metrics[f"grounding_false_negatives_count_{cls}"] = class_fn[cls]
        metrics[f"grounding_existing_ground_truth_count_{cls}"] = class_exist_n[cls]

    # Global Aggregates
    metrics["grounding_mask_iou_all_pooled_mean_tn0"] = sum(all_ious_total_tn0_mask) / len(all_ious_total_tn0_mask) if all_ious_total_tn0_mask else 0.0
    metrics["grounding_mask_iou_all_macro_mean_tn0"] = sum(metrics[f"grounding_mask_iou_all_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)
    metrics["grounding_mask_iou_all_micro_mean"] = total_inter_all_t_mask / total_union_all_t_mask if total_union_all_t_mask > 0 else 0.0

    metrics["grounding_greedy_iou_all_pooled_mean_tn0"] = sum(all_ious_total_tn0_greedy) / len(all_ious_total_tn0_greedy) if all_ious_total_tn0_greedy else 0.0
    metrics["grounding_greedy_iou_all_macro_mean_tn0"] = sum(metrics[f"grounding_greedy_iou_all_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)
    metrics["grounding_greedy_iou_all_micro_mean"] = total_inter_all_t_greedy / total_union_all_t_greedy if total_union_all_t_greedy > 0 else 0.0

    # Exist Aggregates (Ignoring True Negatives)
    metrics["grounding_mask_iou_exist_macro_mean"] = sum(
        (sum(class_ious_exist_tn0_mask[cls]) / len(class_ious_exist_tn0_mask[cls])) 
        if class_ious_exist_tn0_mask[cls] else 0.0 for cls in GROUNDING_CLASSES
    ) / len(GROUNDING_CLASSES)

    metrics["grounding_greedy_iou_exist_macro_mean"] = sum(
        (sum(class_ious_exist_tn0_greedy[cls]) / len(class_ious_exist_tn0_greedy[cls])) 
        if class_ious_exist_tn0_greedy[cls] else 0.0 for cls in GROUNDING_CLASSES
    ) / len(GROUNDING_CLASSES)

    return metrics