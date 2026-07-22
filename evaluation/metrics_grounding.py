"""
Metrics for bounding box grounding evaluation.

ALL ground-truth boxes are collapsed into ONE region via geometric union, and ALL predicted boxes are likewise
collapsed into ONE region. IoU is computed between these two collapsed
regions directly - this is NOT per-instance box matching. Extra/spurious
predicted boxes therefore inflate the candidate union and can only hurt
(never help) the score once that extra area stops overlapping true GT -
correctly penalizing models that emit many redundant/hallucinated boxes.

Reports both:
  - IoU-ObjectExist  : only images where GT is non-empty
  - IoU-Total        : all images; the paper's text
    explicitly states FP cases (pred present, no GT) and FN cases (GT
    present, no pred) score 0. Default to `_tn0` if want use `_excl`.
"""
from typing import Dict, List, Any
import numpy as np

from data.box_utils import (
    normalize_boxes, clean_boxes, scale_1000_to_01, compute_mask_union_iou
)
from core.constants import GROUNDING_CLASSES
from core.logging import get_logger

logger = get_logger(__name__)


def compute_grounding_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes class-wise grounding IoU for object detection tasks, using
    whole-image union-region IoU.

    predictions: list of parsed model-output dicts. Bounding boxes are
                 expected in [0, 1000] scale (Qwen native).
    references:  list of ground-truth dicts. Bounding boxes are expected
                 in [0, 1] scale (dataset native).
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

    # Per-image IoU lists, three TN-handling variants for the "all/Total" bucket
    class_ious_all_tn0: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_all_tn1: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}
    class_ious_all_excl: Dict[str, List[float]] = {cls: [] for cls in GROUNDING_CLASSES}

    # "Exist" bucket - GT always non-empty here, so TN can never occur;
    # all three variants are structurally identical for this bucket. Kept
    # separate anyway for symmetry/clarity with the "all" bucket above.
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
            pred_boxes_1000 = normalize_boxes(pred_objs.get(cls, []))
            gt_boxes_01 = normalize_boxes(gt_objs.get(cls, []))

            pred_boxes_1000 = clean_boxes(pred_boxes_1000)
            gt_boxes_01 = clean_boxes(gt_boxes_01)

            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]

            result = compute_mask_union_iou(pred_boxes_01, gt_boxes_01)
            iou = result["iou"]  # None only when both lists are empty (TN)

            is_tn = not gt_boxes_01 and not pred_boxes_01
            is_fp = not gt_boxes_01 and bool(pred_boxes_01)
            is_fn = bool(gt_boxes_01) and not pred_boxes_01

            if is_tn:
                class_tn[cls] += 1
                class_ious_all_tn0[cls].append(0.0)
                class_ious_all_tn1[cls].append(1.0)
                # _excl: omitted entirely from this list, by design
            else:
                if is_fp:
                    class_fp[cls] += 1
                elif is_fn:
                    class_fn[cls] += 1
                class_ious_all_tn0[cls].append(iou)
                class_ious_all_tn1[cls].append(iou)
                class_ious_all_excl[cls].append(iou)

            class_inter_total[cls] += result["intersection"]
            class_union_total[cls] += result["union"]

            if gt_boxes_01:
                class_exist_n[cls] += 1
                class_ious_exist_tn0[cls].append(iou)
                class_ious_exist_tn1[cls].append(iou)
                class_ious_exist_excl[cls].append(iou)
                class_inter_exist[cls] += result["intersection"]
                class_union_exist[cls] += result["union"]

    # ---------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------
    metrics = {}

    total_inter_all_t = 0.0
    total_union_all_t = 0.0
    total_inter_all_e = 0.0
    total_union_all_e = 0.0

    all_ious_total_tn0, all_ious_total_tn1, all_ious_total_excl = [], [], []
    all_ious_exist_tn0, all_ious_exist_tn1, all_ious_exist_excl = [], [], []

    for cls in GROUNDING_CLASSES:
        # 1. Macro "Total" variants (paper's "IoU-Total")
        metrics[f"grounding_iou_all_macro_{cls}_tn0"] = (
            sum(class_ious_all_tn0[cls]) / len(class_ious_all_tn0[cls]) if class_ious_all_tn0[cls] else 0.0
        )
        metrics[f"grounding_iou_all_macro_{cls}_tn1"] = (
            sum(class_ious_all_tn1[cls]) / len(class_ious_all_tn1[cls]) if class_ious_all_tn1[cls] else 0.0
        )
        metrics[f"grounding_iou_all_macro_{cls}_excl"] = (
            sum(class_ious_all_excl[cls]) / len(class_ious_all_excl[cls]) if class_ious_all_excl[cls] else 0.0
        )

        all_ious_total_tn0.extend(class_ious_all_tn0[cls])
        all_ious_total_tn1.extend(class_ious_all_tn1[cls])
        all_ious_total_excl.extend(class_ious_all_excl[cls])

        # 2. Micro "Total" (pooled sum) — TN contributes 0/0 to both
        #    numerator and denominator regardless of variant.
        inter_t = class_inter_total[cls]
        union_t = class_union_total[cls]
        metrics[f"grounding_iou_all_micro_{cls}"] = inter_t / union_t if union_t > 0 else 0.0
        total_inter_all_t += inter_t
        total_union_all_t += union_t

        # 3. Macro "Exist" variants (paper's "IoU-ObjectExist")
        #    identical across tn0/tn1/excl by construction (TN can't occur here)
        metrics[f"grounding_iou_existing_macro_{cls}_tn0"] = (
            sum(class_ious_exist_tn0[cls]) / len(class_ious_exist_tn0[cls]) if class_ious_exist_tn0[cls] else 0.0
        )
        metrics[f"grounding_iou_existing_macro_{cls}_tn1"] = metrics[f"grounding_iou_existing_macro_{cls}_tn0"]
        metrics[f"grounding_iou_existing_macro_{cls}_excl"] = metrics[f"grounding_iou_existing_macro_{cls}_tn0"]

        all_ious_exist_tn0.extend(class_ious_exist_tn0[cls])
        all_ious_exist_tn1.extend(class_ious_exist_tn1[cls])
        all_ious_exist_excl.extend(class_ious_exist_excl[cls])

        # 4. Micro "Exist"
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

        metrics[f"grounding_iou_total_macro_{cls}"] = metrics[f"grounding_iou_all_macro_{cls}_excl"]
        metrics[f"grounding_iou_exist_macro_{cls}"] = metrics[f"grounding_iou_existing_macro_{cls}_excl"]

    # Pooled/macro-of-macro means across all classes
    metrics["grounding_iou_all_pooled_mean_tn0"] = sum(all_ious_total_tn0) / len(all_ious_total_tn0) if all_ious_total_tn0 else 0.0
    metrics["grounding_iou_all_pooled_mean_tn1"] = sum(all_ious_total_tn1) / len(all_ious_total_tn1) if all_ious_total_tn1 else 0.0
    metrics["grounding_iou_all_pooled_mean_excl"] = sum(all_ious_total_excl) / len(all_ious_total_excl) if all_ious_total_excl else 0.0

    metrics["grounding_iou_all_macro_mean_tn0"] = sum(metrics[f"grounding_iou_all_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_all_macro_mean_tn1"] = sum(metrics[f"grounding_iou_all_macro_{cls}_tn1"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_all_macro_mean_excl"] = sum(metrics[f"grounding_iou_all_macro_{cls}_excl"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)

    metrics["grounding_iou_all_micro_mean"] = total_inter_all_t / total_union_all_t if total_union_all_t > 0 else 0.0

    metrics["grounding_iou_existing_pooled_mean_tn0"] = sum(all_ious_exist_tn0) / len(all_ious_exist_tn0) if all_ious_exist_tn0 else 0.0
    metrics["grounding_iou_existing_pooled_mean_tn1"] = metrics["grounding_iou_existing_pooled_mean_tn0"]
    metrics["grounding_iou_existing_pooled_mean_excl"] = metrics["grounding_iou_existing_pooled_mean_tn0"]

    metrics["grounding_iou_existing_macro_mean_tn0"] = sum(metrics[f"grounding_iou_existing_macro_{cls}_tn0"] for cls in GROUNDING_CLASSES) / len(GROUNDING_CLASSES)
    metrics["grounding_iou_existing_macro_mean_tn1"] = metrics["grounding_iou_existing_macro_mean_tn0"]
    metrics["grounding_iou_existing_macro_mean_excl"] = metrics["grounding_iou_existing_macro_mean_tn0"]

    metrics["grounding_iou_existing_micro_mean"] = total_inter_all_e / total_union_all_e if total_union_all_e > 0 else 0.0

    return metrics