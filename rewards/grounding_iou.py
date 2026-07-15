"""
Grounding IoU reward for GRPO training.

Computes multi-box IoU between predicted and ground-truth bounding boxes
for each grounding class (excavator, rebar, worker_with_white_hard_hat),
then averages across classes.

Uses greedy_multibox_iou to match the evaluation metric — each GT box is
greedily matched to its best remaining predicted box.

Scale conversion: predictions are in [0, 1000], ground truth in [0, 1].
"""
from typing import Any, Dict, List

from core.constants import GROUNDING_CLASSES
from core.logging import get_logger
from data.box_utils import (
    clean_boxes,
    greedy_multibox_iou,
    normalize_boxes,
    scale_1000_to_01,
)
from rewards.json_validity import try_parse_json

logger = get_logger(__name__)


def _extract_pred_boxes(parsed: Dict[str, Any], cls_name: str) -> List[List[float]]:
    """Extract and scale predicted boxes for a grounding class.

    Predicted boxes are in [0, 1000] scale; converts to [0, 1].
    """
    raw_boxes = parsed.get(cls_name, [])
    boxes = normalize_boxes(raw_boxes)
    scaled = [scale_1000_to_01(b) for b in boxes if len(b) == 4]
    return clean_boxes(scaled)


def _extract_gt_boxes(ground_truth: dict, cls_name: str) -> List[List[float]]:
    """Extract ground truth boxes for a grounding class.

    GT boxes are already in [0, 1] scale.
    """
    raw_boxes = ground_truth.get(cls_name, [])
    return clean_boxes(normalize_boxes(raw_boxes))


def compute_reward(prediction: str, ground_truth: dict) -> float:
    """Reward function for object grounding IoU.

    Parses the prediction JSON, extracts detected_objects for each class,
    converts predicted boxes from [0, 1000] to [0, 1], then computes
    greedy_multibox_iou against ground truth. Returns the mean IoU across
    all grounding classes.

    Args:
        prediction: Raw model output string (fenced JSON).
        ground_truth: Ground truth dict with 'detected_objects' key.

    Returns:
        Mean IoU across grounding classes, in [0, 1].
    """
    parsed = try_parse_json(prediction)
    if parsed is None:
        return 0.0

    class_ious: List[float] = []
    for cls_name in GROUNDING_CLASSES:
        pred_boxes = _extract_pred_boxes(parsed, cls_name)
        gt_boxes = _extract_gt_boxes(ground_truth, cls_name)
        iou = greedy_multibox_iou(pred_boxes, gt_boxes)
        class_ious.append(iou)

    if not class_ious:
        return 0.0

    return sum(class_ious) / len(class_ious)