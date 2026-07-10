"""
Shared bounding-box helpers used by preprocessing and evaluation.
Box format everywhere: [ymin, xmin, ymax, xmax], normalized 0-1.
"""
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]


def is_valid_box(box: Optional[BBox], min_dim: float = 1e-4) -> bool:
    """Filters degenerate (zero-width/zero-height) or out-of-range boxes —
    per the 4 flagged cases found during data inspection (Task 1.4)."""
    if box is None or len(box) != 4:
        return False
    ymin, xmin, ymax, xmax = box
    if not all(-1e-6 <= c <= 1 + 1e-6 for c in box):
        return False
    return (xmax - xmin) > min_dim and (ymax - ymin) > min_dim


def clean_boxes(boxes: Optional[List[BBox]]) -> List[BBox]:
    if not boxes:
        return []
    return [tuple(b) for b in boxes if is_valid_box(b)]


def compute_iou(box_a: Optional[BBox], box_b: Optional[BBox]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ymin_a, xmin_a, ymax_a, xmax_a = box_a
    ymin_b, xmin_b, ymax_b, xmax_b = box_b

    inter_ymin = max(ymin_a, ymin_b)
    inter_xmin = max(xmin_a, xmin_b)
    inter_ymax = min(ymax_a, ymax_b)
    inter_xmax = min(xmax_a, xmax_b)

    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w

    area_a = max(0.0, ymax_a - ymin_a) * max(0.0, xmax_a - xmin_a)
    area_b = max(0.0, ymax_b - ymin_b) * max(0.0, xmax_b - xmin_b)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_multibox_iou(pred_boxes: List[BBox], gt_boxes: List[BBox]) -> float:
    """
    Handles the multi-violator / multi-object case (67 images had 2+ Rule 1
    violators alone). Greedily matches each GT box to its best remaining
    predicted box; score = mean IoU over matched GT boxes.
    - No GT, no pred  -> 1.0 (correctly predicted "nothing here")
    - No GT, some pred -> 0.0 (false positive)
    - Some GT, no pred -> 0.0 (missed everything)
    """
    gt_boxes = [tuple(b) for b in gt_boxes if is_valid_box(b)]
    pred_boxes = [tuple(b) for b in pred_boxes if is_valid_box(b)]

    if not gt_boxes and not pred_boxes:
        return 1.0
    if not gt_boxes or not pred_boxes:
        return 0.0

    remaining_pred = list(pred_boxes)
    matched_ious = []
    for gt in gt_boxes:
        if not remaining_pred:
            matched_ious.append(0.0)
            continue
        ious = [compute_iou(gt, p) for p in remaining_pred]
        best_idx = max(range(len(ious)), key=lambda i: ious[i])
        matched_ious.append(ious[best_idx])
        remaining_pred.pop(best_idx)

    return sum(matched_ious) / len(matched_ious)