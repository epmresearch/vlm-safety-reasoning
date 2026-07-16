"""
Bounding-box utilities used by preprocessing, evaluation, and rewards.

Coordinate convention:
    [xmin, ymin, xmax, ymax]

Two scales:
    Dataset native : normalized [0, 1]
    Qwen3-VL native: integer    [0, 1000]
"""
from typing import List, Optional, Tuple, Union

BBox = List[float]  # [xmin, ymin, xmax, ymax], 4 elements


# ---------------------------------------------------------------------------
# Normalization — handles edge cases in raw dataset
# ---------------------------------------------------------------------------

def normalize_boxes(raw_boxes: Optional[Union[list, None]]) -> List[List[float]]:
    """Ensures bounding boxes are always List[List[float]].

    Handles edge cases:
        None / null         → []
        []                  → []
        [x1,y1,x2,y2]      → [[x1,y1,x2,y2]]   (flat single-box)
        [[x1,y1,x2,y2],..] → [[x1,y1,x2,y2],..]  (already correct)
    """
    if not isinstance(raw_boxes, list) or len(raw_boxes) == 0:
        return []
    # If first element is a number, it's a flat single-box
    if isinstance(raw_boxes[0], (int, float)):
        return [list(raw_boxes)]
        
    valid_boxes = []
    for b in raw_boxes:
        if isinstance(b, (list, tuple)):
            valid_boxes.append(list(b))
        elif isinstance(b, dict):
            # Model hallucinated a dictionary wrapper
            if "bounding_box" in b and isinstance(b["bounding_box"], (list, tuple)):
                valid_boxes.append(list(b["bounding_box"]))
            elif all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                try:
                    valid_boxes.append([float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])])
                except (ValueError, TypeError):
                    pass
    return valid_boxes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_box(box: Optional[BBox], min_dim: float = 1e-4) -> bool:
    """Filters degenerate (zero-area) or out-of-range boxes."""
    if box is None or not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
        
    # Guard against hallucinated non-numeric types (e.g., strings from bad JSON)
    if not all(isinstance(c, (int, float)) for c in box):
        return False
        
    xmin, ymin, xmax, ymax = box
    
    # We allow up to 1000 + epsilon to accommodate BOTH [0, 1] and [0, 1000] scale boxes
    if not all(-1e-6 <= c <= 1000 + 1e-6 for c in box):
        return False
        
    return (xmax - xmin) > min_dim and (ymax - ymin) > min_dim


def clean_boxes(boxes: Optional[List[BBox]]) -> List[BBox]:
    """Filters out invalid/degenerate boxes from a list."""
    if not boxes:
        return []
    return [list(b) for b in boxes if is_valid_box(b)]


# ---------------------------------------------------------------------------
# Scale conversion
# ---------------------------------------------------------------------------

def scale_01_to_1000(box_01: BBox) -> List[int]:
    """[xmin, ymin, xmax, ymax] in [0,1] → [0,1000] (integers)."""
    return [round(c * 1000) for c in box_01]


def scale_1000_to_01(box_1000: BBox) -> List[float]:
    """[xmin, ymin, xmax, ymax] in [0,1000] → [0,1] (floats)."""
    return [c / 1000.0 for c in box_1000]


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def compute_iou(box_a: Optional[BBox], box_b: Optional[BBox]) -> Tuple[float, float, float]:
    """Computes IoU between two boxes [xmin, ymin, xmax, ymax].

    Both boxes must be in the same scale (either [0,1] or [0,1000]).
    Returns (iou, inter_area, union_area).
    """
    if box_a is None or box_b is None:
        return 0.0, 0.0, 0.0
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b

    inter_xmin = max(xmin_a, xmin_b)
    inter_ymin = max(ymin_a, ymin_b)
    inter_xmax = min(xmax_a, xmax_b)
    inter_ymax = min(ymax_a, ymax_b)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xmax_a - xmin_a) * max(0.0, ymax_a - ymin_a)
    area_b = max(0.0, xmax_b - xmin_b) * max(0.0, ymax_b - ymin_b)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0, 0.0, 0.0
    return inter_area / union, inter_area, union


def greedy_multibox_iou(pred_boxes: List[BBox], gt_boxes: List[BBox]) -> Tuple[float, float, float]:
    """Greedy multi-box IoU matching.

    Handles multi-violator / multi-object cases. Greedily matches each GT box
    to its best remaining predicted box; score = mean IoU over matched GT boxes.

    Returns:
        (mean_iou, total_inter_area, total_union_area)
    """
    if not gt_boxes and not pred_boxes:
        return 1.0, 0.0, 0.0
    if not gt_boxes:
        # False Positives
        union_sum = sum(max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) for b in pred_boxes)
        return 0.0, 0.0, union_sum
    if not pred_boxes:
        # False Negatives
        union_sum = sum(max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) for b in gt_boxes)
        return 0.0, 0.0, union_sum

    remaining_pred = list(pred_boxes)
    matched_ious = []
    total_inter = 0.0
    total_union = 0.0
    
    for gt in gt_boxes:
        if not remaining_pred:
            matched_ious.append(0.0)
            area_gt = max(0.0, gt[2]-gt[0]) * max(0.0, gt[3]-gt[1])
            total_union += area_gt
            continue
            
        ious_data = [compute_iou(gt, p) for p in remaining_pred]
        best_idx = max(range(len(ious_data)), key=lambda i: ious_data[i][0])
        best_iou, best_inter, best_union = ious_data[best_idx]
        
        matched_ious.append(best_iou)
        total_inter += best_inter
        total_union += best_union
        
        remaining_pred.pop(best_idx)
        
    # Unmatched predictions (False Positives) add to total_union
    for p in remaining_pred:
        area_p = max(0.0, p[2]-p[0]) * max(0.0, p[3]-p[1])
        total_union += area_p

    return sum(matched_ious) / len(matched_ious), total_inter, total_union