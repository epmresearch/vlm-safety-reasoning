"""
Computes the object grounding IoU reward for GRPO.
Calculates IoU for specific classes and correctly handles true negatives.
"""

from rewards.reward_utils import _strict_parse, _safe_reward
from data.box_utils import normalize_boxes, clean_boxes, scale_1000_to_01, compute_mask_union_iou
from core.constants import GROUNDING_CLASSES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    parsed = _strict_parse(completion)
    if parsed is None:
        return 0.0

    class_scores = []
    for cls in GROUNDING_CLASSES:
        pred_boxes = clean_boxes([scale_1000_to_01(b) for b in normalize_boxes(parsed.get(cls, []))])
        gt_boxes = clean_boxes(normalize_boxes(ground_truth.get(cls, [])))

        if not gt_boxes and not pred_boxes:
            class_scores.append(1.0)  # Correct True Negative
        elif not gt_boxes and pred_boxes:
            class_scores.append(0.0)  # False Positive: hallucinated
        elif gt_boxes and not pred_boxes:
            class_scores.append(0.0)  # False Negative: missed
        else:
            # True Positive: both present
            mask_result = compute_mask_union_iou(pred_boxes, gt_boxes)
            iou = mask_result["iou"] if mask_result["iou"] is not None else 0.0
            class_scores.append(0.5 + 0.5 * iou)  # 0.5 presence + 0.5*IoU quality

    return sum(class_scores) / len(class_scores) if class_scores else 0.0

compute_reward.__name__ = "reward_grounding"
