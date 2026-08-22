"""
Violation bounding box IoU, conditioned on TP rules only (rules present in BOTH prediction and GT).
"""

from rewards.reward_utils import _strict_parse_for_task, _safe_reward, _is_violation_present
from data.box_utils import normalize_boxes, clean_boxes, scale_1000_to_01, compute_mask_union_iou
from core.constants import RULES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    task = kwargs.get("task", "unified")
    parsed = _strict_parse_for_task(completion, task=task)
    if parsed is None:
        return 0.0

    pred_rules = set()
    gt_rules = set()
    common_rules = []
    
    for r in RULES:
        has_pred = _is_violation_present(parsed.get(f"{r}_violation"))
        has_gt = _is_violation_present(ground_truth.get(f"{r}_violation"))
        
        if has_pred: pred_rules.add(r)
        if has_gt: gt_rules.add(r)
        if has_pred and has_gt:
            common_rules.append(r)

    if not common_rules:
        # If True Negative (safe image, correctly predicted safe), give baseline anti-hack reward
        if not pred_rules and not gt_rules:
            return 0.10
        return 0.0  # FP or FN (no TPs to score)

    ious = []
    for r in common_rules:
        pv = parsed.get(f"{r}_violation", {}) or {}
        gv = ground_truth.get(f"{r}_violation", {}) or {}

        pred_boxes_raw = pv.get("bounding_box", []) if isinstance(pv, dict) else []
        gt_boxes_raw = gv.get("bounding_box", []) if isinstance(gv, dict) else []

        pred_boxes = clean_boxes([scale_1000_to_01(b) for b in normalize_boxes(pred_boxes_raw)])
        gt_boxes = clean_boxes(normalize_boxes(gt_boxes_raw))

        mask_result = compute_mask_union_iou(pred_boxes, gt_boxes)
        iou = mask_result["iou"] if mask_result["iou"] is not None else 0.0
        ious.append(iou)

    return sum(ious) / len(ious) if ious else 0.0

compute_reward.__name__ = "reward_violation_grounding"
