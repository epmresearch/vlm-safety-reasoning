"""
Rule violation accuracy reward for GRPO training.

Multi-label F1 over predicted vs. ground-truth rule IDs.

The model outputs a list of safety_violations, each with a rule_id.
Ground truth has the same structure. We treat the set of predicted
rule_ids as multi-label predictions and compute F1.
"""
from typing import Any, Dict, Set

from core.constants import RULES
from core.logging import get_logger
from rewards.json_validity import try_parse_json

logger = get_logger(__name__)


def _extract_pred_rule_ids(parsed: Dict[str, Any]) -> Set[str]:
    """Extract the set of predicted rule IDs from parsed output."""
    violations = parsed.get("safety_violations", [])
    if not isinstance(violations, list):
        return set()
    return {
        v.get("rule_id")
        for v in violations
        if isinstance(v, dict) and v.get("rule_id") in RULES
    }


def _extract_gt_rule_ids(ground_truth: dict) -> Set[str]:
    """Extract the set of ground truth rule IDs.

    Ground truth safety_violations is a list of dicts with rule_id keys.
    """
    violations = ground_truth.get("safety_violations", [])
    if not isinstance(violations, list):
        return set()
    return {
        v.get("rule_id")
        for v in violations
        if isinstance(v, dict) and v.get("rule_id") in RULES
    }


def _multi_label_f1(pred_ids: Set[str], gt_ids: Set[str]) -> float:
    """Compute multi-label F1 between predicted and GT rule ID sets.

    Special case: if both sets are empty (no violations in GT and model
    correctly predicts none), return 1.0 — the model is correct.
    """
    if not pred_ids and not gt_ids:
        return 1.0
    if not pred_ids or not gt_ids:
        return 0.0

    tp = len(pred_ids & gt_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(gt_ids)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_reward(prediction: str, ground_truth: dict) -> float:
    """Reward function for rule violation accuracy (multi-label F1).

    Args:
        prediction: Raw model output string (fenced JSON).
        ground_truth: Ground truth dict with 'safety_violations' key.

    Returns:
        F1 score in [0, 1].
    """
    parsed = try_parse_json(prediction)
    if parsed is None:
        return 0.0

    pred_ids = _extract_pred_rule_ids(parsed)
    gt_ids = _extract_gt_rule_ids(ground_truth)

    return _multi_label_f1(pred_ids, gt_ids)