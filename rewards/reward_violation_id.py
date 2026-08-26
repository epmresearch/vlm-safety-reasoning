"""
Violation identification using F-beta (beta=2, recall-weighted) over predicted vs ground-truth rule sets.
"""

from rewards.reward_utils import _strict_parse_for_task, _safe_reward, _is_violation_present
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
    for r in RULES:
        if _is_violation_present(parsed.get(f"{r}_violation")):
            pred_rules.add(r)
        if _is_violation_present(ground_truth.get(f"{r}_violation")):
            gt_rules.add(r)

    # Both empty = correctly identified as safe
    # We return 0.15 instead of 1.0 to mathematically balance the Expected Value
    # against the 91% class imbalance, preventing the "Predict Safe Always" local minimum.
    if not pred_rules and not gt_rules:
        return 0.15

    tp = len(pred_rules & gt_rules)
    fp = len(pred_rules - gt_rules)
    fn = len(gt_rules - pred_rules)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta = 2.0  # recall-weighted for safety inspection domain
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

compute_reward.__name__ = "reward_violation_id"
