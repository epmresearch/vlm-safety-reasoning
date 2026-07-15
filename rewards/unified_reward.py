"""
Unified composite reward for GRPO training.

Parses the unified JSON output once, then calls individual reward functions
(json_validity, grounding_iou, rule_violation_accuracy, caption_quality)
and returns a weighted sum.

Weights default to those in configs/tasks/unified.yaml but can be overridden.
"""
from typing import Any, Dict, Optional

from core.logging import get_logger
from rewards import (
    caption_quality,
    grounding_iou,
    json_validity,
    rule_violation_accuracy,
)

logger = get_logger(__name__)

# Default weights from configs/tasks/unified.yaml.
# Keys match the reward function module names.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "json_validity": 1.0,
    "caption_quality": 1.0,
    "rule_violation_accuracy": 2.0,
    "grounding_iou": 1.5,
}

# Map of reward name → compute_reward callable
_REWARD_FUNCTIONS: Dict[str, Any] = {
    "json_validity": json_validity.compute_reward,
    "caption_quality": caption_quality.compute_reward,
    "rule_violation_accuracy": rule_violation_accuracy.compute_reward,
    "grounding_iou": grounding_iou.compute_reward,
}


def compute_reward(
    prediction: str,
    ground_truth: dict,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite reward: weighted sum of individual reward functions.

    The prediction is parsed once by json_validity (which gives 0/1). If
    JSON is invalid, only the json_validity score contributes (all others
    would be 0 anyway since they also call try_parse_json internally).

    Args:
        prediction: Raw model output string (with ```json fences).
        ground_truth: Ground truth dict with keys:
            caption, rule_X_violation, and object classes.
        weights: Optional weight overrides. Keys must be a subset of
            DEFAULT_WEIGHTS. Missing keys use defaults.

    Returns:
        Weighted sum of rewards, normalized to [0, 1] by dividing by
        the sum of weights.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    total_weight = sum(w.values())
    if total_weight <= 0:
        return 0.0

    weighted_sum = 0.0
    component_scores: Dict[str, float] = {}

    for name, reward_fn in _REWARD_FUNCTIONS.items():
        weight = w.get(name, 0.0)
        if weight <= 0:
            continue
        score = reward_fn(prediction, ground_truth)
        component_scores[name] = score
        weighted_sum += score * weight

    logger.debug(
        "Component scores: %s → weighted=%.4f",
        component_scores,
        weighted_sum / total_weight,
    )

    return weighted_sum / total_weight


def compute_reward_with_breakdown(
    prediction: str,
    ground_truth: dict,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Like compute_reward but also returns per-component scores.

    Useful for logging during training.

    Returns:
        Dict with keys for each component score plus 'total'.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    total_weight = sum(w.values())
    if total_weight <= 0:
        return {"total": 0.0}

    result: Dict[str, float] = {}
    weighted_sum = 0.0

    for name, reward_fn in _REWARD_FUNCTIONS.items():
        weight = w.get(name, 0.0)
        if weight <= 0:
            continue
        score = reward_fn(prediction, ground_truth)
        result[name] = score
        weighted_sum += score * weight

    result["total"] = weighted_sum / total_weight
    return result
