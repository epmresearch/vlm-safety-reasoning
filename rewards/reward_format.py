"""
Computes the format reward for GRPO.
Returns 1.0 if the completion parses to valid JSON and passes UnifiedOutput schema validation, 0.0 otherwise.
"""

from rewards.reward_utils import _strict_parse, _safe_reward
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    parsed = _strict_parse(completion)
    return 1.0 if parsed is not None else 0.0

compute_reward.__name__ = "reward_format"