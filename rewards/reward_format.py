"""
Computes the format reward for GRPO.
Returns 1.0 if the completion parses to valid JSON and passes schema
validation for the given task, 0.0 otherwise.
"""

from rewards.reward_utils import _strict_parse_for_task, _safe_reward
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, task: str = "unified", **kwargs) -> float:
    """Task-aware format reward.

    Validates against the correct Pydantic schema for the given task
    (UnifiedOutput for 'unified', ViolationsOnlyOutput for 'violations_only').
    """
    parsed = _strict_parse_for_task(completion, task=task)
    return 1.0 if parsed is not None else 0.0

compute_reward.__name__ = "reward_format"

# Keep compute_reward_for_task as an explicit alias for backwards compatibility
# (used by _make_task_aware_batch_reward's special-case path).
compute_reward_for_task = compute_reward