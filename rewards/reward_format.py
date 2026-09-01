"""
Computes the format reward for GRPO.

For a fenced-JSON task: 1.0 if the completion parses to valid JSON and passes schema
validation for that task, 0.0 otherwise.

For a plain-text task (caption_only): 1.0 if the completion is clean prose that
yields a non-blank caption, 0.0 otherwise. "Clean" means the model did not fall back
to the JSON habits the other three tasks train — no code fence, no JSON object, no
key/value pairs. Without that check the reward would be free (any non-empty string
parses), and a caption_only model that emitted `{"caption": "..."}` would be scored
as perfectly formatted despite ignoring the prompt.
"""
from evaluation.output_parser import is_clean_prose
from rewards.reward_utils import _strict_parse_for_task, _safe_reward
from core.logging import get_logger

logger = get_logger(__name__)


@_safe_reward
def compute_reward(completion: str, ground_truth: dict, task: str = "unified", **kwargs) -> float:
    """Task-aware format reward.

    Validates against the correct Pydantic schema for the given task via
    _strict_parse_for_task (which also applies the task's wire format), then for
    plain-text tasks additionally requires the raw completion to be clean prose.
    """
    from core.tasks import is_plain_text_task

    # Normalize a chat-style completion down to its text, matching what
    # _strict_parse_for_task does, so the prose check sees the same string.
    raw = completion
    if isinstance(raw, list):
        raw = raw[-1].get("content", "") if raw else ""
    elif not isinstance(raw, str):
        raw = str(raw)

    parsed = _strict_parse_for_task(completion, task=task)
    if parsed is None:
        return 0.0

    if is_plain_text_task(task) and not is_clean_prose(raw):
        return 0.0

    return 1.0


compute_reward.__name__ = "reward_format"

# Keep compute_reward_for_task as an explicit alias for backwards compatibility
# (tests and any older caller reference it by this name).
compute_reward_for_task = compute_reward
