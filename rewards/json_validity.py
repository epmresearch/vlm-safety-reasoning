"""
JSON validity reward for GRPO training.

Returns 1.0 if the model output is valid, parseable JSON (after stripping
code fences), 0.0 otherwise.
"""
import json
from typing import Any, Dict, Optional

from core.logging import get_logger
from evaluation.output_parser import strip_fences

logger = get_logger(__name__)

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse model output as JSON, stripping fences first.

    Returns:
        Parsed dict on success, None on failure.
    """
    try:
        return json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def compute_reward(prediction: str, ground_truth: dict) -> float:
    """Reward function for JSON validity.

    Args:
        prediction: Raw model output string (potentially fenced).
        ground_truth: Ground truth dict (unused, kept for uniform signature).

    Returns:
        1.0 if prediction is valid JSON, 0.0 otherwise.
    """
    parsed = try_parse_json(prediction)
    return 1.0 if parsed is not None else 0.0