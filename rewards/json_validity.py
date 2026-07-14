"""
JSON validity reward for GRPO training.

Returns 1.0 if the model output is valid, parseable JSON (after stripping
code fences), 0.0 otherwise.
"""
import json
import re
from typing import Any, Dict, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def strip_fences(text: str) -> str:
    """Strip ```json ... ``` code fences from model output.

    Uses regex to extract content between fences, ignoring any pre-text.
    """
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback if no fences are found
    return text.strip()


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