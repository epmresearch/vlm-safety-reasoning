"""
Output parsing module for extracting JSON from VLM responses.
"""
import json
from typing import Any, Dict, Optional

from data.schemas import UnifiedOutput
from core.logging import get_logger

logger = get_logger(__name__)

def strip_fences(text: str) -> str:
    """
    Strips markdown code fences (e.g., ```json ... ```) from a string.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text.strip()

def parse_model_output(raw_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses a raw string from the VLM into a dictionary.
    Handles potential code fences.
    """
    text = strip_fences(raw_str)
    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None

def validate_unified_output(parsed_data: Dict[str, Any]) -> Optional[UnifiedOutput]:
    """
    Validates a parsed dictionary against the UnifiedOutput schema.
    """
    if parsed_data is None:
        return None
    try:
        return UnifiedOutput(**parsed_data)
    except Exception as e:
        logger.warning(f"Failed to validate UnifiedOutput schema: {e}")
        return None
