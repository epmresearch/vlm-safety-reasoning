"""
Output parsing module for extracting JSON from VLM responses.
"""
import json
from typing import Any, Dict, Optional

from data.schemas import UnifiedOutput
from core.logging import get_logger

logger = get_logger(__name__)

import re

def strip_fences(text: str) -> str:
    """
    Strips markdown code fences (e.g., ```json ... ```) from a string.
    Uses regex to extract content between fences, ignoring any pre-text.
    """
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback if no fences are found
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
        logger.debug(f"Failed to parse JSON: {e}")
        return None

from core.constants import GROUNDING_CLASSES

def validate_unified_output(parsed_data: Dict[str, Any]) -> Optional[UnifiedOutput]:
    """
    Validates a parsed dictionary against the UnifiedOutput schema.
    """
    if parsed_data is None:
        return None
        
    try:
        return UnifiedOutput(**parsed_data)
    except Exception as e:
        logger.debug(f"Failed to validate UnifiedOutput schema: {e}")
        return None

def validate_output_for_task(parsed_data: Dict[str, Any], task: str = "unified") -> Optional[Any]:
    """Validates a parsed dictionary against the task-specific output schema.
    
    For task='unified', behaves identically to validate_unified_output().
    For task='violations_only', validates against ViolationsOnlyOutput
    (which does NOT require caption or grounding class fields).
    
    Args:
        parsed_data: Dictionary parsed from model JSON output.
        task: Task name ('unified' or 'violations_only').
    
    Returns:
        Validated Pydantic model instance, or None if validation fails.
    """
    if parsed_data is None:
        return None
    try:
        from data.schemas import get_output_schema
        schema_cls = get_output_schema(task)
        return schema_cls(**parsed_data)
    except Exception as e:
        logger.debug(f"Failed to validate {task} output schema: {e}")
        return None
