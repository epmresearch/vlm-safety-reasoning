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
    Also rescues boxes from hallucinated 'detected_objects' arrays.
    """
    if parsed_data is None:
        return None
        
    # Rescue operation for hallucinated detected_objects
    if "detected_objects" in parsed_data and isinstance(parsed_data["detected_objects"], list):
        for obj in parsed_data["detected_objects"]:
            if isinstance(obj, dict) and "type" in obj and "bounding_box" in obj:
                cls = obj["type"]
                if cls in GROUNDING_CLASSES:
                    if not parsed_data.get(cls):
                        parsed_data[cls] = []
                    if isinstance(parsed_data[cls], list):
                        parsed_data[cls].append(obj["bounding_box"])
    try:
        return UnifiedOutput(**parsed_data)
    except Exception as e:
        logger.debug(f"Failed to validate UnifiedOutput schema: {e}")
        return None
