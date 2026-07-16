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
        
    # Rescue operation for hallucinated nested arrays
    for hallucinated_key in ["detected_objects", "spatial_grounding"]:
        if hallucinated_key in parsed_data and isinstance(parsed_data[hallucinated_key], list):
            for obj in parsed_data[hallucinated_key]:
                if isinstance(obj, dict):
                    # Sometimes it outputs 'type', sometimes 'object'
                    cls = obj.get("type") or obj.get("object")
                    # Sometimes it outputs 'bounding_box', sometimes 'bbox', sometimes 'coordinates'
                    bbox = obj.get("bounding_box") or obj.get("bbox") or obj.get("coordinates")
                    
                    if cls in GROUNDING_CLASSES and bbox is not None:
                        if not parsed_data.get(cls):
                            parsed_data[cls] = []
                        if isinstance(parsed_data[cls], list):
                            # Ensure we don't nest it if it's already a list of lists, but be safe
                            if len(bbox) > 0 and isinstance(bbox[0], list):
                                parsed_data[cls].extend(bbox)
                            else:
                                parsed_data[cls].append(bbox)
    try:
        return UnifiedOutput(**parsed_data)
    except Exception as e:
        logger.debug(f"Failed to validate UnifiedOutput schema: {e}")
        return None
