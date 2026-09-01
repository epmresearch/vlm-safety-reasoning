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


_FENCE_RE = re.compile(r"```")
_JSON_OBJECT_RE = re.compile(r"^\s*[{\[]")
_CAPTION_KEY_RE = re.compile(r"""["']caption["']\s*:""", re.IGNORECASE)


def is_clean_prose(completion: str) -> bool:
    """True if a raw completion carries no JSON/code-fence formatting artifacts.

    The plain-text contract for caption_only: no fence, no JSON object/array, no
    ``"caption":`` label. Shared by rewards/reward_format.py (so the format reward
    actually discriminates instead of being free) and by
    preprocessing/structural_repair.py (so it can tell "already clean" from
    "recoverable").
    """
    if not isinstance(completion, str):
        return False
    if _FENCE_RE.search(completion):
        return False
    if _JSON_OBJECT_RE.search(completion):
        return False
    if _CAPTION_KEY_RE.search(completion):
        return False
    return True


def _parse_plain_caption(raw_str: str) -> Optional[Dict[str, Any]]:
    """Parses a plain-prose caption completion into ``{"caption": <text>}``.

    Used by the caption_only task, whose model is prompted to emit bare prose so
    that JSON formatting is not a confound on the caption-quality measurement.

    Deliberately tolerant, in this order:
      1. A fenced block -> use its contents (the model wrapped prose in a fence).
      2. A JSON object carrying a ``caption`` key -> unwrap it (the model reverted
         to the unified/vo habit). A list-valued caption is joined.
      3. A JSON string literal -> unwrap the quotes.
      4. Anything else -> treat the whole stripped text as the caption.

    Returns None only when nothing usable is left, so an empty or whitespace-only
    completion cannot collect the format reward.
    """
    if raw_str is None:
        return None
    if not isinstance(raw_str, str):
        raw_str = str(raw_str)

    text = strip_fences(raw_str)
    if not text.strip():
        return None

    # The model may have emitted JSON anyway. Only treat it as such if it actually
    # parses AND carries a caption — a prose caption that merely happens to contain
    # a brace must not be mangled.
    if text.lstrip().startswith(("{", '"')):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            cap = obj.get("caption")
            if isinstance(cap, list):
                cap = " ".join(str(x) for x in cap if x)
            if isinstance(cap, str) and cap.strip():
                return {"caption": cap.strip()}
            # A dict with no usable caption is not a caption_only answer.
            return None
        if isinstance(obj, str) and obj.strip():
            return {"caption": obj.strip()}

    return {"caption": text.strip()}


def parse_output_for_task(raw_str: str, task: str = "unified") -> Optional[Dict[str, Any]]:
    """Task-aware raw-completion parser.

    The single entry point every consumer of a raw model completion should use:
    the reward gate (rewards/reward_utils.py), the evaluator, the structural
    metrics, and structural repair. For fenced-JSON tasks it is exactly
    ``parse_model_output``; for plain-text tasks (caption_only) it routes to
    ``_parse_plain_caption`` so the rest of the stack keeps receiving a dict.
    """
    from core.tasks import is_plain_text_task

    if is_plain_text_task(task):
        return _parse_plain_caption(raw_str)
    return parse_model_output(raw_str)


def serialize_output_for_task(payload: Dict[str, Any], task: str) -> str:
    """Renders a parsed/repaired payload back into the task's wire format.

    Inverse of ``parse_output_for_task``. Structural repair uses it to write the
    repaired ``raw_output`` so evaluation re-parses it with the same contract the
    model was trained on: minimized fenced JSON for JSON tasks, bare prose for
    plain-text tasks.
    """
    from core.tasks import is_plain_text_task

    if is_plain_text_task(task):
        return str((payload or {}).get("caption", "") or "")
    # Bare JSON, no fence: strip_fences falls back to the whole string when no
    # fence is present, and this is byte-for-byte what the repair driver has
    # always written for JSON tasks.
    return json.dumps(payload, ensure_ascii=False)


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
    # Resolved OUTSIDE the try: an unregistered task raises ValueError here, and
    # swallowing it would turn a configuration mistake into a 100%-schema-failure
    # report that looks like a bad model.
    from data.schemas import get_output_schema
    schema_cls = get_output_schema(task)
    try:
        return schema_cls(**parsed_data)
    except Exception as e:
        logger.debug(f"Failed to validate {task} output schema: {e}")
        return None
