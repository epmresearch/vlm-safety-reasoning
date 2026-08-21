"""
Pydantic contracts — single source of truth for data schemas.

Covers:
  - Raw dataset sample (mirrors HF ConstructionSite schema)
  - Unified model output schema (what the VLM produces)
  - Evaluation result container
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, conlist


# ---------------------------------------------------------------------------
# Bounding box type alias
# ---------------------------------------------------------------------------
BBox = conlist(float, min_length=4, max_length=4)  # [xmin, ymin, xmax, ymax]


# ---------------------------------------------------------------------------
# Raw dataset schemas (mirrors HF ConstructionSite 10k exactly)
# ---------------------------------------------------------------------------

class RuleViolation(BaseModel):
    """A single rule violation from the raw dataset."""
    bounding_box: Optional[List[BBox]] = None  # list of lists, or None
    reason: Optional[str] = None


class RawSample(BaseModel):
    """Exactly matches the HF dataset record. No modification to structure."""
    image_id: str
    image_caption: str
    illumination: str
    camera_distance: str
    view: str
    quality_of_info: str
    rule_1_violation: Optional[RuleViolation] = None
    rule_2_violation: Optional[RuleViolation] = None
    rule_3_violation: Optional[RuleViolation] = None
    rule_4_violation: Optional[RuleViolation] = None
    excavator: List[BBox] = Field(default_factory=list)
    rebar: List[BBox] = Field(default_factory=list)
    worker_with_white_hard_hat: List[BBox] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Unified model output schema
# What the VLM is trained to produce in a single JSON response.
# ---------------------------------------------------------------------------

class UnifiedOutput(BaseModel):
    """The complete unified model output — one per image. Flat structure."""
    caption: str
    rule_1_violation: Optional[RuleViolation] = None
    rule_2_violation: Optional[RuleViolation] = None
    rule_3_violation: Optional[RuleViolation] = None
    rule_4_violation: Optional[RuleViolation] = None
    excavator: List[BBox] = Field(default_factory=list)
    rebar: List[BBox] = Field(default_factory=list)
    worker_with_white_hard_hat: List[BBox] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Violations-only model output schema
# Only the 4 rule violations — no caption, no object grounding.
# ---------------------------------------------------------------------------

class ViolationsOnlyOutput(BaseModel):
    """Output schema for the violations-only task.
    Only the 4 rule violations — no caption, no object grounding."""
    rule_1_violation: Optional[RuleViolation] = None
    rule_2_violation: Optional[RuleViolation] = None
    rule_3_violation: Optional[RuleViolation] = None
    rule_4_violation: Optional[RuleViolation] = None


# ---------------------------------------------------------------------------
# Schema registry — maps task name to output Pydantic model
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY = {
    "unified": UnifiedOutput,
    "violations_only": ViolationsOnlyOutput,
}


def get_output_schema(task: str) -> type:
    """Returns the Pydantic output schema class for the given task.

    Args:
        task: Task name (e.g., 'unified', 'violations_only').

    Returns:
        The Pydantic BaseModel subclass for that task's output.

    Raises:
        ValueError: If the task name is not in the registry.
    """
    if task not in SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown task: {task!r}. Known tasks: {list(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[task]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Per-image evaluation result with all computed metrics."""
    image_id: str
    task: str
    model_id: str
    prediction: Dict[str, Any]
    ground_truth: Dict[str, Any]
    scores: Dict[str, float]
    raw_output: str = ""   # Raw model output string (for debugging)
    parse_success: bool = False
    schema_valid: bool = False