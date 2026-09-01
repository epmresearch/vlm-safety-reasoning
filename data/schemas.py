"""
Pydantic contracts — single source of truth for data schemas.

Covers:
  - Raw dataset sample (mirrors HF ConstructionSite schema)
  - Unified model output schema (what the VLM produces)
  - Evaluation result container
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, conlist, field_validator


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
# Object-only model output schema
# Only the 3 object grounding classes — no caption, no violations.
#
# All three fields are REQUIRED, unlike UnifiedOutput's object fields which carry
# default_factory=list. The asymmetry is deliberate. Under `unified`, `caption` is
# required and therefore anchors the strict schema gate, so the object fields can
# default harmlessly. object_only has no such anchor: with every field defaulting,
# the schema would validate `{}` and even `{"excavators": [...]}`, which has three
# bad consequences —
#   1. structural_schema_adherence_rate collapses onto json_validity_rate and
#      stops measuring anything,
#   2. reward_format becomes free (any parseable JSON object scores 1.0),
#   3. structural repair's key-alias pass never runs, because the strict gate
#      already accepted the output — so a real detection emitted under the alias
#      `excavators` would be silently scored as "no objects detected".
# Requiring the keys also matches the prompt, which says to return [] for an
# absent class rather than to omit it. Post-repair evaluation is unaffected either
# way: metrics_grounding reads pred.get(cls, []), so a missing key and an empty
# list already score identically.
# ---------------------------------------------------------------------------

class ObjectOnlyOutput(BaseModel):
    """Output schema for the object-only task.
    Only the 3 grounding classes — no caption, no rule violations."""
    excavator: List[BBox]
    rebar: List[BBox]
    worker_with_white_hard_hat: List[BBox]


# ---------------------------------------------------------------------------
# Caption-only model output schema
# A single scene description. The model emits BARE PROSE for this task (no JSON,
# no code fence) — evaluation/output_parser.py::parse_output_for_task wraps the
# raw completion into {"caption": ...} before validation, so the rest of the
# stack (rewards, structural repair, metrics) stays dict-shaped and unchanged.
# ---------------------------------------------------------------------------

class CaptionOnlyOutput(BaseModel):
    """Output schema for the caption-only task. One non-blank caption string."""
    caption: str

    @field_validator("caption")
    @classmethod
    def _caption_must_not_be_blank(cls, v: str) -> str:
        # For caption_only the caption IS the entire output, so an empty string is
        # not a degenerate-but-valid answer the way it is under `unified` (where
        # the other fields still carry signal) — it is a non-answer. Rejecting it
        # here is what stops an empty completion collecting the format reward.
        if not isinstance(v, str) or not v.strip():
            raise ValueError("caption must be a non-empty, non-whitespace string")
        return v


# ---------------------------------------------------------------------------
# Schema registry — maps task name to output Pydantic model
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY = {
    "unified": UnifiedOutput,
    "violations_only": ViolationsOnlyOutput,
    "object_only": ObjectOnlyOutput,
    "caption_only": CaptionOnlyOutput,
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