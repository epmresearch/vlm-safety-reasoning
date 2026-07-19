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
# Unified model output schema (the ONLY output schema)
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