"""
Pydantic contracts — the single source of truth for how data flows through
the system. Mirrors the ConstructionSite 10k schema exactly (unmodified).
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

BBox = Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax], normalized 0-1


class RuleViolation(BaseModel):
    bounding_box: Optional[List[BBox]] = None
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


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class SFTSample(BaseModel):
    """Chat-format training example for a given task."""
    image_id: str
    task: str
    messages: List[ChatMessage]


class GRPOPrompt(BaseModel):
    """Prompt-only format (no assistant answer) for GRPO rollouts."""
    image_id: str
    task: str
    prompt_messages: List[ChatMessage]
    ground_truth: dict  # task-specific ground-truth fields used by reward functions


# --- Per-task model output schemas (what the VLM must produce as JSON) ---

class RuleViolationOutput(BaseModel):
    rule_id: str          # "rule_1" | "rule_2" | "rule_3" | "rule_4" | "none"
    violated: bool
    reasoning: str
    bounding_box: Optional[BBox] = None


class CaptioningOutput(BaseModel):
    caption: str


class GroundingOutput(BaseModel):
    class_name: str
    bounding_boxes: List[BBox] = Field(default_factory=list)


class AttributesOutput(BaseModel):
    illumination: str
    camera_distance: str
    view: str
    quality_of_info: str


class EvaluationResult(BaseModel):
    image_id: str
    task: str
    model_id: str
    prediction: dict
    ground_truth: dict
    scores: dict  # metric_name -> float