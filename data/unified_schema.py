"""
Pydantic contracts. RawSample mirrors the ConstructionSite 10k schema exactly.
UnifiedOutput is the SINGLE schema the model is trained/evaluated to produce —
one JSON object per image containing caption + detected objects + safety
violations, all from one prompt.
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

BBox = Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax], normalized 0-1

class RuleViolation(BaseModel):
    bounding_box: Optional[List[BBox]] = None
    reason: Optional[str] = None

class RawSample(BaseModel):
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

# --- Chat / SFT sample containers ---

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class SFTSample(BaseModel):
    image_id: str
    task: str  # always "full_unified" now
    messages: List[ChatMessage]

class GRPOPrompt(BaseModel):
    """Kept for later — not used while GRPO is paused."""
    image_id: str
    task: str
    prompt_messages: List[ChatMessage]
    ground_truth: dict

# --- Unified model output schema (the ONLY output schema now) ---

class DetectedObjects(BaseModel):
    excavators: List[BBox] = Field(default_factory=list)
    rebar: List[BBox] = Field(default_factory=list)
    white_hard_hat_workers: List[BBox] = Field(default_factory=list)

class SafetyViolationEntry(BaseModel):
    rule_id: str          # "rule_1" | "rule_2" | "rule_3" | "rule_4"
    reason: str
    bounding_boxes: List[BBox] = Field(default_factory=list)  # 0..N boxes (multi-violator support)

class UnifiedOutput(BaseModel):
    caption: str
    detected_objects: DetectedObjects
    safety_violations: List[SafetyViolationEntry] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    image_id: str
    task: str
    model_id: str
    prediction: dict
    ground_truth: dict
    scores: dict