"""
Single source of truth for all project-wide constants.

Naming convention:
  - HF column names (singular): used for dataset access and model output keys
  - Display names: used for logging and figure labels
  - Model IDs: Unsloth HuggingFace paths
"""

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
UNIFIED_TASK_NAME = "full_unified"

# ---------------------------------------------------------------------------
# Safety Rules
# ---------------------------------------------------------------------------
RULES = ["rule_1", "rule_2", "rule_3", "rule_4"]

RULE_DESCRIPTIONS = {
    "rule_1": (
        "Use of basic PPE when on foot (hard hats, appropriate clothing, "
        "closed-toe shoes, high-visibility vests at night, face protection "
        "for cutting/welding/grinding/drilling)."
    ),
    "rule_2": (
        "Use of a safety harness when working from a height of >3 meters "
        "with unprotected edges."
    ),
    "rule_3": (
        "Edge protection/warning for underground projects >3 meters deep "
        "with steep retaining walls."
    ),
    "rule_4": (
        "Worker appearing in an excavator's blind spot or operation radius."
    ),
}

# ---------------------------------------------------------------------------
# Object Grounding Classes
# Keys are the canonical names used in HF dataset columns AND model output.
# ---------------------------------------------------------------------------
GROUNDING_CLASSES = ["excavator", "rebar", "worker_with_white_hard_hat"]

# Mapping from canonical name → human-readable display name (for figures/tables)
GROUNDING_CLASS_DISPLAY = {
    "excavator": "Excavator",
    "rebar": "Rebar",
    "worker_with_white_hard_hat": "Worker (White Hard Hat)",
}

# ---------------------------------------------------------------------------
# Metadata fields — excluded from training, used for stratified evaluation
# ---------------------------------------------------------------------------
METADATA_FIELDS = ["illumination", "camera_distance", "view", "quality_of_info"]

# Stratification categories per metadata field (from dataset inspection)
METADATA_VALUES = {
    "illumination": ["normal lighting", "underexposed", "overexposed", "night"],
    "camera_distance": ["short distance", "mid distance", "long distance"],
    "view": ["elevation view", "bird's-eye view", "ground view"],
    "quality_of_info": ["rich", "average", "poor"],
}



# ---------------------------------------------------------------------------
# Bounding Box
# ---------------------------------------------------------------------------
BBOX_FORMAT = "[xmin, ymin, xmax, ymax]"
BBOX_SCALE_DATASET = (0.0, 1.0)     # Dataset native scale
BBOX_SCALE_QWEN = (0, 1000)          # Qwen3-VL native scale

# ---------------------------------------------------------------------------
# Validation Split
# ---------------------------------------------------------------------------
VALIDATION_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

DEFAULT_MAX_SEQ_LENGTH = 2048