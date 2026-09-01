"""
Converts raw ConstructionSite 10k samples into the Unsloth multimodal
conversation format for SFT training.

Each training sample becomes a dict with:
  {
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
      {"role": "user",   "content": [{"type": "image", "image": pil_image},
                                      {"type": "text",  "text": INSPECTION_PROMPT}]},
      {"role": "assistant", "content": [{"type": "text", "text": target_json_str}]}
    ]
  }

The target JSON is minimized (no indentation) and wrapped in ```json ... ``` fences.
Bounding boxes are scaled from dataset [0,1] to Qwen3-VL [0,1000].
"""
import json
from typing import Any, Dict, List, Optional
from datasets import Dataset, Image as HFImage

from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT  # noqa: F401  (UNIFIED_INSPECTION_PROMPT kept for the legacy task-blind builders below)
from data.box_utils import normalize_boxes, clean_boxes, scale_01_to_1000
from core.constants import GROUNDING_CLASSES, RULES
from core.logging import get_logger

logger = get_logger(__name__)


def _build_target_json(raw: Dict[str, Any]) -> str:
    """Builds the minimized JSON target string wrapped in code fences.

    Returns flat JSON with rule_X_violation and object keys at the root.
    """
    target_dict = {"caption": raw.get("image_caption", "")}
    
    for i in range(1, 5):
        v = raw.get(f"rule_{i}_violation")
        if v is None:
            target_dict[f"rule_{i}_violation"] = None
        else:
            raw_boxes = v.get("bounding_box") if isinstance(v, dict) else None
            boxes = clean_boxes(normalize_boxes(raw_boxes))
            target_dict[f"rule_{i}_violation"] = {
                "bounding_box": [scale_01_to_1000(b) for b in boxes],
                "reason": (v.get("reason", "") if isinstance(v, dict) else "") or "",
            }
            
    for cls in GROUNDING_CLASSES:
        raw_boxes = raw.get(cls, [])
        boxes = clean_boxes(normalize_boxes(raw_boxes))
        target_dict[cls] = [scale_01_to_1000(b) for b in boxes]
        
    # Minimized: no indent, compact separators
    json_str = json.dumps(target_dict, separators=(",", ":"), ensure_ascii=False)
    return f"```json\n{json_str}\n```"


def raw_sample_to_conversation(raw: Dict[str, Any], pil_image) -> Dict[str, Any]:
    """Converts a single raw dataset sample into Unsloth multimodal conversation format.

    Args:
        raw: Dict from the HF dataset (one row).
        pil_image: The PIL Image object for this sample.

    Returns:
        Dict with "messages" key containing the system/user/assistant conversation.
    """
    target_str = _build_target_json(raw)

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": UNIFIED_INSPECTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_str}],
            },
        ]
    }


def build_unified_sft_dataset(
    hf_dataset,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Converts a full HF dataset split into a list of Unsloth conversation dicts.

    Args:
        hf_dataset: A HuggingFace Dataset split (train or val).
        max_samples: Optional cap on number of samples (for debugging).

    Returns:
        List of conversation dicts ready for SFTTrainer.
    """
    dataset_iter = hf_dataset
    if max_samples is not None:
        dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    conversations = []
    skipped = 0
    for sample in dataset_iter:
        try:
            pil_image = sample["image"]  # PIL Image from HF datasets
            conv = raw_sample_to_conversation(sample, pil_image)
            conversations.append(conv)
        except Exception as e:
            skipped += 1
            logger.warning(
                f"Skipping sample {sample.get('image_id', '?')}: {e}"
            )

    logger.info(
        f"Built unified SFT dataset: {len(conversations)} samples "
        f"({skipped} skipped)"
    )
    return conversations


def build_ground_truth_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the ground-truth dict for evaluation comparison.

    Returns the same structure as the model output but with ground-truth values.
    Boxes remain in dataset [0,1] scale (evaluation handles scale conversion).
    """
    gt = {
        # image_id is carried so failure reports can name the offending image.
        # Without it, evaluator.py falls back to "unknown_0", "unknown_1", ... for
        # every record, making json_parse_failures.json / schema_validation_failures.json
        # impossible to trace back to a source image.
        "image_id": raw.get("image_id", ""),
        "caption": raw.get("image_caption", ""),
        "illumination": raw.get("illumination", ""),
        "camera_distance": raw.get("camera_distance", ""),
        "view": raw.get("view", ""),
        "quality_of_info": raw.get("quality_of_info", ""),
    }
    
    for i in range(1, 5):
        v = raw.get(f"rule_{i}_violation")
        if v is None:
            gt[f"rule_{i}_violation"] = None
        else:
            raw_boxes = v.get("bounding_box") if isinstance(v, dict) else None
            boxes = clean_boxes(normalize_boxes(raw_boxes))
            gt[f"rule_{i}_violation"] = {
                "bounding_box": [list(b) for b in boxes],
                "reason": (v.get("reason", "") if isinstance(v, dict) else "") or "",
            }
            
    for cls in GROUNDING_CLASSES:
        raw_boxes = raw.get(cls, [])
        boxes = clean_boxes(normalize_boxes(raw_boxes))
        gt[cls] = [list(b) for b in boxes]
        
    return gt


# ---------------------------------------------------------------------------
# GRPO prompt preparation
# ---------------------------------------------------------------------------

def to_grpo_prompt(raw: Dict[str, Any], pil_image) -> Dict[str, Any]:
    """Converts a single raw dataset sample into GRPO prompt format.

    Unlike SFT (which includes the assistant response for teacher forcing),
    GRPO only needs the prompt (system + user messages) and the ground truth
    for reward computation. The model generates its own response during
    rollouts.

    Args:
        raw: Dict from the HF dataset (one row).
        pil_image: The PIL Image object for this sample.

    Returns:
        Dict with:
            - "prompt": list of message dicts (system + user, no assistant)
            - "ground_truth": ground truth dict for reward computation
            - "image_id": the image identifier
    """
    prompt_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": UNIFIED_INSPECTION_PROMPT},
            ],
        },
    ]

    ground_truth = build_ground_truth_dict(raw)

    return {
        "prompt": prompt_messages,
        "ground_truth": json.dumps(ground_truth),
        "image_id": raw.get("image_id", ""),
        "image": pil_image,
    }


def build_grpo_dataset(
    hf_dataset,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Converts a full HF dataset split into GRPO prompt format.

    Args:
        hf_dataset: A HuggingFace Dataset split (train).
        max_samples: Optional cap on number of samples (for debugging).

    Returns:
        List of GRPO prompt dicts ready for GRPOTrainer.
    """
    dataset_iter = hf_dataset
    if max_samples is not None:
        if hasattr(hf_dataset, "select"):
            dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        else:
            dataset_iter = hf_dataset[:max_samples]

    prompts = []
    skipped = 0
    for sample in dataset_iter:
        try:
            pil_image = sample["image"]  # PIL Image from HF datasets
            prompt_dict = to_grpo_prompt(sample, pil_image)
            prompts.append(prompt_dict)
        except Exception as e:
            skipped += 1
            logger.warning(
                f"Skipping sample {sample.get('image_id', '?')}: {e}"
            )

    logger.info(
        f"Built GRPO prompt dataset: {len(prompts)} samples "
        f"({skipped} skipped)"
    )
    ds = Dataset.from_list(prompts)
    ds = ds.cast_column("image", HFImage())
    return ds


def _build_violations_only_target_json(raw: Dict[str, Any]) -> str:
    """Builds the minimized JSON target string wrapped in code fences for violations_only."""
    target_dict = {}
    
    for i in range(1, 5):
        v = raw.get(f"rule_{i}_violation")
        if v is None:
            target_dict[f"rule_{i}_violation"] = None
        else:
            raw_boxes = v.get("bounding_box") if isinstance(v, dict) else None
            boxes = clean_boxes(normalize_boxes(raw_boxes))
            target_dict[f"rule_{i}_violation"] = {
                "bounding_box": [scale_01_to_1000(b) for b in boxes],
                "reason": (v.get("reason", "") if isinstance(v, dict) else "") or "",
            }
            
    # Minimized: no indent, compact separators
    json_str = json.dumps(target_dict, separators=(",", ":"), ensure_ascii=False)
    return f"```json\n{json_str}\n```"


def build_violations_only_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the ground-truth dict for evaluation comparison for violations_only."""
    gt = {
        "image_id": raw.get("image_id", ""),
        "illumination": raw.get("illumination", ""),
        "camera_distance": raw.get("camera_distance", ""),
        "view": raw.get("view", ""),
        "quality_of_info": raw.get("quality_of_info", ""),
    }
    
    for i in range(1, 5):
        v = raw.get(f"rule_{i}_violation")
        if v is None:
            gt[f"rule_{i}_violation"] = None
        else:
            raw_boxes = v.get("bounding_box") if isinstance(v, dict) else None
            boxes = clean_boxes(normalize_boxes(raw_boxes))
            gt[f"rule_{i}_violation"] = {
                "bounding_box": [list(b) for b in boxes],
                "reason": (v.get("reason", "") if isinstance(v, dict) else "") or "",
            }
            
    return gt


def _build_object_only_target_json(raw: Dict[str, Any]) -> str:
    """Builds the minimized fenced-JSON SFT target for object_only.

    Only the 3 grounding classes. Boxes are cleaned in dataset [0,1] space (so
    is_valid_box's [0,1] range check still applies) and only then scaled to the
    Qwen3-VL [0,1000] space the model is trained to emit.
    """
    target_dict = {}
    for cls in GROUNDING_CLASSES:
        raw_boxes = raw.get(cls, [])
        boxes = clean_boxes(normalize_boxes(raw_boxes))
        target_dict[cls] = [scale_01_to_1000(b) for b in boxes]

    json_str = json.dumps(target_dict, separators=(",", ":"), ensure_ascii=False)
    return f"```json\n{json_str}\n```"


def build_object_only_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the evaluation ground-truth dict for object_only.

    Boxes remain in dataset [0,1] scale — evaluation/metrics_grounding.py rescales
    the *predictions* from [0,1000] and expects ground truth untouched.
    """
    gt = {
        "image_id": raw.get("image_id", ""),
        "illumination": raw.get("illumination", ""),
        "camera_distance": raw.get("camera_distance", ""),
        "view": raw.get("view", ""),
        "quality_of_info": raw.get("quality_of_info", ""),
    }
    for cls in GROUNDING_CLASSES:
        raw_boxes = raw.get(cls, [])
        boxes = clean_boxes(normalize_boxes(raw_boxes))
        gt[cls] = [list(b) for b in boxes]
    return gt


def _build_caption_only_target(raw: Dict[str, Any]) -> str:
    """Builds the SFT target for caption_only: the bare caption string.

    No JSON, no code fence — caption_only's wire format is plain prose (see
    core/tasks.py::FORMAT_PLAIN_TEXT). Emitting a fence here would train exactly
    the behaviour reward_format penalizes.
    """
    return str(raw.get("image_caption", "") or "").strip()


def build_caption_only_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the evaluation ground-truth dict for caption_only.

    ``caption`` is mandatory here: reward_caption only scores a sample when both
    the predicted and the reference caption are non-empty, and the captioning
    metrics compare against this field.
    """
    return {
        "image_id": raw.get("image_id", ""),
        "caption": raw.get("image_caption", ""),
        "illumination": raw.get("illumination", ""),
        "camera_distance": raw.get("camera_distance", ""),
        "view": raw.get("view", ""),
        "quality_of_info": raw.get("quality_of_info", ""),
    }


# ---------------------------------------------------------------------------
# Task routers
#
# Explicit dispatch tables, not if/else chains ending in a unified fallback. The
# old form (`if task == 'violations_only': ... ; return <unified>`) silently gave
# any unregistered task the full unified target — which would have taught an
# object_only model to emit captions and violations it is never evaluated on.
# ---------------------------------------------------------------------------

_TARGET_BUILDERS = {
    "unified": _build_target_json,
    "violations_only": _build_violations_only_target_json,
    "object_only": _build_object_only_target_json,
    "caption_only": _build_caption_only_target,
}

_GT_BUILDERS = {
    "unified": build_ground_truth_dict,
    "violations_only": build_violations_only_ground_truth,
    "object_only": build_object_only_ground_truth,
    "caption_only": build_caption_only_ground_truth,
}


def build_target_json(raw: Dict[str, Any], task: str = 'unified') -> str:
    """Returns the SFT assistant-turn target text for ``task``.

    Fenced minimized JSON for every task except caption_only, which returns bare
    prose. Raises ValueError for an unregistered task.
    """
    builder = _TARGET_BUILDERS.get(task)
    if builder is None:
        raise ValueError(
            f"No SFT target builder registered for task {task!r}. "
            f"Known: {sorted(_TARGET_BUILDERS)}"
        )
    return builder(raw)


def build_gt_dict(raw: Dict[str, Any], task: str = 'unified') -> Dict[str, Any]:
    """Returns the evaluation/reward ground-truth dict for ``task``.

    Boxes stay in dataset [0,1] scale for every task. Raises ValueError for an
    unregistered task.
    """
    builder = _GT_BUILDERS.get(task)
    if builder is None:
        raise ValueError(
            f"No ground-truth builder registered for task {task!r}. "
            f"Known: {sorted(_GT_BUILDERS)}"
        )
    return builder(raw)


def raw_sample_to_conversation_for_task(raw: Dict[str, Any], pil_image, task: str = 'unified') -> Dict[str, Any]:
    """Task-aware conversation builder."""
    from data.prompt_templates import get_prompt_for_task
    
    target_str = build_target_json(raw, task)
    prompt = get_prompt_for_task(task)

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_str}],
            },
        ]
    }


def build_sft_dataset(
    hf_dataset,
    task: str = 'unified',
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Task-aware SFT dataset builder."""
    # The task-aware builder now handles 'unified' natively via raw_sample_to_conversation_for_task
    dataset_iter = hf_dataset
    if max_samples is not None:
        dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    conversations = []
    skipped = 0
    for sample in dataset_iter:
        try:
            pil_image = sample["image"]
            conv = raw_sample_to_conversation_for_task(sample, pil_image, task)
            conversations.append(conv)
        except Exception as e:
            skipped += 1
            logger.warning(
                f"Skipping sample {sample.get('image_id', '?')}: {e}"
            )

    logger.info(
        f"Built {task} SFT dataset: {len(conversations)} samples "
        f"({skipped} skipped)"
    )
    return conversations


def to_grpo_prompt_for_task(raw: Dict[str, Any], pil_image, task: str = 'unified') -> Dict[str, Any]:
    """Task-aware GRPO prompt builder."""
    from data.prompt_templates import get_prompt_for_task
    prompt = get_prompt_for_task(task)

    prompt_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    ground_truth = build_gt_dict(raw, task)

    return {
        "prompt": prompt_messages,
        "ground_truth": json.dumps(ground_truth),
        "image_id": raw.get("image_id", ""),
        "image": pil_image,
    }


def build_grpo_dataset_for_task(
    hf_dataset,
    task: str = 'unified',
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Task-aware GRPO dataset builder."""
    # The task-aware builder now handles 'unified' natively via to_grpo_prompt_for_task
    dataset_iter = hf_dataset
    if max_samples is not None:
        if hasattr(hf_dataset, "select"):
            dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        else:
            dataset_iter = hf_dataset[:max_samples]

    prompts = []
    skipped = 0
    for sample in dataset_iter:
        try:
            pil_image = sample["image"]
            prompt_dict = to_grpo_prompt_for_task(sample, pil_image, task)
            prompts.append(prompt_dict)
        except Exception as e:
            skipped += 1
            logger.warning(
                f"Skipping sample {sample.get('image_id', '?')}: {e}"
            )

    logger.info(
        f"Built {task} GRPO prompt dataset: {len(prompts)} samples "
        f"({skipped} skipped)"
    )
    ds = Dataset.from_list(prompts)
    ds = ds.cast_column("image", HFImage())
    return ds