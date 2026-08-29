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
from datasets import Dataset, Image as HFImage, Sequence

from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT
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
        "images": [pil_image],
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
    ds = ds.cast_column("images", Sequence(HFImage()))
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


def build_target_json(raw: Dict[str, Any], task: str = 'unified') -> str:
    """Router that calls the appropriate target builder based on task."""
    if task == 'violations_only':
        return _build_violations_only_target_json(raw)
    return _build_target_json(raw)


def build_gt_dict(raw: Dict[str, Any], task: str = 'unified') -> Dict[str, Any]:
    """Router that calls the appropriate ground truth builder based on task."""
    if task == 'violations_only':
        return build_violations_only_ground_truth(raw)
    return build_ground_truth_dict(raw)


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
        "images": [pil_image],
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
    ds = ds.cast_column("images", Sequence(HFImage()))
    return ds