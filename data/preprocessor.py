"""
Converts raw dataset samples into per-task chat-format training examples
(SFT) or prompt-only examples (GRPO). One function per task, dispatched
by task name.
"""
import json
from typing import Any, Dict, List, Optional

from data.schemas import SFTSample, GRPOPrompt, ChatMessage
from data.prompt_templates import SYSTEM_PROMPT_BASE, PROMPT_REGISTRY
from core.logging import get_logger

logger = get_logger(__name__)


def _get_prompt_text(task_cfg: Dict[str, Any]) -> str:
    return PROMPT_REGISTRY[task_cfg["prompt_key"]]


def _rule_violation_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    for rule_id in ["rule_1", "rule_2", "rule_3", "rule_4"]:
        v = raw.get(f"{rule_id}_violation")
        if v is not None:
            return {
                "rule_id": rule_id,
                "violated": True,
                "reasoning": v.get("reason", ""),
                "bounding_box": (v.get("bounding_box") or [None])[0],
            }
    return {"rule_id": "none", "violated": False, "reasoning": "", "bounding_box": None}


def _captioning_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"caption": raw.get("image_caption", "")}


def _grounding_ground_truth(raw: Dict[str, Any], classes: List[str]) -> Dict[str, Any]:
    # Returns one ground truth dict per class; caller picks/iterates as needed.
    return {cls: raw.get(cls, []) for cls in classes}


def _attributes_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "illumination": raw.get("illumination", ""),
        "camera_distance": raw.get("camera_distance", ""),
        "view": raw.get("view", ""),
        "quality_of_info": raw.get("quality_of_info", ""),
    }


GROUND_TRUTH_BUILDERS = {
    "rule_violation": _rule_violation_ground_truth,
    "captioning": _captioning_ground_truth,
    "attributes": _attributes_ground_truth,
    # "grounding" handled separately since it needs task_cfg["classes"]
}


def to_sft_format(raw_sample: Dict[str, Any], task: str, task_cfg: Dict[str, Any]) -> SFTSample:
    """
    raw_sample: one row from the HF dataset (dict-like).
    task: one of "rule_violation", "captioning", "grounding", "attributes".
    task_cfg: loaded from configs/tasks/<task>.yaml
    """
    prompt_text = _get_prompt_text(task_cfg)

    if task == "grounding":
        ground_truth = _grounding_ground_truth(raw_sample, task_cfg["classes"])
        assistant_content = json.dumps(ground_truth)
    else:
        builder = GROUND_TRUTH_BUILDERS.get(task)
        if builder is None:
            raise ValueError(f"Unknown task: {task}")
        ground_truth = builder(raw_sample)
        assistant_content = json.dumps(ground_truth)

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT_BASE),
        ChatMessage(role="user", content=prompt_text),
        ChatMessage(role="assistant", content=assistant_content),
    ]

    return SFTSample(image_id=raw_sample["image_id"], task=task, messages=messages)


def to_grpo_prompt(raw_sample: Dict[str, Any], task: str, task_cfg: Dict[str, Any]) -> GRPOPrompt:
    """Same as to_sft_format but without the assistant answer — used for GRPO rollouts."""
    prompt_text = _get_prompt_text(task_cfg)

    if task == "grounding":
        ground_truth = _grounding_ground_truth(raw_sample, task_cfg["classes"])
    else:
        builder = GROUND_TRUTH_BUILDERS.get(task)
        if builder is None:
            raise ValueError(f"Unknown task: {task}")
        ground_truth = builder(raw_sample)

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT_BASE),
        ChatMessage(role="user", content=prompt_text),
    ]

    return GRPOPrompt(
        image_id=raw_sample["image_id"],
        task=task,
        prompt_messages=messages,
        ground_truth=ground_truth,
    )


def build_sft_dataset(raw_dataset, task: str, task_cfg: Dict[str, Any]) -> List[SFTSample]:
    samples = []
    for raw in raw_dataset:
        try:
            samples.append(to_sft_format(raw, task, task_cfg))
        except Exception as e:
            logger.warning(f"Skipping sample {raw.get('image_id')} due to error: {e}")
    return samples