"""
Converts raw dataset samples into per-task chat-format training examples
(SFT) and prompt-only examples (kept for future GRPO). Also builds the
combined multi-task dataset used for the multi-size SFT strategy: ONE model
per size, trained on ALL tasks together — not one model per task.

Project 1 tasks: rule_violation, captioning, grounding.
(attributes/metadata task is explicitly excluded from Project 1 scope.)
"""
import json
import random
from typing import Any, Dict, List

from data.schemas import SFTSample, GRPOPrompt, ChatMessage
from data.prompt_templates import SYSTEM_PROMPT_BASE, PROMPT_REGISTRY, get_grounding_prompt
from data.box_utils import clean_boxes
from core.logging import get_logger

logger = get_logger(__name__)

GROUNDING_CLASSES = ["excavator", "rebar", "worker_with_white_hard_hat"]


def _get_prompt_text(task_cfg: Dict[str, Any]) -> str:
    return PROMPT_REGISTRY[task_cfg["prompt_key"]]


# ---------------------------------------------------------------------------
# Ground-truth builders
# ---------------------------------------------------------------------------

def _rule_violation_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-label: collects EVERY violated rule present in the image (fixes the
    earlier single-label bug that silently dropped co-occurring violations).
    Degenerate boxes are filtered; a violation with no valid box left keeps
    its reasoning but reports bounding_box=None (matches the earlier
    0000167 zero-box edge case — reasoning kept, grounding excluded).
    """
    violations = []
    for i in range(1, 5):
        v = raw.get(f"rule_{i}_violation")
        if v is None:
            continue
        boxes = clean_boxes(v.get("bounding_box"))
        violations.append({
            "rule_id": f"rule_{i}",
            "reasoning": v.get("reason", "") or "",
            "bounding_box": list(boxes[0]) if boxes else None,
            "all_bounding_boxes": [list(b) for b in boxes],  # internal use only
        })
    return {"violations": violations}


def _captioning_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"caption": raw.get("image_caption", "")}


def _grounding_ground_truth_single_class(raw: Dict[str, Any], class_name: str) -> Dict[str, Any]:
    boxes = clean_boxes(raw.get(class_name, []))
    return {"class_name": class_name, "bounding_boxes": [list(b) for b in boxes]}


GROUND_TRUTH_BUILDERS = {
    "rule_violation": _rule_violation_ground_truth,
    "captioning": _captioning_ground_truth,
    # "grounding" handled separately per-class — see build_grounding_sft_samples()
}


# ---------------------------------------------------------------------------
# SFT sample construction — rule_violation / captioning
# ---------------------------------------------------------------------------

def to_sft_format(raw_sample: Dict[str, Any], task: str, task_cfg: Dict[str, Any]) -> SFTSample:
    prompt_text = _get_prompt_text(task_cfg)
    ground_truth = GROUND_TRUTH_BUILDERS[task](raw_sample)

    if task == "rule_violation":
        # Strip internal-only "all_bounding_boxes" before training — the
        # model should only ever see/produce the schema-defined fields.
        assistant_payload = {
            "violations": [
                {"rule_id": v["rule_id"], "reasoning": v["reasoning"], "bounding_box": v["bounding_box"]}
                for v in ground_truth["violations"]
            ]
        }
    else:
        assistant_payload = ground_truth

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT_BASE),
        ChatMessage(role="user", content=prompt_text),
        ChatMessage(role="assistant", content=json.dumps(assistant_payload)),
    ]
    return SFTSample(image_id=raw_sample["image_id"], task=task, messages=messages)


def build_sft_dataset(raw_dataset, task: str, task_cfg: Dict[str, Any]) -> List[SFTSample]:
    samples = []
    for raw in raw_dataset:
        try:
            samples.append(to_sft_format(raw, task, task_cfg))
        except Exception as e:
            logger.warning(f"Skipping sample {raw.get('image_id')} (task={task}) due to error: {e}")
    return samples


# ---------------------------------------------------------------------------
# Grounding — one sample PER (image, class), matching the paper's methodology
# ---------------------------------------------------------------------------

def build_grounding_sft_samples(raw_dataset, classes: List[str] = None) -> List[SFTSample]:
    classes = classes or GROUNDING_CLASSES
    samples = []
    for raw in raw_dataset:
        for cls in classes:
            try:
                gt = _grounding_ground_truth_single_class(raw, cls)
                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT_BASE),
                    ChatMessage(role="user", content=get_grounding_prompt(cls)),
                    ChatMessage(role="assistant", content=json.dumps(gt)),
                ]
                samples.append(SFTSample(image_id=raw["image_id"], task="grounding", messages=messages))
            except Exception as e:
                logger.warning(f"Skipping grounding sample {raw.get('image_id')} class={cls}: {e}")
    return samples


# ---------------------------------------------------------------------------
# Combined multi-task dataset (Project 1 decision: ONE model per size,
# trained across ALL tasks together — not one model per task)
# ---------------------------------------------------------------------------

def build_multitask_sft_dataset(raw_dataset, task_cfgs: Dict[str, Dict[str, Any]], seed: int = 42) -> List[SFTSample]:
    """task_cfgs: {"rule_violation": <cfg>, "captioning": <cfg>, "grounding": <cfg>}"""
    all_samples: List[SFTSample] = []

    if "rule_violation" in task_cfgs:
        all_samples += build_sft_dataset(raw_dataset, "rule_violation", task_cfgs["rule_violation"])
    if "captioning" in task_cfgs:
        all_samples += build_sft_dataset(raw_dataset, "captioning", task_cfgs["captioning"])
    if "grounding" in task_cfgs:
        classes = task_cfgs["grounding"].get("classes", GROUNDING_CLASSES)
        all_samples += build_grounding_sft_samples(raw_dataset, classes)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    logger.info(f"Built multi-task SFT dataset: {len(all_samples)} total samples "
                f"across tasks={list(task_cfgs.keys())}")
    return all_samples


# ---------------------------------------------------------------------------
# GRPO prompt builder (kept for later — not used while GRPO is paused)
# ---------------------------------------------------------------------------

def to_grpo_prompt(raw_sample: Dict[str, Any], task: str, task_cfg: Dict[str, Any]) -> GRPOPrompt:
    prompt_text = _get_prompt_text(task_cfg)
    ground_truth = GROUND_TRUTH_BUILDERS[task](raw_sample)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT_BASE),
        ChatMessage(role="user", content=prompt_text),
    ]
    return GRPOPrompt(image_id=raw_sample["image_id"], task=task, prompt_messages=messages, ground_truth=ground_truth)