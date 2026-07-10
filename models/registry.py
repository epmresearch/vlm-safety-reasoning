"""
Single source of truth: model_id -> {hf_path, task, size, lora_path}.
Built dynamically based on configs/model_strategy.yaml so the multi_size vs
task_specialized decision is a config change, not a code change.
"""
from typing import Any, Dict, Optional

from core.config import load_model_strategy
from core.io import get_drive_path
from core.logging import get_logger

logger = get_logger(__name__)


def build_registry() -> Dict[str, Dict[str, Any]]:
    strategy_cfg = load_model_strategy()
    strategy = strategy_cfg["strategy"]
    registry: Dict[str, Dict[str, Any]] = {}

    if strategy == "multi_size":
        variants = strategy_cfg["multi_size"]["variants"]
        for size_name, hf_path in variants.items():
            model_id = f"{size_name}-base"
            registry[model_id] = {
                "hf_path": hf_path,
                "task": None,          # multi_size models are evaluated across ALL tasks
                "size": size_name,
                "lora_path": None,
            }

    elif strategy == "task_specialized":
        per_task = strategy_cfg["task_specialized"]
        for task_name, hf_path in per_task.items():
            model_id = f"{task_name}-base"
            registry[model_id] = {
                "hf_path": hf_path,
                "task": task_name,
                "size": None,
                "lora_path": None,
            }
    else:
        raise ValueError(f"Unknown model strategy: {strategy}")

    return registry


REGISTRY = build_registry()


def get_model_entry(model_id: str) -> Dict[str, Any]:
    if model_id not in REGISTRY:
        raise KeyError(
            f"model_id '{model_id}' not found in registry. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[model_id]


def register_finetuned_variant(base_model_id: str, new_model_id: str, adapter_local_path: str) -> None:
    """
    Call this after SFT/GRPO training completes to register the new checkpoint,
    e.g. register_finetuned_variant("rule_violation-base", "rule_violation-sft-v1", "<path>")
    """
    base_entry = get_model_entry(base_model_id)
    REGISTRY[new_model_id] = {
        **base_entry,
        "lora_path": adapter_local_path,
    }
    logger.info(f"Registered new model variant: {new_model_id} -> {REGISTRY[new_model_id]}")


def checkpoint_path(task: str, variant: str) -> str:
    """e.g. checkpoint_path('rule_violation', 'sft_v1') -> Drive path"""
    return str(get_drive_path("checkpoints", task, variant))


def checkpoint_path_multitask(size_name: str, variant: str) -> str:
    """e.g. checkpoint_path_multitask('small', 'sft_v1') -> Drive path.
    Used instead of checkpoint_path() since multi-task SFT isn't tied to one task."""
    return str(get_drive_path("checkpoints", "multitask", size_name, variant))