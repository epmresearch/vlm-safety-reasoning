"""
Config loading and merging.

Every experiment loads: base.yaml + a task yaml + (sft.yaml or grpo.yaml) +
model_strategy.yaml, merged into a single dict-like object.
"""
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_base_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_ROOT / "base.yaml")


def load_model_strategy() -> Dict[str, Any]:
    return _load_yaml(CONFIG_ROOT / "model_registry.yaml")


def load_task_config(task: str) -> Dict[str, Any]:
    path = CONFIG_ROOT / "tasks" / f"{task}.yaml"
    return _load_yaml(path)


def load_training_config(kind: str) -> Dict[str, Any]:
    """kind: 'sft' or 'grpo'"""
    if kind not in ("sft", "grpo"):
        raise ValueError(f"Unknown training config kind: {kind}")
    return _load_yaml(CONFIG_ROOT / f"{kind}.yaml")


def merge_configs(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge; later dicts override earlier ones except for nested dicts,
    which are merged one level deep."""
    merged: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    return merged


def load_config(task: str = None, training_kind: str = None) -> Dict[str, Any]:
    """
    Convenience loader used by experiments/*.py.

    Example:
        cfg = load_config(task="rule_violation", training_kind="sft")
        cfg["drive_root"], cfg["reward_weights"], cfg["lr"], etc. are all present.
    """
    parts = [load_base_config(), load_model_strategy()]
    if task:
        parts.append(load_task_config(task))
    if training_kind:
        parts.append(load_training_config(training_kind))
    return merge_configs(*parts)