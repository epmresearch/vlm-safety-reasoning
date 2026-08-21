"""
Config loading and merging.

Every experiment loads: base.yaml + model_registry.yaml + (sft.yaml or
grpo.yaml) + task yaml, merged into a single dict-like object.
Task config is applied last so task-specific overrides always win.
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

    Merge precedence (last wins):
        base.yaml → model_registry.yaml → {sft,grpo}.yaml → tasks/{task}.yaml

    Task config is applied LAST so that task-specific overrides (e.g.
    violations_only.yaml setting max_completion_length: 1024) always take
    priority over generic training defaults (e.g. grpo.yaml's 1000).

    Example:
        cfg = load_config(task="violations_only", training_kind="grpo")
        cfg["max_completion_length"]  # → 1024, from the task config
    """
    parts = [load_base_config(), load_model_strategy()]
    if training_kind:
        parts.append(load_training_config(training_kind))
    if task:
        parts.append(load_task_config(task))
    return merge_configs(*parts)