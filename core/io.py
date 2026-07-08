"""
Filesystem helpers. Every save operation should call ensure_dir() first —
this is the "self-healing" layer described in the setup discussion, so a
missing Drive subfolder never crashes a training run mid-way.
"""
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union

from core.config import load_base_config

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_drive_path(*subpaths: str) -> Path:
    """
    Resolves a path under drive_root, e.g.:
        get_drive_path("checkpoints", "rule_violation", "sft_v1")
    """
    base_cfg = load_base_config()
    root = Path(base_cfg["drive_root"])
    return root.joinpath(*subpaths)


def safe_save_json(data: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_load_json(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_to_csv(row: Dict[str, Any], path: PathLike) -> None:
    """Thread-unsafe but process-safe-enough append for single-process eval loops.
    Writes header only if the file doesn't exist yet."""
    path = Path(path)
    ensure_dir(path.parent)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_rows_to_csv(rows: List[Dict[str, Any]], path: PathLike) -> None:
    """Overwrite-style save for a full results table (used by evaluator.py)."""
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)