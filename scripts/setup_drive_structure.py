"""
setup_drive_structure.py

Creates the Project's Google Drive skeleton (data/checkpoints/results/logs).
Run this ONCE, inside Colab, after mounting Drive. Safe to re-run, it never
overwrites existing files, only fills in missing folders/placeholders.

Usage:
    !python scripts/setup_drive_structure.py /content/drive/MyDrive/vlm-finetuning-project1
"""

import sys
from pathlib import Path

# Use short names instead of old task names for checkpointing
MODELS = ["qwen-2b", "qwen-4b", "qwen-8b"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def touch_if_missing(p: Path, content: str = "") -> None:
    if not p.exists():
        p.write_text(content, encoding="utf-8")
        print(f"  created: {p}")
    else:
        print(f"  skip (exists): {p}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_drive_structure.py <drive_root_path>")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser()
    print(f"Setting up Drive structure at: {root}\n")

    # --- secrets/ ---
    secrets_dir = root / "secrets"
    ensure_dir(secrets_dir)
    env_path = secrets_dir / ".env"
    if not env_path.exists():
        touch_if_missing(env_path, "HF_TOKEN=\nWANDB_API_KEY=\n")
    else:
        print(f"  skip (exists): {env_path}")

    # --- dataset cache/ ---
    ensure_dir(root / "datasets" / "unified_cache")

    # --- checkpoints/ ---
    for model in MODELS:
        ensure_dir(root / "checkpoints" / model)

    # --- results/ ---
    for model in MODELS:
        ensure_dir(root / "results" / model)
    ensure_dir(root / "results" / "figures")

    # --- logs/ ---
    logs_dir = root / "logs"
    ensure_dir(logs_dir)
    touch_if_missing(logs_dir / "colab_session_log.txt")

    print(f"\nDone. Drive structure ready at: {root}")


if __name__ == "__main__":
    main()