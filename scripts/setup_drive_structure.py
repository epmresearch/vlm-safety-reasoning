"""
setup_drive_structure.py

Creates the Project 1 Google Drive skeleton (data/checkpoints/results/logs).
Run this ONCE, inside Colab, after mounting Drive. Safe to re-run — it never
overwrites existing files, only fills in missing folders/placeholders.

Usage (inside a Colab cell):

    from google.colab import drive
    drive.mount('/content/gdrive')

    !python setup_drive_structure.py /content/gdrive/MyDrive/vlm-finetuning-project1

Or from a notebook, after uploading this file to the repo's scripts/ folder:

    !python scripts/setup_drive_structure.py /content/gdrive/MyDrive/vlm-finetuning-project1
"""

import sys
import json
from pathlib import Path

TASKS = ["rule_violation", "captioning", "grounding", "attributes"]


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
        print("Example: python setup_drive_structure.py /content/gdrive/MyDrive/vlm-finetuning-project1")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser()
    print(f"Setting up Drive structure at: {root}\n")

    # --- secrets/ ---
    secrets_dir = root / "secrets"
    ensure_dir(secrets_dir)
    env_path = secrets_dir / ".env"
    if not env_path.exists():
        touch_if_missing(env_path, "HF_TOKEN=\nWANDB_API_KEY=\n")
        print("  >>> IMPORTANT: open secrets/.env in Drive and paste your REAL tokens in manually.")
    else:
        print(f"  skip (exists, not overwriting real secrets): {env_path}")

    # --- datasets/ ---
    ensure_dir(root / "datasets" / "raw")
    for task in TASKS:
        ensure_dir(root / "datasets" / "processed" / task)
    touch_if_missing(
        root / "datasets" / "split_manifest.json",
        json.dumps({"note": "Uses HF native 7009/3004 split, unmodified."}, indent=2),
    )

    # --- checkpoints/ ---
    for task in TASKS:
        ensure_dir(root / "checkpoints" / task)

    # --- results/ ---
    for task in TASKS:
        ensure_dir(root / "results" / task)
    ensure_dir(root / "results" / "error_analyses")

    # --- figures/ ---
    ensure_dir(root / "figures")

    # --- logs/ ---
    logs_dir = root / "logs"
    ensure_dir(logs_dir)
    touch_if_missing(logs_dir / "colab_session_log.txt")
    touch_if_missing(logs_dir / "error_log.txt")

    print(f"\nDone. Drive structure ready at: {root}")
    print("Next: paste real HF_TOKEN / WANDB_API_KEY into secrets/.env manually (never via script).")


if __name__ == "__main__":
    main()