"""
Save/load the fully preprocessed multi-task SFT dataset to/from Drive as
JSONL — so we don't rebuild it every session, AND so small/medium/large all
train on the EXACT same data (required for a fair size comparison). This
file also becomes the versioned 'dataset package' deliverable.
"""
import json
from typing import List

from data.schemas import SFTSample, ChatMessage
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def save_sft_samples_jsonl(samples: List[SFTSample], filename: str) -> str:
    path = get_drive_path("datasets", "processed", filename)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.dict()) + "\n")
    logger.info(f"Saved {len(samples)} preprocessed samples to {path}")
    return str(path)


def load_sft_samples_jsonl(filename: str) -> List[SFTSample]:
    path = get_drive_path("datasets", "processed", filename)
    if not path.exists():
        raise FileNotFoundError(f"No cached dataset at {path} — build it first.")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            samples.append(SFTSample(
                image_id=d["image_id"], task=d["task"],
                messages=[ChatMessage(**m) for m in d["messages"]],
            ))
    logger.info(f"Loaded {len(samples)} preprocessed samples from {path}")
    return samples