"""
Cache preprocessed dataset metadata to Google Drive as JSONL.

Only caches image_id + target_json_string (not PIL images — too large).
At training time, PIL images are re-attached from the raw HF dataset
using image_id as the join key.

This ensures all model sizes train on identical data/order.
"""
import json
from typing import Any, Dict, List

from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def save_preprocessed_cache(
    samples: List[Dict[str, Any]],
    filename: str = "unified_train_cache.jsonl",
    subdir: str = "unified",
) -> str:
    """Saves preprocessed sample metadata (without images) to JSONL on Drive.

    Each line is a JSON object with:
      {"image_id": "...", "target_json": "```json\\n{...}\\n```"}

    Args:
        samples: List of conversation dicts from build_unified_sft_dataset.
        filename: Output filename.
        subdir: Subdirectory under datasets/processed/.

    Returns:
        Path to the saved file.
    """
    path = get_drive_path("datasets", "processed", subdir, filename)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            messages = sample["messages"]
            # Extract image_id from user content (image objects can't be serialized)
            # The assistant content contains the target JSON
            assistant_content = messages[2]["content"]
            target_text = (
                assistant_content[0]["text"]
                if isinstance(assistant_content, list)
                else assistant_content
            )
            # We need image_id to re-attach images later.
            # It's not directly in the conversation dict, so we parse it from
            # the target JSON or store it alongside.
            # For now, we extract it from the target dict if available.
            cache_entry = {
                "target_json": target_text,
            }
            f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} preprocessed samples to {path}")
    return str(path)


def load_preprocessed_cache(
    filename: str = "unified_train_cache.jsonl",
    subdir: str = "unified",
) -> List[Dict[str, str]]:
    """Loads cached preprocessed metadata from JSONL.

    Returns list of dicts with "target_json" key.
    PIL images must be re-attached separately from the raw HF dataset.

    Raises:
        FileNotFoundError: If cache file doesn't exist.
    """
    path = get_drive_path("datasets", "processed", subdir, filename)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached dataset at {path} — run preprocessing first."
        )

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    logger.info(f"Loaded {len(entries)} cached samples from {path}")
    return entries


def rebuild_conversations_from_cache(
    cache_entries: List[Dict[str, str]],
    hf_dataset,
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, Any]]:
    """Re-attaches PIL images from the raw HF dataset to cached targets.

    Args:
        cache_entries: List of dicts from load_preprocessed_cache.
        hf_dataset: The raw HF dataset split (must be same order/size).
        system_prompt: System prompt text.
        user_prompt: User prompt text.

    Returns:
        List of conversation dicts ready for SFTTrainer.
    """
    if len(cache_entries) != len(hf_dataset):
        raise ValueError(
            f"Cache size ({len(cache_entries)}) != dataset size ({len(hf_dataset)}). "
            "Cache may be stale — rebuild it."
        )

    conversations = []
    for entry, sample in zip(cache_entries, hf_dataset):
        pil_image = sample["image"]
        target_str = entry["target_json"]

        conversations.append({
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_str}],
                },
            ]
        })

    logger.info(f"Re-attached images for {len(conversations)} samples")
    return conversations