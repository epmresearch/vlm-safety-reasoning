"""
Cache preprocessed dataset metadata to Google Drive as JSONL.

Only caches image_id + target_json_string (not PIL images — too large).
At training time, PIL images are re-attached from the raw HF dataset
using image_id as the join key.

This ensures all model sizes train on identical data/order.
"""
import json
from typing import Any, Dict, List, Optional

from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def save_preprocessed_cache(
    samples: List[Dict[str, Any]],
    filename: str = "unified_train_cache.jsonl",
    task: str = "unified",
    image_ids: Optional[List[str]] = None,
) -> str:
    """Saves preprocessed sample metadata (without images) to JSONL on Drive.

    Each line is a JSON object with:
      {"image_id": "...", "target_json": "```json\\n{...}\\n```"}

    Args:
        samples: List of conversation dicts from build_unified_sft_dataset.
        filename: Output filename.
        task: Task subdirectory under datasets/processed/.
        image_ids: Parallel list of image_id strings (same length as samples).
                   Required for robust image re-attachment on reload.

    Returns:
        Path to the saved file.
    """
    if image_ids is not None and len(image_ids) != len(samples):
        raise ValueError(
            f"image_ids length ({len(image_ids)}) != samples length ({len(samples)})"
        )

    path = get_drive_path("datasets", "processed", task, filename)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            messages = sample["messages"]
            # Extract image_id from user content (image objects can't be serialized)
            # The assistant content contains the target JSON
            assistant_content = messages[2]["content"]
            target_text = (
                assistant_content[0]["text"]
                if isinstance(assistant_content, list)
                else assistant_content
            )
            cache_entry = {
                "image_id": image_ids[idx] if image_ids is not None else f"unknown_{idx}",
                "target_json": target_text,
            }
            f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} preprocessed samples to {path}")
    return str(path)


def load_preprocessed_cache(
    filename: str = "unified_train_cache.jsonl",
    task: str = "unified",
) -> List[Dict[str, str]]:
    """Loads cached preprocessed metadata from JSONL.

    Returns list of dicts with "image_id" and "target_json" keys.
    PIL images must be re-attached separately from the raw HF dataset.

    Raises:
        FileNotFoundError: If cache file doesn't exist.
    """
    path = get_drive_path("datasets", "processed", task, filename)
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

    Uses image_id as a join key to match cached targets to their images,
    making the rebuild robust to dataset ordering changes.

    Args:
        cache_entries: List of dicts from load_preprocessed_cache
                       (each must have "image_id" and "target_json" keys).
        hf_dataset: The raw HF dataset split (must contain all image_ids).
        system_prompt: System prompt text.
        user_prompt: User prompt text.

    Returns:
        List of conversation dicts ready for SFTTrainer.
    """
    # Build image_id → PIL image lookup from the HF dataset
    image_map = {str(sample["image_id"]): sample["image"] for sample in hf_dataset}

    missing_ids = []
    conversations = []
    for entry in cache_entries:
        image_id = entry.get("image_id")
        target_str = entry["target_json"]

        if image_id is None or str(image_id) not in image_map:
            missing_ids.append(image_id)
            continue

        pil_image = image_map[str(image_id)]

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

    if missing_ids:
        logger.warning(
            f"{len(missing_ids)} cached entries had no matching image_id in the "
            f"HF dataset — these samples were skipped. First few: {missing_ids[:5]}"
        )

    logger.info(f"Re-attached images for {len(conversations)} samples")
    return conversations