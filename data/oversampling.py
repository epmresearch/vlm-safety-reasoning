"""
Oversampling for rare safety-rule violations (Rule 2/3/4).

  - Rule 2 or Rule 4 present  -> 4 total copies (3 clones added)
  - Rule 3 present, AND NOT (Rule 2 or Rule 4) -> 2 total copies (1 clone added)
  - Everything else -> unchanged

Rule 2 and Rule 4 never co-occur (verified: 0 in the co-occurrence matrix),
so there is no double-counting risk in the "Rule 2 or Rule 4" bucket.
Rule 3 co-occurs with Rule 4 in exactly 1 image train-side; that image is
correctly routed to the Rule 2/4 bucket by the "AND NOT" clause below.

Uses HF Dataset.select() with repeated indices so images (and every other
field) are duplicated cheaply, without re-encoding or re-loading anything.
"""
from typing import Dict, List, Tuple

from core.logging import get_logger

logger = get_logger(__name__)


def build_oversampled_indices(
    hf_dataset,
    rule24_multiplier: int = 4,
    rule3_multiplier: int = 2,
) -> Tuple[List[int], Dict]:
    """Builds a list of dataset indices (with repeats) implementing the
    locked oversampling logic above.

    Args:
        hf_dataset: The (already stratified-split) train HF Dataset.
        rule24_multiplier: Total copies for images with Rule 2 or Rule 4.
        rule3_multiplier: Total copies for images with Rule 3 only.

    Returns:
        (indices, manifest). Pass `indices` directly to
        `hf_dataset.select(indices)`. `manifest` is a stats dict suitable
        for logging / saving as a run artifact.
    """
    indices: List[int] = []
    rule24_images = 0
    rule3_only_images = 0

    for i, sample in enumerate(hf_dataset):
        has_r2 = sample.get("rule_2_violation") is not None
        has_r4 = sample.get("rule_4_violation") is not None
        has_r3 = sample.get("rule_3_violation") is not None

        indices.append(i)

        if has_r2 or has_r4:
            indices.extend([i] * (rule24_multiplier - 1))
            rule24_images += 1
        elif has_r3:
            indices.extend([i] * (rule3_multiplier - 1))
            rule3_only_images += 1

    manifest = {
        "total_rows_before": len(hf_dataset),
        "total_rows_after": len(indices),
        "rule24_unique_images": rule24_images,
        "rule24_multiplier": rule24_multiplier,
        "rule24_added_rows": rule24_images * (rule24_multiplier - 1),
        "rule3_only_unique_images": rule3_only_images,
        "rule3_multiplier": rule3_multiplier,
        "rule3_added_rows": rule3_only_images * (rule3_multiplier - 1),
        "net_added_rows": len(indices) - len(hf_dataset),
    }
    logger.info(
        f"Oversampling: {len(hf_dataset)} -> {len(indices)} rows "
        f"(+{manifest['net_added_rows']}). Rule2/4 images={rule24_images} "
        f"(x{rule24_multiplier}), Rule3-only images={rule3_only_images} (x{rule3_multiplier})"
    )
    return indices, manifest


def build_rare_mask(hf_dataset) -> List[bool]:
    """Boolean mask, one entry per row of `hf_dataset` (call AFTER
    oversampling, on the already-duplicated dataset), marking a row as
    'rare' if it contains any Rule 2/3/4 violation. Consumed by
    StratifiedRareClassSampler to guarantee even spread across batches.
    """
    mask = []
    for sample in hf_dataset:
        is_rare = any(sample.get(f"rule_{i}_violation") is not None for i in (2, 3, 4))
        mask.append(is_rare)
    return mask