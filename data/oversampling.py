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
from typing import Any, Dict, List, Optional, Tuple

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

# Rare-class definitions per capability. "Rare" means: an image containing at least one
# instance of a class the model would otherwise rarely see in a batch.
#
# For a batch of 32 and a class present in a fraction p of images, the chance a batch
# contains NONE of it is (1-p)^32 -- 80% at p=0.7%, 19% at p=5%, 1.7% at p=12%. Those
# starved steps contribute no gradient for that class, which is why violations needed
# both augmentation and this sampler: un-augmented rule_4 sits at 46/6308 = 0.7%.
_RARE_VIOLATION_RULES = (2, 3, 4)
_RARE_OBJECT_CLASSES = ("rebar", "worker_with_white_hard_hat")


def build_rare_mask_for_task(hf_dataset, task: str) -> Optional[List[bool]]:
    """Task-aware rare mask for StratifiedRareClassSampler.

    Returns None when the task has no meaningful notion of a rare target, in which case
    the caller should fall back to plain shuffling.

    Rarity is defined by capability, not by task name:

    * ``violations`` -> an image with any Rule 2/3/4 violation. Identical to the legacy
      ``build_rare_mask``, so ``unified`` and ``violations_only`` behaviour is unchanged.
    * ``objects`` (and not ``violations``) -> an image containing rebar or a worker in a
      white hard hat, the two hard classes. Excavator is deliberately excluded: it is the
      common, visually salient class (2415 train occurrences against 846 and 680), so
      marking it rare would mark most images rare and the stratification would degenerate
      to a plain shuffle.

    ``violations`` takes precedence for ``unified`` on purpose. Its violation components
    carry 0.55 of that task's reward weight against 0.25 for grounding, and stratifying on
    two different axes at once would over-constrain the ordering without a clear winner.
    """
    from core.tasks import CAP_OBJECTS, CAP_VIOLATIONS, task_has

    if task_has(task, CAP_VIOLATIONS):
        return [
            any(s.get(f"rule_{i}_violation") is not None for i in _RARE_VIOLATION_RULES)
            for s in hf_dataset
        ]

    if task_has(task, CAP_OBJECTS):
        return [
            any(s.get(c) for c in _RARE_OBJECT_CLASSES)
            for s in hf_dataset
        ]

    return None


def rare_class_incidence(hf_dataset, task: str) -> Dict[str, Any]:
    """Measures what fraction of images carry each rare target, and how often a batch of
    ``batch_size`` would contain none of it. Reporting only -- no training effect.

    This is the number the object_only stratification decision actually turns on, and it
    cannot be derived from the paper: Table 4 reports box *occurrences* (rebar 846), not
    the number of images containing at least one, and images may hold several boxes.
    """
    from core.tasks import CAP_OBJECTS, CAP_VIOLATIONS, task_has

    n = len(hf_dataset)
    counts: Dict[str, int] = {}
    if task_has(task, CAP_VIOLATIONS):
        for i in _RARE_VIOLATION_RULES:
            key = f"rule_{i}_violation"
            counts[key] = sum(1 for s in hf_dataset if s.get(key) is not None)
    if task_has(task, CAP_OBJECTS):
        for c in ("excavator",) + _RARE_OBJECT_CLASSES:
            counts[c] = sum(1 for s in hf_dataset if s.get(c))
    return {"n_images": n, "counts": counts}
