"""
Shared helper so baseline and SFT evaluation always use the IDENTICAL test
subsample when max_samples is set — required for a valid Base vs SFT comparison.
"""
from typing import Optional

def get_eval_split(dataset_split, max_samples: Optional[int] = None, seed: int = 42):
    if max_samples is None or max_samples >= len(dataset_split):
        return dataset_split
    return dataset_split.shuffle(seed=seed).select(range(max_samples))