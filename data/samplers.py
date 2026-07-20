# Your processed dataset already has a resolution column (from your optimization notebook) and is pre-sorted ascending. This sampler groups consecutive (similarly-sized) samples into batches, then shuffles batch order each epoch — so you get the VRAM safety of bucketing plus real epoch-to-epoch randomization, without ever mixing a 240p image and a 4K image in the same batch.
"""
Resolution-aware batch sampler.

Prevents the padding-blowup OOM pattern (a single 4K image forcing every
other image in the batch to be padded to match it) by grouping samples of
similar resolution into the same batch, while still shuffling batch ORDER
each epoch so training isn't in a fixed easy->hard curriculum forever.
"""
from typing import Iterator, List, Optional, Sequence
import numpy as np
from torch.utils.data import Sampler

from core.logging import get_logger

logger = get_logger(__name__)


class ResolutionBucketSampler(Sampler[int]):
    """Groups dataset indices into fixed-size buckets sorted by resolution,
    then shuffles bucket order (not intra-bucket order) every epoch.

    Args:
        resolutions: Per-sample pixel counts (len == dataset length).
        batch_size: Per-device batch size (must match the DataLoader's batch_size
            for the bucketing to actually land samples in the same forward pass).
        shuffle: Whether to shuffle bucket order each epoch.
        seed: Base seed; combined with epoch for reproducible-but-varying shuffles.
        drop_last: If True, drop the final partial bucket.
    """

    def __init__(
        self,
        resolutions: Sequence[float],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        # Stable sort by resolution -> similar-sized images land together
        self.sorted_indices = np.argsort(np.asarray(resolutions), kind="stable")
        n_buckets = len(self.sorted_indices) // self.batch_size
        logger.info(
            f"ResolutionBucketSampler: {len(self.sorted_indices)} samples -> "
            f"{n_buckets} full buckets of size {self.batch_size} (shuffle={shuffle})"
        )

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch so bucket order varies but is reproducible."""
        self.epoch = epoch

    def _buckets(self) -> List[np.ndarray]:
        idx = self.sorted_indices
        n_full = (len(idx) // self.batch_size) * self.batch_size
        buckets = [idx[i:i + self.batch_size] for i in range(0, n_full, self.batch_size)]
        remainder = idx[n_full:]
        if len(remainder) and not self.drop_last:
            buckets.append(remainder)
        return buckets

    def __iter__(self) -> Iterator[int]:
        buckets = self._buckets()
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            order = rng.permutation(len(buckets))
            buckets = [buckets[i] for i in order]
        flat = [int(i) for bucket in buckets for i in bucket]
        return iter(flat)

    def __len__(self) -> int:
        return len(self.sorted_indices)


def get_resolutions(dataset) -> Optional[List[float]]:
    """Extracts the 'resolution' column if present; computes it on the fly otherwise."""
    if "resolution" in dataset.column_names:
        return list(dataset["resolution"])
    logger.warning("No 'resolution' column found — computing on the fly (slower).")
    try:
        return [img.width * img.height for img in dataset["image"]]
    except Exception as e:
        logger.warning(f"Could not compute resolutions: {e}. Bucketing disabled.")
        return None
    

class StratifiedRareClassSampler(Sampler[int]):
    """Full random shuffle every epoch (unlike ResolutionBucketSampler's
    static buckets), while guaranteeing 'rare' rows (rare_mask[i]=True,
    i.e. any Rule 2/3/4 violation) land at roughly evenly-spaced positions
    across the epoch instead of being left to chance.

    Use this INSTEAD OF ResolutionBucketSampler now that OOM safety is
    handled by image_min_pixels/image_max_pixels capping rather than
    resolution-sorted bucketing.
    """

    def __init__(
        self,
        rare_mask: Sequence[bool],
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.rare_mask = list(rare_mask)
        self.rare_indices = [i for i, r in enumerate(self.rare_mask) if r]
        self.common_indices = [i for i, r in enumerate(self.rare_mask) if not r]
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        logger.info(
            f"StratifiedRareClassSampler: {len(self.rare_mask)} samples -> "
            f"{len(self.rare_indices)} rare, {len(self.common_indices)} common"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        rare = list(self.rare_indices)
        common = list(self.common_indices)
        if self.shuffle:
            rng.shuffle(rare)
            rng.shuffle(common)

        n = len(rare) + len(common)
        if not rare:
            order = common
            if self.shuffle:
                perm = rng.permutation(len(order))
                order = [order[i] for i in perm]
            return iter(order)

        # Evenly-spaced (jittered) target slots for rare indices across [0, n)
        stride = n / len(rare)
        positions_set = set()
        
        for i in range(len(rare)):
            pos = min(int((i + rng.uniform(0.0, 1.0)) * stride), n - 1)
            while pos in positions_set:
                pos = (pos + 1) % n
            positions_set.add(pos)
            
        positions = sorted(list(positions_set))

        slots: List[Optional[int]] = [None] * n
        for pos, idx in zip(positions, rare):
            slots[pos] = idx
        common_iter = iter(common)
        for i in range(n):
            if slots[i] is None:
                slots[i] = next(common_iter)

        return iter(int(x) for x in slots)

    def __len__(self) -> int:
        return len(self.rare_mask)