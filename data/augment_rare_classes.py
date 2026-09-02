"""
Offline script to augment rare-rule images (Rules 2, 3 and 4) in the dataset.
Uses Albumentations to apply pixel-level transformations (brightness, contrast, noise, etc.).
By avoiding spatial transformations (flips, crops), we guarantee that all bounding boxes 
and text descriptions ("on the left", "on the right") remain perfectly accurate.
"""

import json
import os
import argparse
import numpy as np
from datasets import load_from_disk, concatenate_datasets, Dataset
from PIL import Image
from tqdm import tqdm
from core.io import get_drive_path, ensure_dir
from core.config import load_base_config
from core.logging import get_logger

try:
    import albumentations as A
except ImportError:
    raise ImportError("Please install albumentations: pip install albumentations")

logger = get_logger(__name__)

# HPC login nodes often have tiny /tmp drives or strict quotas on ~/.cache
# We must force HuggingFace to use a safe scratch directory to prevent "Killed" disk-quota OOMs
if os.path.exists(os.path.expanduser("~/scratch")):
    safe_cache = os.path.expanduser("~/scratch/hf_datasets_cache")
    os.makedirs(safe_cache, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = safe_cache

# Define the robust pixel-level augmentation pipeline
def get_pixel_augmentation_pipeline():
    return A.Compose([
        # Always change lighting at least a little bit (p=1.0) so it's never identical
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        # Removed GaussNoise because it was destroying the images with heavy colored static
        # Make compression artifacts a bit stronger (quality 30-70)
        A.ImageCompression(quality_range=(30, 70), p=0.5),
        # Shift the gamma to simulate different camera sensors
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
    ])

# Extra copies generated per rare image, keyed by the rule that qualifies it. These
# are EXTRA copies: an image with num_augs = 16 ends up as 17 rows (original + 16).
# Values are derived from each rule's scarcity in the train split (rule_4 is the
# rarest at 46 images). Precedence is rule_4 > rule_2 > rule_3, so an image tripping
# several rules is counted once, under its rarest rule.
RULE_MULTIPLIERS = {4: 16, 2: 12, 3: 6}


def is_rare_class(sample: dict) -> bool:
    """Returns True if the sample contains Rule 2, 3, or 4 violations."""
    has_rule2 = sample.get("rule_2_violation") is not None
    has_rule3 = sample.get("rule_3_violation") is not None
    has_rule4 = sample.get("rule_4_violation") is not None
    return has_rule2 or has_rule3 or has_rule4

def augment_sample(sample: dict, transform: A.Compose, aug_index: int) -> dict:
    """Applies augmentation to a single dataset sample, returning a new dictionary."""
    # Deep copy the sample to avoid mutating the original
    new_sample = {k: v for k, v in sample.items()}
    
    # Modify the image ID to ensure uniqueness
    original_id = new_sample.get("image_id", "")
    new_sample["image_id"] = f"{original_id}_aug{aug_index}"
    
    # Extract the original PIL image and convert to NumPy array (RGB)
    pil_image = new_sample["image"]
    image_np = np.array(pil_image.convert("RGB"))
    
    # Apply pixel-level augmentation
    augmented = transform(image=image_np)
    aug_image_np = augmented["image"]
    
    # Convert back to PIL Image and store
    new_sample["image"] = Image.fromarray(aug_image_np)
    
    # Bounding boxes and text remain UNTOUCHED because spatial layout didn't change!
    return new_sample

def main():
    # No CLI arguments: per-rule multiplicity lives in the module-level
    # RULE_MULTIPLIERS constant. A --num_augmentations flag used to exist but was
    # never read, so it silently did nothing.
    argparse.ArgumentParser(
        description="Pixel-only augmentation of rare-rule images. Takes no arguments."
    ).parse_args()

    base_cfg = load_base_config()
    
    # Load the original processed dataset
    input_dir = get_drive_path("datasets", "processed")
    if not os.path.exists(str(input_dir)):
        # Fallback for local testing if VLM_DATA_ROOT isn't set perfectly
        input_dir = os.path.join(os.getcwd(), "data", "processed")
        if not os.path.exists(input_dir):
            input_dir = os.path.join(os.getcwd(), "datasets", "processed")
            
    logger.info(f"Loading original dataset from {input_dir}")
    ds = load_from_disk(str(input_dir))
    
    train_split = ds["train"]
    logger.info(f"Original train split size: {len(train_split)}")
    
    # 1. Filter for rare images
    rare_indices = []
    for i in tqdm(range(len(train_split)), desc="Scanning for rare classes"):
        if is_rare_class(train_split[i]):
            rare_indices.append(i)
            
    logger.info(f"Found {len(rare_indices)} rare images (Rule 2, 3 or 4).")
    
    if len(rare_indices) == 0:
        logger.info("No rare images found. Exiting.")
        return
        
    # 2. Generate augmented variations
    aug_pipeline = get_pixel_augmentation_pipeline()
    
    # Create a debug folder to save a few visual examples
    debug_dir = os.path.join(os.getcwd(), "debug_augmentations")
    ensure_dir(debug_dir)
    
    augmented_samples = []
    debug_samples_saved = 0
    
    for i in tqdm(rare_indices, desc="Augmenting"):
        sample = train_split[i]
        pil_img = sample["image"]
        has_r2 = sample.get("rule_2_violation") is not None
        has_r3 = sample.get("rule_3_violation") is not None
        has_r4 = sample.get("rule_4_violation") is not None
        
        if has_r4:
            num_augs = RULE_MULTIPLIERS[4]
            rule_name = "Rule4"
        elif has_r2:
            num_augs = RULE_MULTIPLIERS[2]
            rule_name = "Rule2"
        elif has_r3:
            num_augs = RULE_MULTIPLIERS[3]
            rule_name = "Rule3"
        else:
            continue
        
        for aug_idx in range(1, num_augs + 1):
            # Augment image using the existing augment_sample function
            new_sample = augment_sample(sample, aug_pipeline, aug_idx)
            aug_img = new_sample["image"]
            augmented_samples.append(new_sample)
            
            # Save first 5 examples for debugging
            if debug_samples_saved < 5:
                comparison = Image.new('RGB', (pil_img.width * 2, pil_img.height))
                comparison.paste(pil_img, (0, 0))
                comparison.paste(aug_img, (pil_img.width, 0))
                save_path = os.path.join(debug_dir, f"debug_{rule_name}_{debug_samples_saved}.jpg")
                comparison.save(save_path)
                debug_samples_saved += 1
                
    logger.info(f"Saved 5 before/after comparison images to {debug_dir} for you to inspect!")

    logger.info("Converting generated samples to HuggingFace Dataset...")
    
    augmented_ds = Dataset.from_list(augmented_samples)
    logger.info(f"Generated {len(augmented_ds)} new augmented samples.")
    
    # 3. Concatenate and shuffle
    combined_train = concatenate_datasets([train_split, augmented_ds])
    combined_train = combined_train.shuffle(seed=42)
    logger.info(f"New combined train split size: {len(combined_train)}")
    
    # 4. Save the new augmented dataset
    # processed_subdir is the augmented set — see the naming trap in CLAUDE.md
    # (base.yaml: processed_subdir -> datasets/augmented, raw_processed_subdir ->
    # datasets/processed). This previously read a nonexistent "augmented_subdir" key and
    # only worked because its hardcoded default happened to match.
    augmented_subdir = base_cfg["dataset"]["processed_subdir"]
    output_dir = get_drive_path(augmented_subdir)
    ensure_dir(output_dir)
    
    ds["train"] = combined_train
    logger.info(f"Saving fully augmented dataset to {output_dir}")
    # Use max_shard_size="100MB" to prevent memory spikes when saving Arrow files
    ds.save_to_disk(str(output_dir), max_shard_size="100MB")

    # Persist what was actually done. Until now this step logged its counts to stdout
    # and nothing else, so once a SLURM job's output scrolled away there was no record
    # of the multipliers used or the resulting per-rule balance -- the exact numbers the
    # augmentation decision and the SFT step count both depend on. build_grpo_pool.py
    # has always written a manifest; this brings augmentation in line with it.
    def _rule_counts(split):
        cols = [c for c in split.column_names if c != "image"]
        out = {}
        for r in (1, 2, 3, 4):
            key = f"rule_{r}_violation"
            out[key] = sum(1 for row in split.select_columns(cols)
                           if isinstance(row.get(key), dict))
        out["safe"] = sum(
            1 for row in split.select_columns(cols)
            if not any(isinstance(row.get(f"rule_{r}_violation"), dict) for r in (1, 2, 3, 4))
        )
        out["total"] = len(split)
        return out

    manifest = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": 42,
        "transforms": "pixel-only (brightness/contrast, JPEG compression, gamma); "
                      "no spatial transforms, so boxes and directional caption phrases "
                      "stay valid",
        "multipliers": {f"rule_{r}": m for r, m in RULE_MULTIPLIERS.items()},
        "train_rows_before": len(train_split),
        "rows_generated": len(augmented_ds),
        "train_rows_after": len(combined_train),
        "val_rows": len(ds["val"]) if "val" in ds else None,
        "test_rows": len(ds["test"]) if "test" in ds else None,
        "note": "val and test are carried over untouched, so all four tasks are still "
                "evaluated on byte-identical images.",
        "train_by_rule_before": _rule_counts(train_split),
        "train_by_rule_after": _rule_counts(combined_train),
    }
    manifest_path = output_dir / "augment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Augmentation manifest:\n{json.dumps(manifest, indent=2, default=str)}")
    logger.info(f"Saved augmentation manifest to {manifest_path}")
    logger.info("Success! Run your data processing script again, but point it to this augmented folder.")

if __name__ == "__main__":
    main()
