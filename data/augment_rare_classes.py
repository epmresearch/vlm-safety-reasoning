"""
Offline script to augment rare classes (Rule 3 and Rule 4) in the dataset.
Uses Albumentations to apply pixel-level transformations (brightness, contrast, noise, etc.).
By avoiding spatial transformations (flips, crops), we guarantee that all bounding boxes 
and text descriptions ("on the left", "on the right") remain perfectly accurate.
"""

import os
import argparse
import numpy as np
from datasets import load_from_disk, concatenate_datasets
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

def is_rare_class(sample: dict) -> bool:
    """Returns True if the sample contains Rule 3 or Rule 4 violations."""
    has_rule3 = sample.get("rule_3_violation") is not None
    has_rule4 = sample.get("rule_4_violation") is not None
    return has_rule3 or has_rule4

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_augmentations", type=int, default=5,
                        help="Number of augmented variations to generate per rare image.")
    args = parser.parse_args()
    
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
            
    logger.info(f"Found {len(rare_indices)} rare images (Rule 3 or Rule 4).")
    
    if len(rare_indices) == 0:
        logger.info("No rare images found. Exiting.")
        return
        
    # 2. Generate augmented variations
    transform = get_pixel_augmentation_pipeline()
    augmented_samples_dict = {col: [] for col in train_split.column_names}
    
    # Create a debug folder to save a few visual examples
    debug_dir = os.path.join(os.getcwd(), "debug_augmentations")
    ensure_dir(debug_dir)
    debug_samples_saved = 0
    
    logger.info("Generating dynamic augmentations to balance at ~500 images per rule...")
    for idx in tqdm(rare_indices, desc="Augmenting"):
        original_sample = train_split[idx]
        
        has_rule4 = original_sample.get("rule_4_violation") is not None
        has_rule3 = original_sample.get("rule_3_violation") is not None
        
        if has_rule4:
            num_augs = 10  # 46 * 11 = ~506
        elif has_rule3:
            num_augs = 4   # 109 * 5 = ~545
        else:
            num_augs = args.num_augmentations
            
        for aug_idx in range(1, num_augs + 1):
            aug_sample = augment_sample(original_sample, transform, aug_idx)
            for col in train_split.column_names:
                augmented_samples_dict[col].append(aug_sample[col])
                
            # Save the first 5 generated augmentations (and their originals) for visual inspection
            if debug_samples_saved < 5:
                orig_img = original_sample["image"]
                aug_img = aug_sample["image"]
                
                # Create a side-by-side comparison
                total_width = orig_img.width + aug_img.width
                max_height = max(orig_img.height, aug_img.height)
                
                comparison = Image.new('RGB', (total_width, max_height))
                comparison.paste(orig_img, (0, 0))
                comparison.paste(aug_img, (orig_img.width, 0))
                
                rule_name = "Rule4" if has_rule4 else "Rule3"
                save_path = os.path.join(debug_dir, f"debug_{rule_name}_{debug_samples_saved}.jpg")
                comparison.save(save_path)
                debug_samples_saved += 1
                
    if debug_samples_saved > 0:
        logger.info(f"Saved {debug_samples_saved} before/after comparison images to {debug_dir} for you to inspect!")
                
    # Create a new dataset from the augmented dict
    from datasets import Dataset
    aug_dataset = Dataset.from_dict(augmented_samples_dict)
    logger.info(f"Generated {len(aug_dataset)} new augmented samples.")
    
    # 3. Concatenate and shuffle
    combined_train = concatenate_datasets([train_split, aug_dataset])
    combined_train = combined_train.shuffle(seed=42)
    logger.info(f"New combined train split size: {len(combined_train)}")
    
    # 4. Save the new augmented dataset
    augmented_subdir = base_cfg["dataset"].get("augmented_subdir", "datasets/augmented")
    output_dir = get_drive_path(augmented_subdir)
    ensure_dir(output_dir)
    
    ds["train"] = combined_train
    logger.info(f"Saving fully augmented dataset to {output_dir}")
    ds.save_to_disk(str(output_dir))
    logger.info("Success! Run your data processing script again, but point it to this augmented folder.")

if __name__ == "__main__":
    main()
