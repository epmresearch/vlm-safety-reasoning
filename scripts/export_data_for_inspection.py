import os
import sys

# Ensure imports work when run from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.constants import VALID_TASKS
from data.loader import load_processed_dataset
from data.preprocessor import build_sft_dataset
from data.dataset_cache import save_preprocessed_cache

def main():
    print("Loading raw HF datasets...")
    splits = load_processed_dataset()

    # Every registered task, from core/tasks.py — this used to be its own hardcoded
    # two-element list, a second place to forget when adding a pipeline.
    tasks = list(VALID_TASKS)
    split_names = ["train", "val", "test"]
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Exporting datasets for task: {task.upper()}")
        print(f"{'='*50}")
        
        for split_name in split_names:
            if split_name not in splits:
                print(f"Split '{split_name}' not found. Skipping.")
                continue
                
            dataset_split = splits[split_name]
            print(f"Processing {split_name} split ({len(dataset_split)} samples)...")
            
            # This applies the exact prompt templates and builds the targets
            samples = build_sft_dataset(dataset_split, task=task)
            
            # Extract image_ids from the raw dataset for cache join keys
            image_ids = [str(dataset_split[i]["image_id"]) for i in range(len(dataset_split))]
            
            # This dumps them to datasets/processed/{task}/{split_name}_cache.jsonl
            filename = f"{split_name}_cache.jsonl"
            path = save_preprocessed_cache(samples, filename=filename, task=task, image_ids=image_ids)
            
            print(f"SUCCESS: Saved {task}/{split_name} to {path}")

if __name__ == "__main__":
    main()
