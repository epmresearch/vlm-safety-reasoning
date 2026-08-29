import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset, Image as HFImage, Sequence
from transformers import AutoProcessor, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch
from PIL import Image

def main():
    print("Initializing dummy processor...")
    # Use a tiny vision model processor for testing
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Create a dummy blank image
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    print("\n--- Test 1: My Format (image inside the message, returned as List) ---")
    rows_my_format = [{
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image", "image": dummy_img},
                {"type": "text", "text": "user"}
            ]}
        ],
    }]
    
    config = GRPOConfig(output_dir="./tmp", remove_unused_columns=False)
    
    try:
        trainer1 = GRPOTrainer(
            model=None, 
            args=config, 
            train_dataset=rows_my_format,
            processing_class=processor,
            reward_funcs=[lambda **kwargs: [1.0]],
        )
        batch1 = next(iter(trainer1.get_train_dataloader()))
        print("My Format Output keys:", batch1.keys())
        if "pixel_values" in batch1:
            print("My Format SUCCESS: pixel_values generated.")
        else:
            print("My Format FAIL: no pixel_values.")
    except Exception as e:
        print("My Format crashed:", e)
        
    print("\n--- Test 2: AI's Format (image in separate column, returned as HF Dataset) ---")
    rows_ai_format = [{
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "user"}
            ]}
        ],
        "images": [dummy_img]
    }]
    ds_ai = Dataset.from_list(rows_ai_format)
    # The AI suggested casting, but let's see if it works without casting or with casting
    ds_ai = ds_ai.cast_column("images", Sequence(HFImage()))
    
    try:
        trainer2 = GRPOTrainer(
            model=None, 
            args=config, 
            train_dataset=ds_ai,
            processing_class=processor,
            reward_funcs=[lambda **kwargs: [1.0]],
        )
        batch2 = next(iter(trainer2.get_train_dataloader()))
        print("AI Format Output keys:", batch2.keys())
        if "pixel_values" in batch2:
            print("AI Format SUCCESS: pixel_values generated.")
        else:
            print("AI Format FAIL: no pixel_values.")
    except Exception as e:
        print("AI Format crashed:", e)

if __name__ == "__main__":
    main()
