import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from data.loader import load_processed_dataset
from data.preprocessor import to_grpo_prompt_for_task, raw_sample_to_conversation_for_task
from models.model_loader import get_model_info, load_model_for_training

def main():
    import argparse
    from core.constants import VALID_TASKS
    parser = argparse.ArgumentParser(
        description="Verify that images actually reach the model in the GRPO data path "
                    "(invariant #1 in CLAUDE.md: the image column must be named 'image')."
    )
    parser.add_argument("--task", default="violations_only", choices=VALID_TASKS,
                        help="Task whose prompt/target formatting to inspect.")
    args = parser.parse_args()
    task = args.task

    print(f"Loading a single sample from the dataset (task={task})...")
    # This is exactly how SFT and GRPO load data
    raw_dataset = load_processed_dataset()
    sample = raw_dataset["train"][0]
    pil_image = sample["image"]
    
    # 1. Look at the SFT format
    print("\n--- SFT Format ---")
    sft_conv = raw_sample_to_conversation_for_task(sample, pil_image, task=task)
    print("User message content:")
    for item in sft_conv["messages"][1]["content"]:
        if item["type"] == "image":
            print(f"  - Image: {type(item.get('image'))}")
        else:
            print(f"  - Text: {len(item.get('text', ''))} chars")
            
    # 2. Look at the GRPO format
    print("\n--- GRPO Format ---")
    grpo_prompt = to_grpo_prompt_for_task(sample, pil_image, task=task)
    print("User message content:")
    for item in grpo_prompt["prompt"][1]["content"]:
        if item["type"] == "image":
            print(f"  - Image: {type(item.get('image'))} (Notice anything missing?)")
        else:
            print(f"  - Text: {len(item.get('text', ''))} chars")

    print("\nLoading processor to test tokenization...")
    entry = get_model_info("2b")
    hf_path = entry["hf_path"]
    
    # Load just the processor/tokenizer
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_path)
    
    # Apply chat template for GRPO
    try:
        # TRL GRPO usually applies chat template to the prompt
        grpo_text = processor.apply_chat_template(grpo_prompt["prompt"], tokenize=False, add_generation_prompt=True)
        # Process it (this simulates what happens inside the trainer)
        grpo_inputs = processor(text=[grpo_text], padding=True, return_tensors="pt")
        print(f"\nGRPO input token length: {grpo_inputs.input_ids.shape[1]}")
    except Exception as e:
        print(f"Error processing GRPO prompt: {e}")

    # Now let's fix the GRPO prompt manually and see the difference
    print("\n--- Fixing the GRPO format in memory ---")
    grpo_prompt["prompt"][1]["content"][0]["image"] = pil_image
    try:
        from qwen_vl_utils import process_vision_info
        fixed_text = processor.apply_chat_template(grpo_prompt["prompt"], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(grpo_prompt["prompt"])
        fixed_inputs = processor(text=[fixed_text], images=image_inputs, padding=True, return_tensors="pt")
        print(f"FIXED GRPO input token length: {fixed_inputs.input_ids.shape[1]}")
        print(f"Number of image tokens added: {fixed_inputs.input_ids.shape[1] - grpo_inputs.input_ids.shape[1]}")
    except Exception as e:
        print(f"Error processing fixed GRPO prompt: {e}")

if __name__ == "__main__":
    main()
