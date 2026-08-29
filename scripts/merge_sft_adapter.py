import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_config
from models.model_loader import get_model_info, load_model_for_training

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge SFT adapter into base Qwen3-VL model")
    parser.add_argument("--tier", required=True, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--adapter_path", required=True, help="Path to your trained SFT adapter (e.g. results/vo-sft-2b-v4/final)")
    parser.add_argument("--output_path", required=True, help="Path to save the new merged base model")
    args = parser.parse_args()

    print(f">>> Loading Base Model for tier: {args.tier}")
    sft_cfg = load_config(task="violations_only", training_kind="sft")
    entry = get_model_info(args.tier)
    hf_path = entry["hf_path"]
    
    # We must load the model in 16-bit (not 4-bit) in order to safely merge it
    sft_cfg["load_in_4bit"] = False
    sft_cfg["load_in_8bit"] = False
    
    # Load model with the SFT adapter applied
    model, tokenizer, _ = load_model_for_training(
        model_name=hf_path,
        tier=args.tier,
        sft_cfg=sft_cfg,
        adapter_path=args.adapter_path
    )
    
    print(f">>> Merging adapter from {args.adapter_path} into base weights...")
    os.makedirs(args.output_path, exist_ok=True)
    
    # Unsloth makes merging very easy!
    model.save_pretrained_merged(args.output_path, tokenizer, save_method="merged_16bit")
    
    print(f">>> SUCCESS! Merged model saved to: {args.output_path}")
    print(f"You can now point your GRPO job to use this merged model as its base!")

if __name__ == "__main__":
    main()
