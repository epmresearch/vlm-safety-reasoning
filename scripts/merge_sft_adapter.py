import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_config
from models.model_loader import get_model_info, load_model_for_training

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge SFT adapter into base Qwen3-VL model")
    parser.add_argument("--tier", required=True, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--adapter_path", required=True, help="Path to your trained SFT adapter (e.g. results/vo-sft-2b-vN/final)")
    parser.add_argument("--output_path", required=True, help="Path to save the new merged base model")
    parser.add_argument(
        "--task", default="violations_only",
        help="Task the SFT adapter was trained for (e.g. 'unified', 'violations_only'). "
             "Determines which task-specific config overrides (e.g. max_seq_length) are "
             "applied when loading the base model to merge into. Default kept as "
             "'violations_only' for backward compatibility with existing callers."
    )
    args = parser.parse_args()

    print(f">>> Loading Base Model for tier: {args.tier}, task: {args.task}")
    sft_cfg = load_config(task=args.task, training_kind="sft")
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
    
    # Unsloth's save_pretrained_merged sometimes fails to crush weights on vision models.
    # We use standard PEFT merge_and_unload() to guarantee the LoRA weights are 
    # physically added to the base matrices, removing the PEFT wrappers completely.
    print("Crushing LoRA weights into base model (merge_and_unload)...")
    model = model.merge_and_unload()
    
    print("Saving the crushed model...")
    # Now save it as a standard HuggingFace model
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Unsloth/Qwen3-VL quirk: tokenizer.save_pretrained() doesn't always write
    # preprocessor_config.json, which leaves the reloaded processor with no
    # image_processor (apply_pixel_bounds then has nothing to cap, and Unsloth's
    # own VLM processor fallback can return None). Same backfill SFT checkpoints
    # already use via SaveBestModelCallback.
    from core.callbacks import _ensure_preprocessor_config
    _ensure_preprocessor_config(args.output_path, hf_path)

    print(f">>> SUCCESS! Merged model saved to: {args.output_path}")
    print(f"You can now point your GRPO job to use this merged model as its base!")

if __name__ == "__main__":
    main()
