"""
Isolated test: does the same "images collapse to 1 token in batched
apply_chat_template" bug happen with ZERO Unsloth involvement?

Loads the merged model via plain transformers + peft (no unsloth import
anywhere), builds a vanilla trl.GRPOTrainer, and runs the exact same
_tokenize_prompts check we ran through Unsloth's patched trainer.

Usage:
    python scripts/test_no_unsloth_tokenize.py \
        --merged_path /home/$USER/vlm-finetuning-project1/checkpoints/qwen3vl-2b/merged-sft-2b-v4 \
        --raw_hf_path unsloth/Qwen3-VL-2B-Instruct
"""
import argparse
from PIL import Image

# Deliberately no `import unsloth` anywhere in this file or its imports.
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_path", required=True)
    parser.add_argument("--raw_hf_path", default="unsloth/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    print(">>> 1. Loading processor from the RAW HF repo (same tokenizer_name principle)...")
    processor = AutoProcessor.from_pretrained(args.raw_hf_path)

    print(">>> 2. Loading model weights from the MERGED local checkpoint (bf16, no quantization)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(">>> 3. Applying a fresh LoRA adapter via plain peft (language layers only)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(">>> 4. Building a vanilla trl.GRPOTrainer (no Unsloth patching involved)...")

    def mock_reward(prompts, completions, **kwargs):
        return [1.0] * len(prompts)

    grpo_config = GRPOConfig(
        output_dir="./tmp_no_unsloth_test",
        per_device_train_batch_size=1,
        num_generations=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # Minimal 2-row dataset so the trainer object exists; we only care about
    # calling _tokenize_prompts ourselves below, not about actually training.
    from datasets import Dataset
    dummy_ds = Dataset.from_list([
        {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]}],
         "images": [Image.new("RGB", (896, 896), color=(255, 0, 0))]},
        {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]}],
         "images": [Image.new("RGB", (600, 400), color=(0, 255, 0))]},
    ])
    from datasets import Image as HFImage, Sequence
    dummy_ds = dummy_ds.cast_column("images", Sequence(HFImage()))

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dummy_ds,
        reward_funcs=[mock_reward],
        processing_class=processor,
    )

    print(">>> 5. Calling the EXACT same _tokenize_prompts TRL uses internally, with 2 DIFFERENT images...")
    prompts = [dummy_ds[0]["prompt"], dummy_ds[1]["prompt"]]
    # Embed the real PIL images inline the same way TRL's prepare_multimodal_messages does
    prompts[0][0]["content"][0]["image"] = dummy_ds[0]["images"][0]
    prompts[1][0]["content"][0]["image"] = dummy_ds[1]["images"][0]

    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)
    lens = [len(p) for p in prompt_ids]
    print(f"prompt_ids lengths (per conversation): {lens}")
    print(f"multimodal_fields keys: {list(multimodal_fields.keys())}")
    if "pixel_values" in multimodal_fields:
        pv = multimodal_fields["pixel_values"]
        print(f"pixel_values shape/len: {pv.shape if hasattr(pv, 'shape') else len(pv)}")

    if min(lens) > 100:
        print("\n✅✅ NO UNSLOTH: images correctly expanded in the batched, multi-image call!")
        print("   => The bug is specific to Unsloth's patched trainer, not TRL/transformers itself.")
    else:
        print("\n❌❌ NO UNSLOTH: still collapsed to a tiny prompt length!")
        print("   => The bug lives in vanilla TRL/transformers, not something Unsloth introduced.")


if __name__ == "__main__":
    main()
