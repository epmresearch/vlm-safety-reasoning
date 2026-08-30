"""
Isolates whether the "only row 0 gets real vision tokens" bug lives in the
Qwen3-VL AutoProcessor's batched __call__ itself, independent of TRL/Unsloth.

Calls processor(...) once with a single image (baseline), then once with N
copies of the SAME image/prompt (N = num_generations) — exactly what TRL's
_generate_and_score_completions sends when generation_batch_size == num_generations
(see trl/trainer/grpo_trainer.py: no dedup happens before tokenization).

No GPU, no model weights, no Unsloth import needed — just the processor.

Usage:
    python scripts/test_processor_batch_collapse.py \
        --raw_hf_path unsloth/Qwen3-VL-2B-Instruct \
        --image_path /path/to/any/real/construction_photo.jpg \
        --batch_size 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoProcessor
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_hf_path", default="unsloth/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image_path", required=True,
                         help="Any real image, e.g. a sample from datasets/processed")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="num_generations — how many copies of the SAME image/prompt to batch")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.raw_hf_path)
    img = Image.open(args.image_path).convert("RGB")

    text = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    )

    print(">>> Baseline: single-item call (batch=1)...")
    single = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)
    single_thw = single.get("image_grid_thw")
    print(f"  image_grid_thw (batch=1): {single_thw}")
    single_patches = single_thw[0].prod().item() if single_thw is not None else None
    print(f"  patch count for the one image: {single_patches}")

    print(f"\n>>> Test: batched call, {args.batch_size} copies of the SAME image/prompt "
          f"(mirrors GRPO's generation_batch_size == num_generations)...")
    batch = processor(
        text=[text] * args.batch_size,
        images=[[img]] * args.batch_size,
        return_tensors="pt",
        padding=True,
    )
    batch_thw = batch.get("image_grid_thw")
    print(f"  image_grid_thw (batch={args.batch_size}):\n{batch_thw}")

    if batch_thw is None or single_patches is None:
        print("\n⚠️  No image_grid_thw returned — inspect the processor output manually:")
        print(f"  keys: {list(batch.keys())}")
        return

    patches_per_row = batch_thw.prod(dim=-1) if batch_thw.dim() > 1 else batch_thw
    print(f"  patches per row: {patches_per_row.tolist()}")

    all_equal = bool((patches_per_row == patches_per_row[0]).all())
    matches_baseline = patches_per_row[0].item() == single_patches

    print("\n================ RESULT ================")
    if all_equal and matches_baseline:
        print("NO COLLAPSE: every row in the batch got the same, correct patch count "
              "as the single-item baseline. The processor itself is fine — the bug "
              "must be somewhere else (TRL/Unsloth-side, or something specific to "
              "how our real prompts differ from this synthetic one).")
    else:
        print("COLLAPSE CONFIRMED: rows in the batch have DIFFERENT patch counts than "
              "the single-item baseline (row 0 likely matches, rows 1+ don't).")
        print("This proves the bug is in the processor's batched image handling itself, "
              "NOT specific to 'different images' — a batch of N identical images still "
              "breaks. No GRPOConfig batch-size trick can avoid this, since generation_"
              "batch_size can never go below num_generations.")
    print("=========================================")


if __name__ == "__main__":
    main()
