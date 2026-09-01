"""
Isolates whether the "only row 0 gets real vision tokens" bug lives in the
Qwen3-VL AutoProcessor's batched __call__ itself, independent of TRL/Unsloth.

Two modes:

  --num_unique_images 1 (default): N copies of the SAME image/prompt, where
  N = --batch_size. Mirrors GRPO's generation_batch_size == num_generations
  (i.e. steps_per_generation=1) — TRL's _generate_and_score_completions sends
  exactly this when every generation call repeats one prompt.

  --num_unique_images K (K > 1): K genuinely DIFFERENT real images from the
  dataset, batched together in one call, one copy each. This is what a
  generation call looks like when steps_per_generation > 1 (or
  per_device_train_batch_size > num_generations) — the scenario that's
  actually at risk, since the earlier synthetic 2-different-image test showed
  a real collapse ([795, 239] tokens) that mode 1 alone can't detect.

No GPU, no model weights, no Unsloth import needed — just the processor.

Usage:
    # Same-image-repeated check (steps_per_generation=1 scenario)
    python scripts/test_processor_batch_collapse.py --batch_size 8

    # Different-real-images check (steps_per_generation=2, 3, 4 scenario)
    python scripts/test_processor_batch_collapse.py --num_unique_images 2
    python scripts/test_processor_batch_collapse.py --num_unique_images 4

    (pulls real image(s) straight from your processed dataset via the same
    data.loader/data.preprocessor pipeline the real GRPO run uses — no need
    to point at a loose image file)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_hf_path", default="unsloth/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image_path", default=None,
                         help="Optional: path to a real image file to use instead of "
                              "pulling one from datasets/processed (only valid with "
                              "--num_unique_images 1)")
    from core.constants import VALID_TASKS
    parser.add_argument("--task", default="violations_only", choices=VALID_TASKS)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Only used when --num_unique_images=1: how many copies "
                              "of the SAME image/prompt to batch (mirrors num_generations)")
    parser.add_argument("--num_unique_images", type=int, default=1,
                         help="How many genuinely DIFFERENT real images to batch together, "
                              "one copy each. 1 = same-image-repeated mode (--batch_size "
                              "copies); >1 = different-images mode, testing the scenario "
                              "steps_per_generation > 1 actually creates.")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.raw_hf_path)

    text = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    )

    if args.num_unique_images > 1:
        _run_different_images_mode(processor, text, args)
    else:
        _run_repeated_image_mode(processor, text, args)


def _load_images(task: str, n: int):
    print(f">>> Pulling {n} real sample(s) from datasets/processed via the actual data pipeline...")
    from data.loader import load_processed_dataset
    from data.preprocessor import build_grpo_dataset_for_task
    raw_dataset = load_processed_dataset()
    train_split = raw_dataset["train"].select(range(n))
    train_data = build_grpo_dataset_for_task(train_split, task=task)
    imgs = [train_data[i]["image"] for i in range(n)]
    for i, img in enumerate(imgs):
        print(f"  Image {i}: type={type(img)}, size={getattr(img, 'size', 'N/A')}")
    return imgs


def _run_repeated_image_mode(processor, text, args):
    if args.image_path:
        from PIL import Image
        img = Image.open(args.image_path).convert("RGB")
    else:
        img = _load_images(args.task, 1)[0]

    print(">>> Baseline: single-item call (batch=1)...")
    single = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)
    single_thw = single.get("image_grid_thw")
    print(f"  image_grid_thw (batch=1): {single_thw}")
    single_patches = single_thw[0].prod().item() if single_thw is not None else None
    print(f"  patch count for the one image: {single_patches}")

    print(f"\n>>> Test: batched call, {args.batch_size} copies of the SAME image/prompt "
          f"(mirrors GRPO's generation_batch_size == num_generations, steps_per_generation=1)...")
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

    print("\n================ RESULT (same-image-repeated) ================")
    if all_equal and matches_baseline:
        print("NO COLLAPSE: every row in the batch got the same, correct patch count "
              "as the single-item baseline.")
    else:
        print("COLLAPSE CONFIRMED even with IDENTICAL images repeated — this would "
              "break steps_per_generation=1 too, not just >1.")
    print("================================================================")


def _run_different_images_mode(processor, text, args):
    n = args.num_unique_images
    imgs = _load_images(args.task, n)

    print(f"\n>>> Baseline: tokenizing each of the {n} images ALONE...")
    solo_patches = []
    for i, img in enumerate(imgs):
        solo = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)
        thw = solo.get("image_grid_thw")
        patches = thw[0].prod().item() if thw is not None else None
        solo_patches.append(patches)
        print(f"  Image {i} solo: patch count = {patches}")

    print(f"\n>>> Test: ONE batched call with all {n} DIFFERENT images together "
          f"(mirrors steps_per_generation={n}, generation_batch_size={n}x num_generations)...")
    batch = processor(
        text=[text] * n,
        images=[[img] for img in imgs],
        return_tensors="pt",
        padding=True,
    )
    batch_thw = batch.get("image_grid_thw")
    print(f"  image_grid_thw (batch of {n} different images):\n{batch_thw}")

    if batch_thw is None or any(p is None for p in solo_patches):
        print("\n⚠️  No image_grid_thw returned — inspect the processor output manually:")
        print(f"  keys: {list(batch.keys())}")
        return

    patches_per_row = batch_thw.prod(dim=-1) if batch_thw.dim() > 1 else batch_thw
    patches_per_row = patches_per_row.tolist()
    print(f"  patches per row (batched): {patches_per_row}")
    print(f"  patches per row (solo baseline): {solo_patches}")

    all_match = all(b == s for b, s in zip(patches_per_row, solo_patches))

    print(f"\n================ RESULT (num_unique_images={n}) ================")
    if all_match:
        print(f"NO COLLAPSE: all {n} different images kept their correct, solo-matching "
              f"patch count when batched together. steps_per_generation={n} is SAFE to use.")
    else:
        mismatches = [i for i, (b, s) in enumerate(zip(patches_per_row, solo_patches)) if b != s]
        print(f"COLLAPSE CONFIRMED: row(s) {mismatches} got a DIFFERENT patch count batched "
              f"than when processed alone (row 0 usually matches, later rows usually don't).")
        print(f"Do NOT use steps_per_generation={n} (or any config where "
              f"generation_batch_size/num_generations >= {n}) — those rows would train on "
              f"corrupted/missing vision tokens for this image.")
    print("==================================================================")


if __name__ == "__main__":
    main()
