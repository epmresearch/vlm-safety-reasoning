"""
Isolates which of two differences between our earlier successful single-image
test and TRL's real _tokenize_prompts call is causing images to collapse to a
single placeholder token:

    Test A: batch of 2 DIFFERENT images in one apply_chat_template call
            (default chat_template, same as our earlier successful test)
    Test B: single image, but with an explicit chat_template= argument
            (matching exactly how TRL's _tokenize_prompts calls it)

Usage:
    python scripts/test_batch_vs_explicit_template.py \
        --merged_path /home/$USER/vlm-finetuning-project1/checkpoints/qwen3vl-2b/merged-sft-2b-v4 \
        --raw_hf_path unsloth/Qwen3-VL-2B-Instruct
"""
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_path", required=True)
    parser.add_argument("--raw_hf_path", default="unsloth/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    import unsloth
    from unsloth import FastVisionModel

    print(">>> Loading model + processor...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.merged_path,
        tokenizer_name=args.raw_hf_path,
        load_in_4bit=True,
    )

    img1 = Image.new("RGB", (896, 896), color=(255, 0, 0))
    img2 = Image.new("RGB", (600, 400), color=(0, 255, 0))
    p1 = [{"role": "user", "content": [{"type": "image", "image": img1}, {"type": "text", "text": "describe"}]}]
    p2 = [{"role": "user", "content": [{"type": "image", "image": img2}, {"type": "text", "text": "describe"}]}]

    print("\n>>> Test A: batch of 2 different images, default chat_template")
    out_a = tokenizer.apply_chat_template(
        conversation=[p1, p2], tokenize=True, return_dict=True, add_generation_prompt=True,
    )
    lens_a = [len(x) for x in out_a["input_ids"]]
    print(f"A) lengths: {lens_a}")

    print("\n>>> Test B: single conversation, explicit chat_template= (matches TRL's call)")
    out_b = tokenizer.apply_chat_template(
        conversation=[p1], chat_template=tokenizer.chat_template,
        tokenize=True, return_dict=True, add_generation_prompt=True,
    )
    lens_b = [len(x) for x in out_b["input_ids"]]
    print(f"B) length: {lens_b}")

    print("\n================ VERDICT ================")
    if min(lens_a) < 100:
        print("A FAILS: batching multiple different images collapses tokens.")
    else:
        print("A PASSES: batching multiple different images is fine.")
    if min(lens_b) < 100:
        print("B FAILS: passing explicit chat_template= collapses tokens.")
    else:
        print("B PASSES: explicit chat_template= is fine.")
    print("===========================================")


if __name__ == "__main__":
    main()
