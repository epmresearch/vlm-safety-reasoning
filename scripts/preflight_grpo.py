import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_config, load_task_config
from models.model_loader import get_model_info, load_model_for_training
from data.loader import load_processed_dataset
from data.preprocessor import build_grpo_dataset_for_task

def run_preflight(task="violations_only", model_id="2b", base_model_override=None):
    print(">>> 1. Loading configs...")
    cfg = load_config(task=task, training_kind="grpo")
    sft_cfg = load_config(task=task, training_kind="sft")
    task_cfg = load_task_config(task)
    entry = get_model_info(model_id)
    hf_path = base_model_override or entry["hf_path"]
    if base_model_override:
        print(f">>> Using MERGED base model override: {base_model_override}")
        print(f">>> Tokenizer will be loaded from raw HF repo: {entry['hf_path']}")

    print(">>> 2. Loading model (this may take a minute)...")
    from unsloth import FastVisionModel, PatchFastRL
    PatchFastRL("GRPO", FastVisionModel)

    # We load the model exactly like run_grpo() does — including tokenizer_name,
    # since a local merged checkpoint needs the tokenizer/processor loaded from
    # the original HF repo (see models/model_loader.py for why).
    model, tokenizer, _ = load_model_for_training(
        model_name=hf_path,
        tier=model_id,
        sft_cfg=sft_cfg,
        tokenizer_name=entry["hf_path"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(">>> 3. Loading a tiny dataset slice...")
    raw_dataset = load_processed_dataset()
    # Just take 4 samples for a quick check
    train_split = raw_dataset["train"].select(range(4))
    train_data = build_grpo_dataset_for_task(train_split, task=task)
    
    print(">>> 4. Initializing GRPOTrainer (No training will happen)...")
    from trl import GRPOTrainer, GRPOConfig

    class _PerImageGRPOTrainer(GRPOTrainer):
        """Same fix as models/grpo_trainer.py — see that file for the full
        explanation. TRL's batched apply_chat_template call only expands the
        FIRST image when given multiple different images at once; this
        tokenizes each conversation individually instead."""

        def _tokenize_prompts(self, prompts: list):
            # Deliberately does NOT reimplement the original method's
            # internals (self.tools, self.chat_template, prepare_multimodal_
            # messages, etc.) — those differ across trl/Unsloth versions and
            # kept breaking when hand-copied. Instead, call the ORIGINAL,
            # inherited _tokenize_prompts once per conversation (a batch of
            # exactly 1), which we've repeatedly confirmed produces correctly
            # expanded image tokens. Only multi-conversation batches collapse.
            import torch as _torch

            all_prompt_ids = []
            all_images = []
            merged_fields = {}
            for prompt in prompts:
                ids_list, imgs_list, fields = super(_PerImageGRPOTrainer, self)._tokenize_prompts([prompt])
                all_prompt_ids.append(ids_list[0])
                all_images.append(imgs_list[0] if imgs_list else None)
                for k, v in fields.items():
                    merged_fields.setdefault(k, []).append(v)

            merged_fields = {
                k: _torch.cat(v) if isinstance(v[0], _torch.Tensor)
                else [row for item in v for row in (item if isinstance(item, list) else [item])]
                for k, v in merged_fields.items()
            }
            images = all_images if any(img is not None for img in all_images) else None
            return all_prompt_ids, images, merged_fields

    # Mock reward function just to satisfy the trainer
    def mock_reward(prompts, completions, **kwargs):
        return [1.0] * len(prompts)

    grpo_config = GRPOConfig(
        output_dir="./tmp_preflight",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none" # Disable wandb for this test
    )

    trainer = _PerImageGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_data,
        reward_funcs=[mock_reward],
        processing_class=tokenizer,
    )

    print("\n>>> 4b. VERIFYING THE FIX: calling _tokenize_prompts on all 4 real, DIFFERENT images at once...")
    real_prompts = [train_data[i]["prompt"] for i in range(len(train_data))]
    for i in range(len(real_prompts)):
        for msg in real_prompts[i]:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        part["image"] = train_data[i]["images"][0]
    fixed_prompt_ids, _, _ = trainer._tokenize_prompts(real_prompts)
    fixed_lens = [len(p) for p in fixed_prompt_ids]
    print(f"Per-prompt token lengths (fixed): {fixed_lens}")
    if min(fixed_lens) > 400:
        print("✅✅ FIX CONFIRMED: every image in the batch is now properly expanded!")
    else:
        print("❌❌ Still collapsing — at least one prompt in the batch is still short.")
    
    print(">>> 5. Fetching the first PyTorch batch from the dataloader...")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    
    print("\n================ PREFLIGHT RESULTS ================")
    print(f"Batch type: {type(batch)}")
    
    # TRL GRPO can return either a dict (collated tensors) or a list (raw samples)
    if isinstance(batch, dict):
        print("Batch keys and tensor shapes:")
        for k, v in batch.items():
            shape = v.shape if hasattr(v, 'shape') else type(v)
            print(f"  {k}: {shape}")
            
        print("\n--- SANITY CHECKS ---")
        has_pixels = "pixel_values" in batch or "pixel_values_2d" in batch or any("pixel" in k for k in batch.keys())
        if has_pixels:
            print("✅ SUCCESS: Image tensors (pixel_values) are in the PyTorch batch!")
        else:
            print("❌ FAIL: No pixel_values in batch — images are NOT reaching the model!")
            
        prompt_key = next((k for k in ["prompt_input_ids", "input_ids"] if k in batch), None)
        if prompt_key:
            prompt_len = batch[prompt_key].shape[-1]
            print(f"Prompt sequence length: {prompt_len} tokens")
            if prompt_len > 500:
                print("✅ SUCCESS: Prompt length > 500 — vision tokens are present!")
            else:
                print(f"❌ FAIL: Prompt length is only {prompt_len} — no image tokens in prompt!")
                
    elif isinstance(batch, list):
        # TRL multimodal GRPO returns a list of raw sample dicts before collation
        print(f"Batch is a list of {len(batch)} samples")
        print(f"First sample keys: {list(batch[0].keys()) if batch else 'empty'}")
        
        first = batch[0]
        
        # Check if images survived into the batch
        print("\n--- SANITY CHECK 1: Data structure ---")
        has_images = "images" in first and first["images"] is not None and len(first["images"]) > 0
        if has_images:
            img = first["images"][0]
            print(f"✅ Images present in batch! Type: {type(img)}, Size: {getattr(img, 'size', 'N/A')}")
        else:
            print("❌ FAIL: 'images' key missing or empty in batch samples!")
            
        # Check prompt content for image placeholder
        prompt = first.get("prompt", [])
        has_image_token = any(
            item.get("type") == "image"
            for msg in prompt
            for item in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )
        if has_image_token:
            print("✅ Image placeholder {'type': 'image'} found in prompt messages!")
        else:
            print("❌ FAIL: No image placeholder in prompt messages!")
    else:
        print(f"Unknown batch type: {type(batch)}")
        print(f"Batch: {batch}")
        
    print("===================================================")

    # ------------------------------------------------------------------
    # STEP 6 — THE DEFINITIVE PROOF: Manually call the processor
    # This simulates exactly what TRL does internally during rollouts.
    # If pixel_values appear with >500 tokens, images are injected correctly.
    # ------------------------------------------------------------------
    print("\n>>> 6. DEFINITIVE PROOF: Manually calling processor on first sample...")
    try:
        # Grab the first raw sample from the dataset
        first_sample = train_data[0]
        prompt_messages = first_sample["prompt"]
        pil_images = first_sample["images"]  # list of PIL images from the "images" column
        
        # Step A: Apply chat template — produces text with <|image_pad|> placeholders
        # Note: process_vision_info is NOT used here because our prompt uses
        # {"type": "image"} placeholders. The images come from the separate "images" column.
        # This is EXACTLY how TRL handles it internally.
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"Chat template text length: {len(text)} chars")
        print(f"Contains image placeholder in text: {'<|image_pad|>' in text or 'image' in text.lower()}")
        
        # Step B: Run the full processor passing PIL images directly
        # This is the same call TRL makes internally during rollout generation
        inputs = tokenizer(
            text=[text],
            images=pil_images,  # pass PIL images directly — same as TRL's internal path
            return_tensors="pt",
            padding=True
        )
        
        print("\n================ DEFINITIVE RESULTS ================")
        print("Processor output keys:", list(inputs.keys()))
        
        token_len = inputs["input_ids"].shape[-1]
        print(f"Total token length: {token_len}")
        
        if "pixel_values" in inputs or "pixel_values_videos" in inputs:
            pv_key = "pixel_values" if "pixel_values" in inputs else "pixel_values_videos"
            print(f"pixel_values shape: {inputs[pv_key].shape}")
            print(f"\n✅✅ CONFIRMED: Images ARE being injected as vision tokens!")
            print(f"✅✅ Token count: {token_len} (text-only would be ~233 — difference = {token_len - 233} image tokens)")
        else:
            print(f"\n❌❌ FAIL: pixel_values NOT in processor output!")
            print(f"    Token count {token_len} — if ~233, images are being silently dropped!")
        print("====================================================")
        
    except Exception as e:
        print(f"Step 6 error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_override", default=None,
        help="Point preflight at a local merged checkpoint instead of the raw HF base "
             "(e.g. checkpoints/qwen3vl-2b/merged-sft-2b-v4) to test the exact model+"
             "tokenizer_name loading path used by a real GRPO run.",
    )
    parser.add_argument("--tier", default="2b")
    parser.add_argument("--task", default="violations_only")
    args = parser.parse_args()
    run_preflight(task=args.task, model_id=args.tier, base_model_override=args.base_model_override)
