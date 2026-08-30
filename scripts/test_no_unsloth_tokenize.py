"""
No-Unsloth path, step 1: does vanilla trl.GRPOTrainer correctly tokenize a
batch of REAL, different prompts (your actual system+user+image structure,
real construction-site photos) — and if not, does our per-conversation-loop
fix work cleanly on it (unlike on Unsloth's fused, wrapped trainer)?

Zero unsloth import anywhere. Uses your real data pipeline
(data.preprocessor.build_grpo_dataset_for_task) so this isn't testing a
simplified synthetic prompt shape — it's testing exactly what a real
training step would send through _tokenize_prompts.

Usage:
    python scripts/test_no_unsloth_tokenize.py \
        --merged_path /home/$USER/vlm-finetuning-project1/checkpoints/qwen3vl-2b/merged-sft-2b-v4 \
        --raw_hf_path unsloth/Qwen3-VL-2B-Instruct
"""
import argparse
import importlib.machinery
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Deliberately no `import unsloth` anywhere in this file or its imports.
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

# trl.trainer.grpo_trainer does an unconditional top-level
# `from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator`,
# even though we never set use_vllm=True. vllm isn't installed in this env
# (only pulled in indirectly when Unsloth is imported first), so stub just
# enough of it to satisfy the import chain — never actually exercised below.
# Every stub module needs a real __spec__, or importlib.util.find_spec()
# raises ValueError when it finds the module already cached in sys.modules
# with no spec attached.
if "vllm" not in sys.modules:
    def _make_stub(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
        return mod

    _vllm = _make_stub("vllm")
    _dist = _make_stub("vllm.distributed")
    _dc = _make_stub("vllm.distributed.device_communicators")
    _pynccl = _make_stub("vllm.distributed.device_communicators.pynccl")
    _pynccl.PyNcclCommunicator = object
    _vllm.distributed = _dist
    _dist.device_communicators = _dc
    _dc.pynccl = _pynccl

# Some trl versions gate their vllm import behind is_vllm_available(), which
# tries to parse our stub's (nonexistent) package version and crashes with
# "Invalid version: 'N/A'". Neutralize the check itself so it never tries.
import trl.import_utils as _trl_import_utils
_trl_import_utils._vllm_available = False
_trl_import_utils.is_vllm_available = lambda: False

from trl import GRPOTrainer, GRPOConfig


class _PerImageGRPOTrainer(GRPOTrainer):
    """Only relevant for vanilla trl.GRPOTrainer — this is a NORMAL subclass
    override (unlike the Unsloth case, trl.GRPOTrainer is not fused/wrapped,
    so overriding _tokenize_prompts here should actually take effect)."""

    def _tokenize_prompts(self, prompts: list):
        all_prompt_ids = []
        all_images = []
        merged_fields = {}
        for prompt in prompts:
            ids_list, imgs_list, fields = super()._tokenize_prompts([prompt])
            all_prompt_ids.append(ids_list[0])
            all_images.append(imgs_list[0] if imgs_list else None)
            for k, v in fields.items():
                merged_fields.setdefault(k, []).append(v)
        merged_fields = {
            k: torch.cat(v) if isinstance(v[0], torch.Tensor)
            else [row for item in v for row in (item if isinstance(item, list) else [item])]
            for k, v in merged_fields.items()
        }
        images = all_images if any(img is not None for img in all_images) else None
        return all_prompt_ids, images, merged_fields


def _embed_images_inline(prompts, train_data):
    """_tokenize_prompts expects images already inline in message content
    (that's how vanilla TRL's own upstream code hands prompts to it), so
    embed them the same way here before calling it directly."""
    for i, prompt in enumerate(prompts):
        for msg in prompt:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        part["image"] = train_data[i]["images"][0]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_path", required=True)
    parser.add_argument("--raw_hf_path", default="unsloth/Qwen3-VL-2B-Instruct")
    parser.add_argument("--task", default="violations_only")
    args = parser.parse_args()

    print(">>> 1. Loading processor from the RAW HF repo...")
    processor = AutoProcessor.from_pretrained(args.raw_hf_path)

    print(">>> 2. Loading model weights from the MERGED local checkpoint (bf16, no quantization)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(">>> 3. Applying a fresh LoRA adapter via plain peft (language layers only)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(">>> 4. Loading 4 REAL samples through your actual data pipeline...")
    from data.loader import load_processed_dataset
    from data.preprocessor import build_grpo_dataset_for_task
    raw_dataset = load_processed_dataset()
    train_split = raw_dataset["train"].select(range(4))
    train_data = build_grpo_dataset_for_task(train_split, task=args.task)

    def mock_reward(prompts, completions, **kwargs):
        return [1.0] * len(prompts)

    grpo_config = GRPOConfig(
        output_dir="./tmp_no_unsloth_test",
        per_device_train_batch_size=1,
        num_generations=2,
        remove_unused_columns=False,
        report_to="none",
    )

    print(">>> 5. Building vanilla trl.GRPOTrainer (baseline, UNMODIFIED)...")
    baseline_trainer = GRPOTrainer(
        model=model, args=grpo_config, train_dataset=train_data,
        reward_funcs=[mock_reward], processing_class=processor,
    )
    real_prompts_a = _embed_images_inline(
        [train_data[i]["prompt"] for i in range(len(train_data))], train_data
    )
    prompt_ids, _, fields = baseline_trainer._tokenize_prompts(real_prompts_a)
    lens_a = [len(p) for p in prompt_ids]
    print(f"\n=== TEST A: vanilla trl._tokenize_prompts, 4 real different images, batched together ===")
    print(f"Per-prompt token lengths: {lens_a}")
    print(f"multimodal_fields keys: {list(fields.keys())}")
    if min(lens_a) > 400:
        print("✅ Vanilla trl ALREADY handles this correctly — no fix needed, the bug really was Unsloth-only.")
    else:
        print("❌ Vanilla trl ALSO collapses this — the bug is in trl 0.23.0 itself, not something Unsloth added.")

    print("\n>>> 6. Building the SAME test through our per-conversation-loop fix (_PerImageGRPOTrainer)...")
    fixed_trainer = _PerImageGRPOTrainer(
        model=model, args=grpo_config, train_dataset=train_data,
        reward_funcs=[mock_reward], processing_class=processor,
    )
    real_prompts_b = _embed_images_inline(
        [train_data[i]["prompt"] for i in range(len(train_data))], train_data
    )
    prompt_ids_b, _, fields_b = fixed_trainer._tokenize_prompts(real_prompts_b)
    lens_b = [len(p) for p in prompt_ids_b]
    print(f"\n=== TEST B: our fix's _tokenize_prompts override, same 4 real images ===")
    print(f"Per-prompt token lengths: {lens_b}")
    print(f"multimodal_fields keys: {list(fields_b.keys())}")
    if min(lens_b) > 400:
        print("✅✅ FIX CONFIRMED on vanilla trl: every image properly expanded.")
        print("    => No-Unsloth + this fix is a viable, working path for real GRPO training.")
    else:
        print("❌❌ Still collapsing even with the fix applied to vanilla trl — needs more investigation.")


if __name__ == "__main__":
    main()
