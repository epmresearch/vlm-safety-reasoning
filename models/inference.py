"""
Inference wrapper using Unsloth's FastVisionModel.

Handles single-image generation with proper vision token processing.
Outputs are returned as raw text strings (parsing is done by the
evaluation pipeline's output_parser module).
"""
from typing import Any, Dict, List, Optional

from core.config import load_task_config
from core.logging import get_logger
from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT

logger = get_logger(__name__)
DEFAULT_MAX_NEW_TOKENS = load_task_config("unified").get("max_new_tokens", 1000)


def generate_single(
    model,
    tokenizer,
    pil_image,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    """Generates a response for a single image using the unified prompt.

    Args:
        model: FastVisionModel (in inference mode).
        tokenizer: Associated tokenizer.
        pil_image: PIL Image object.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        do_sample: Whether to use sampling.

    Returns:
        Raw output text string from the model.
    """
    from qwen_vl_utils import process_vision_info

    # Build the message in Qwen chat format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": UNIFIED_INSPECTION_PROMPT},
            ],
        },
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Process vision info for Qwen-VL
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize
    inputs = tokenizer(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Generate
    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.15,
            use_cache=True,
        )

    # Decode only the generated tokens (skip input)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text


def run_inference(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Runs inference on a dataset split, returning raw outputs.

    Args:
        model: FastVisionModel (in inference mode).
        tokenizer: Associated tokenizer.
        dataset: HF Dataset split with "image" and "image_id" columns.
        max_new_tokens: Maximum tokens per generation.
        max_samples: Optional cap (for debugging).
        show_progress: Whether to show tqdm progress bar.

    Returns:
        List of dicts: {"image_id": str, "raw_output": str, "sample": dict}
    """
    from tqdm import tqdm
    import time

    samples_to_process = dataset
    if max_samples is not None:
        samples_to_process = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    results = []
    iterator = tqdm(
        samples_to_process,
        desc="Inference",
        disable=not show_progress,
    )

    for sample in iterator:
        try:
            start_time = time.time()
            pil_image = sample["image"]

            output_text = generate_single(
                model, tokenizer, pil_image,
                max_new_tokens=max_new_tokens,
            )

            elapsed = time.time() - start_time
            results.append({
                "image_id": sample.get("image_id", ""),
                "raw_output": output_text,
                "sample": {k: v for k, v in sample.items() if k != "image"},
                "latency_seconds": elapsed,
            })
        except Exception as e:
            logger.warning(
                f"Inference failed for {sample.get('image_id', '?')}: {e}"
            )
            results.append({
                "image_id": sample.get("image_id", ""),
                "raw_output": "",
                "sample": {k: v for k, v in sample.items() if k != "image"},
                "latency_seconds": 0.0,
                "error": str(e),
            })

    logger.info(
        f"Inference complete: {len(results)} samples processed, "
        f"{sum(1 for r in results if 'error' in r)} errors"
    )
    return results

def generate_batch(
    model,
    tokenizer,
    pil_images: list,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> List[str]:
    """Generates responses for a batch of images using the unified prompt."""
    from qwen_vl_utils import process_vision_info
    import torch

    # CRITICAL: left-padding for batched causal-LM generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": UNIFIED_INSPECTION_PROMPT},
            ]},
        ]
        for img in pil_images
    ]

    texts = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in batch_messages
    ]

    # Process vision info across all messages
    all_image_inputs = []
    all_video_inputs = []
    for conv in batch_messages:
        img_in, vid_in = process_vision_info(conv)
        if img_in:
            all_image_inputs.extend(img_in if isinstance(img_in, list) else [img_in])
        if vid_in:
            all_video_inputs.extend(vid_in if isinstance(vid_in, list) else [vid_in])
    image_inputs = all_image_inputs if all_image_inputs else None
    video_inputs = all_video_inputs if all_video_inputs else None

    inputs = tokenizer(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.15,
            use_cache=True,
        )

    # Slice off the prompt tokens per-row
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def run_inference_batched(
    model,
    tokenizer,
    dataset,
    batch_size: int = 16,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs batched inference on a dataset split with Auto-Resume support."""
    from tqdm import tqdm
    import time
    import json
    import os
    import gc
    import torch

    # 1. Auto-Resume logic: Check existing results
    completed_ids = set()
    results = []  # Initialize early to hold both previously completed and new records
    
    if output_path and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Only consider an image "completed" if it actually generated a valid text output
                        if "image_id" in record and record.get("raw_output", "").strip():
                            completed_ids.add(str(record["image_id"]))
                            results.append(record)  # <-- FIX: Reload prior predictions into memory
                    except json.JSONDecodeError:
                        continue
        if completed_ids:
            logger.info(f"Auto-Resume: Loaded {len(completed_ids)} completed images into memory from {output_path}")

    # 2. Filter dataset
    samples_to_process = dataset
    if completed_ids:
        # Filter out images that have already been processed
        samples_to_process = dataset.filter(lambda x: str(x.get("image_id")) not in completed_ids)
        logger.info(f"Remaining images to process: {len(samples_to_process)}")

    if max_samples is not None:
        # Adjust max_samples to account for images already processed in previous runs
        remaining_allowed = max(0, max_samples - len(completed_ids))
        samples_to_process = samples_to_process.select(range(min(remaining_allowed, len(samples_to_process))))

    n = len(samples_to_process)
    
    if n == 0:
        logger.info("No images left to process! Inference is fully complete.")
        return results

    for start in tqdm(range(0, n, batch_size), desc="Batched Inference", disable=not show_progress):
        batch = samples_to_process.select(range(start, min(start + batch_size, n)))
        pil_images = batch["image"]

        start_time = time.time()
        batch_results = []
        try:
            outputs = generate_batch(model, tokenizer, pil_images, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start_time
            per_image_latency = elapsed / len(pil_images)

            for i, sample in enumerate(batch):
                batch_results.append({
                    "image_id": sample.get("image_id", ""),
                    "raw_output": outputs[i],
                    "sample": {k: v for k, v in sample.items() if k != "image"},
                    "latency_seconds": per_image_latency,
                })
        except Exception as e:
            error_msg = f"Batch inference failed for batch starting at {start}: {e}"
            logger.info(error_msg)      # downgraded from warning -> info
            tqdm.write(f"⚠️  {error_msg}") 

            # logger.warning(f"Batch inference failed for batch starting at {start}: {e}")
            # Fall back to empty output so one bad image doesn't kill the whole dataset
            for sample in batch:
                batch_results.append({
                    "image_id": sample.get("image_id", ""),
                    "raw_output": "",
                    "sample": {k: v for k, v in sample.items() if k != "image"},
                    "latency_seconds": 0.0,
                    "error": str(e),
                })
        
        results.extend(batch_results)
        
        # 3. Incremental Save to JSONL
        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                for res in batch_results:
                    f.write(json.dumps(res) + "\n")
                    
        # 4. Clear CUDA Cache to prevent memory fragmentation and OOM
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Batched inference complete: {len(results)} new samples processed.")
    return results
