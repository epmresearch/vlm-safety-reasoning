"""
Inference wrapper using Unsloth's FastVisionModel.

Handles single-image generation with proper vision token processing.
Outputs are returned as raw text strings (parsing is done by the
evaluation pipeline's output_parser module).
"""
from typing import Any, Dict, List, Optional

from core.constants import MAX_NEW_TOKENS_UNIFIED
from core.logging import get_logger
from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT

logger = get_logger(__name__)


def generate_single(
    model,
    tokenizer,
    pil_image,
    max_new_tokens: int = MAX_NEW_TOKENS_UNIFIED,
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
        text,
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
    max_new_tokens: int = MAX_NEW_TOKENS_UNIFIED,
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
