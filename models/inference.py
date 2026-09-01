"""
Inference wrapper using Unsloth's FastVisionModel.

Handles single-image generation with proper vision token processing.
Outputs are returned as raw text strings (parsing is done by the
evaluation pipeline's output_parser module).
"""
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from data.prompt_templates import SYSTEM_PROMPT, get_prompt_for_task

logger = get_logger(__name__)

# Plain constant, not a config read at import time. This used to be
# load_task_config("unified")["max_new_tokens"], which pinned a hardcoded task name
# into module import for every task's inference run. The real budget always comes
# from the task YAML via experiments/run_inference.py; this is only the fallback for
# a direct caller that passes nothing.
DEFAULT_MAX_NEW_TOKENS = 1000


def generate_single(
    model,
    tokenizer,
    pil_image,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    task: str = "unified",
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
    
    user_prompt = get_prompt_for_task(task)

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
                {"type": "text", "text": user_prompt},
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
            repetition_penalty=repetition_penalty,
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
    repetition_penalty: float = 1.0,
    task: str = "unified",
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

    # No try/except: a generation failure is a real failure and must crash the job
    # rather than be written out as a blank prediction that scores as a bad model.
    # Same reasoning as run_inference_batched — see its docstring.
    for sample in iterator:
        start_time = time.time()
        pil_image = sample["image"]

        output_text = generate_single(
            model, tokenizer, pil_image,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            task=task,
        )

        results.append({
            "image_id": sample.get("image_id", ""),
            "raw_output": output_text,
            "sample": {k: v for k, v in sample.items() if k != "image"},
            "latency_seconds": time.time() - start_time,
        })

    logger.info(f"Inference complete: {len(results)} samples processed.")
    return results

def generate_batch(
    model,
    tokenizer,
    pil_images: list,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    task: str = "unified",
    max_prompt_length: Optional[int] = None,
) -> List[str]:
    """Generates responses for a batch of images using the unified prompt."""
    from qwen_vl_utils import process_vision_info
    import torch

    # CRITICAL: left-padding for batched causal-LM generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    user_prompt = get_prompt_for_task(task)

    batch_messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
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

    # `truncation=True` with no `max_length` falls back to
    # tokenizer.model_max_length, which for Qwen3-VL is far larger than the window
    # the model was actually loaded with. That made the per-task
    # `inference_max_seq_length` (2816/3200/2688/2944) purely a model-load setting
    # that exerted no control over prompt length. Cap the PROMPT explicitly at
    # (window - generation budget) so a long prompt is truncated rather than
    # silently eating the space the completion needs.
    tokenizer_kwargs = {}
    if max_prompt_length is not None and max_prompt_length > 0:
        tokenizer_kwargs["max_length"] = max_prompt_length

    inputs = tokenizer(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        **tokenizer_kwargs,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
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
    repetition_penalty: float = 1.0,
    task: str = "unified",
    max_prompt_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs batched inference over a dataset split, start to finish.

    Deliberately has NO auto-resume and NO per-batch retry. Both existed for
    Google Colab, where the runtime disconnects unpredictably; on a SLURM cluster
    they are a liability rather than a safety net, because together they produced
    silently corrupted metrics:

      * a failing batch was caught and written out as `raw_output: ""` for every
        image in it, so a hard failure looked like a model that emitted nothing;
      * auto-resume then re-ran exactly those images (it only treated a non-empty
        output as complete) and APPENDED the retry, because the file was opened in
        "a" mode;
      * nothing downstream de-duplicates by image_id, so `predictions.jsonl` ended
        up with two records for the same image. One image answered perfectly on
        the retry reported `structural_json_validity_rate = 0.500` over a
        denominator of 2, and every metric was computed over an inflated
        denominator padded with spurious failures.

    The output file is opened once in "w" mode and truncated, so a re-run always
    produces a clean file rather than accumulating onto the last one. A batch
    failure propagates and kills the job, which is the correct behaviour when the
    scheduler will surface the traceback in the job log.
    """
    from tqdm import tqdm
    import time
    import json
    import gc
    import torch

    samples_to_process = dataset
    if max_samples is not None:
        samples_to_process = samples_to_process.select(
            range(min(max_samples, len(samples_to_process)))
        )

    n = len(samples_to_process)
    if n == 0:
        logger.warning("Inference dataset is empty — nothing to do.")
        return []

    results = []

    # Opened once, in "w" mode: truncates any previous run's file up front so a
    # re-run can never accumulate onto stale records. Kept open for the whole loop
    # and flushed per batch, so partial output survives an external kill (walltime,
    # scancel) for debugging — while still being a single clean write.
    out_f = open(output_path, "w", encoding="utf-8") if output_path else None
    try:
        for start in tqdm(range(0, n, batch_size), desc="Batched Inference", disable=not show_progress):
            batch = samples_to_process.select(range(start, min(start + batch_size, n)))
            pil_images = batch["image"]

            start_time = time.time()

            # No try/except: a failure here is a real failure (OOM, a corrupt
            # image, a broken processor) and must crash the job loudly rather than
            # be laundered into blank predictions that score as a bad model.
            outputs = generate_batch(
                model, tokenizer, pil_images,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                task=task,
                max_prompt_length=max_prompt_length,
            )

            per_image_latency = (time.time() - start_time) / len(pil_images)

            batch_results = [
                {
                    "image_id": sample.get("image_id", ""),
                    "raw_output": outputs[i],
                    "sample": {k: v for k, v in sample.items() if k != "image"},
                    "latency_seconds": per_image_latency,
                }
                for i, sample in enumerate(batch)
            ]
            results.extend(batch_results)

            if out_f is not None:
                for res in batch_results:
                    out_f.write(json.dumps(res) + "\n")
                out_f.flush()

            # Clear the CUDA cache to prevent memory fragmentation and OOM.
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if out_f is not None:
            out_f.close()

    logger.info(f"Batched inference complete: {len(results)} samples processed.")

    if len(results) != n:
        raise RuntimeError(
            f"Inference produced {len(results)} records for {n} input samples. "
            "Every downstream metric divides by the record count, so this must "
            "never happen silently."
        )

    # Convenience .json twin of the .jsonl. Written only on a clean finish.
    if output_path:
        json_output_path = output_path.replace(".jsonl", ".json")
        if not json_output_path.endswith(".json"):
            json_output_path += ".json"
        with open(json_output_path, "w", encoding="utf-8") as f_json:
            json.dump(results, f_json, indent=2)
        logger.info(f"Saved complete JSON to {json_output_path}")

    return results
