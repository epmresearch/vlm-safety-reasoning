"""
Entry point: runs batched inference (no evaluation) for a given model/checkpoint
against the processed test set. Works for baseline (no adapter) or any saved
SFT checkpoint (best / final / checkpoint-N).

Usage:
    # Baseline (no fine-tuning)
    python -m experiments.run_inference --tier 2b

    # A specific fine-tuned checkpoint
    python -m experiments.run_inference --tier 2b --variant unified-sft-v4 --checkpoint best

    # A specific intermediate checkpoint
    python -m experiments.run_inference --tier 2b --variant unified-sft-v4 --checkpoint checkpoint-300

    # Limit samples for a quick smoke test
    python -m experiments.run_inference --tier 2b --variant unified-sft-v4 --checkpoint best --max_samples 32
"""
import unsloth
import argparse
import json
import time

from core.config import load_config, load_task_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger, attach_file_logger
from core.run_manifest import save_run_manifest
from data.loader import load_processed_dataset
from data.prompt_templates import SYSTEM_PROMPT, UNIFIED_INSPECTION_PROMPT
from models.model_loader import load_model_for_inference, get_model_info
from models.inference import run_inference_batched

logger = get_logger(__name__)


def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument(
        "--variant", default=None,
        help="SFT checkpoint variant name (e.g., unified-sft-v4). "
             "Omit for baseline (no-adapter) inference."
    )
    parser.add_argument(
        "--checkpoint", default="best",
        help="Subdirectory under checkpoints/<tier>/<variant>/ to load: "
             "'best', 'final', or a specific 'checkpoint-N'. Ignored if --variant is omitted."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap test samples (debugging)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens")
    parser.add_argument(
        "--run_name", default=None,
        help="Name for the results subfolder under results/inference/. "
             "Defaults to '<variant>_<checkpoint>' or 'baseline'."
    )
    parser.add_argument("--max_seq_length", type=int, default=None,
                     help="Override inference max_seq_length")
    parser.add_argument(
        "--repetition_penalty", type=float, default=None,
        help="Override generation repetition_penalty (default: 1.0, from configs/tasks/unified.yaml)"
    )
    parser.add_argument("--task", default="unified", help="Task name: 'unified' or 'violations_only'")
    parser.add_argument(
        "--base_model_override", default=None,
        help="Explicit path to the base model to load (e.g. a merged SFT model, required for GRPO "
             "checkpoints since their adapter was trained on top of that merged model, not the raw "
             "HF base). Always preferred over the naming-convention auto-detect below."
    )
    args = parser.parse_args()

    # --- Resolve run identity + paths ---
    base_model_override = args.base_model_override

    if args.variant:
        run_name = args.run_name or f"{args.variant}_{args.checkpoint}"
        adapter_path = str(
            get_drive_path("checkpoints", f"qwen3vl-{args.tier}", args.variant, args.checkpoint)
        )

        # CRITICAL FIX for GRPO:
        # A GRPO adapter was trained on top of the MERGED SFT model, NOT the raw HF base model.
        # If --base_model_override wasn't passed explicitly, fall back to a naming-convention
        # guess — but if that guess also can't find a merged model, abort rather than silently
        # evaluating Base+GRPO (which drops the entire SFT step from the result).
        if base_model_override is None and "grpo" in args.variant.lower():
            # e.g., vo-grpo-2b-vN -> merged-vo-sft-2b-vN (task-namespaced so unified and
            # violations_only merged checkpoints can never collide at the same version —
            # see core.naming.merged_checkpoint_name). Derives the version from the
            # variant itself (not hardcoded) so this guess stays correct across bumps.
            import re as _re
            from core.naming import merged_checkpoint_name
            version_match = _re.search(r"-(v\d+)(?:_[^-]*)?$", args.variant)
            version = version_match.group(1) if version_match else ""
            merged_name = merged_checkpoint_name(args.task, args.tier, version)
            merged_base = get_drive_path("checkpoints", f"qwen3vl-{args.tier}", merged_name)

            import os
            if os.path.exists(os.path.join(merged_base, "config.json")):
                base_model_override = str(merged_base)
                print(f"Detected GRPO inference! Auto-detected merged base model: {base_model_override}")
            else:
                raise SystemExit(
                    f"Refusing to run inference: variant '{args.variant}' looks like a GRPO checkpoint, "
                    f"but no --base_model_override was passed and no merged model was found at "
                    f"{merged_base}. Running against the raw base model would silently evaluate "
                    f"Base+GRPO instead of SFT+GRPO. Pass --base_model_override explicitly."
                )
    else:
        run_name = args.run_name or "baseline"
        adapter_path = None

    results_dir = ensure_dir(get_drive_path("results", "inference", run_name))
    output_path = str(results_dir / "predictions.jsonl")

    # --- File logging, same pattern as run_sft.py ---
    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"inference_{args.tier}_{run_name}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    # --- Config for max_new_tokens ---
    base_config = load_config(training_kind="sft")
    task_config = load_task_config(args.task)
    max_new_tokens = (
        args.max_new_tokens
        or base_config.get("max_new_tokens")
        or task_config.get("max_new_tokens", 1000)
    )
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else task_config.get("repetition_penalty", 1.0)
    )

    # --- Manifest for reproducibility ---
    from data.prompt_templates import get_prompt_for_task
    run_config = {
        "experiment": f"inference_{run_name}",
        "task": args.task,
        "model_tier": args.tier,
        "variant": args.variant,
        "checkpoint": args.checkpoint if args.variant else None,
        "adapter_path": adapter_path,
        "base_model_override": base_model_override,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "prompts": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": get_prompt_for_task(args.task),
        },
    }
    save_run_manifest(str(results_dir), run_config)
    logger.info(json.dumps(run_config, indent=2))

    # --- Load data ---
    logger.info("Loading fully processed dataset...")
    splits = load_processed_dataset()
    test_data = splits["test"]
    if args.max_samples is not None:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
    logger.info(f"{len(test_data)} test samples loaded")

    # --- Load model ---
    logger.info(f"Loading model (tier={args.tier}, adapter={adapter_path or 'NONE — baseline'})")
    if base_model_override:
        logger.info(f"Using explicitly specified base model: {base_model_override}")
        
    model, tokenizer, info = load_model_for_inference(
        model_name=base_model_override, # overrides the default HF model
        tier=args.tier,
        adapter_path=adapter_path,
        max_seq_length=args.max_seq_length,
        task=args.task,
        # base_model_override may be a local merged checkpoint; always load the
        # tokenizer/processor from the original HF repo (see model_loader for why).
        tokenizer_name=get_model_info(args.tier)["hf_path"],
    )
    logger.info("Model loaded successfully!")

    # --- Run inference ---
    logger.info(f"Starting batched inference on {len(test_data)} samples (batch_size={args.batch_size})...")
    logger.info("Output streams incrementally — safe to re-run this command to auto-resume.")
    results = run_inference_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_data,
        task=args.task,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        max_samples=args.max_samples,
        output_path=output_path,
        repetition_penalty=repetition_penalty,
    )

    logger.info(f"Inference complete: {len(results)} total samples processed.")
    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()