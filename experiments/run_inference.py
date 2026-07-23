"""
Entry point: runs batched inference (no evaluation) for a given model/checkpoint
against the processed test set. Works for baseline (no adapter) or any saved
SFT checkpoint (best / final / checkpoint-N).

Usage:
    # Baseline (no fine-tuning)
    python -m experiments.run_inference --tier 2b

    # A specific fine-tuned checkpoint
    python -m experiments.run_inference --tier 2b --variant unified-sft-v1 --checkpoint best

    # A specific intermediate checkpoint
    python -m experiments.run_inference --tier 2b --variant unified-sft-v1 --checkpoint checkpoint-300

    # Limit samples for a quick smoke test
    python -m experiments.run_inference --tier 2b --variant unified-sft-v1 --checkpoint best --max_samples 32
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
from models.model_loader import load_model_for_inference
from models.inference import run_inference_batched

logger = get_logger(__name__)


def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument(
        "--variant", default=None,
        help="SFT checkpoint variant name (e.g., unified-sft-v1). "
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
    args = parser.parse_args()

    # --- Resolve run identity + paths ---
    if args.variant:
        run_name = args.run_name or f"{args.variant}_{args.checkpoint}"
        adapter_path = str(
            get_drive_path("checkpoints", f"qwen3vl-{args.tier}", args.variant, args.checkpoint)
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
    task_config = load_task_config("unified")
    max_new_tokens = (
        args.max_new_tokens
        or base_config.get("max_new_tokens")
        or task_config.get("max_new_tokens", 1000)
    )

    # --- Manifest for reproducibility ---
    run_config = {
        "experiment": f"inference_{run_name}",
        "model_tier": args.tier,
        "variant": args.variant,
        "checkpoint": args.checkpoint if args.variant else None,
        "adapter_path": adapter_path,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_new_tokens": max_new_tokens,
        "prompts": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": UNIFIED_INSPECTION_PROMPT,
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
    model, tokenizer, info = load_model_for_inference(
        tier=args.tier,
        adapter_path=adapter_path,
        max_seq_length=args.max_seq_length,
    )
    logger.info("Model loaded successfully!")

    # --- Run inference ---
    logger.info(f"Starting batched inference on {len(test_data)} samples (batch_size={args.batch_size})...")
    logger.info("Output streams incrementally — safe to re-run this command to auto-resume.")
    results = run_inference_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_data,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        max_samples=args.max_samples,
        output_path=output_path,
    )

    logger.info(f"Inference complete: {len(results)} total samples processed.")
    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()