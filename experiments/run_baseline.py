"""
Entry point: runs baseline (no fine-tuning) inference + evaluation for the unified task.
Usage: python experiments/run_baseline.py --tier 2b
"""
import argparse
import json
import os
from pathlib import Path

from core.config import load_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from data.loader import load_processed_dataset
from data.preprocessor import build_ground_truth_dict
from models.model_loader import load_model_for_inference
from models.inference import run_inference_batched
from evaluation.evaluator import run_full_evaluation
from core.run_manifest import save_run_manifest

logger = get_logger(__name__)

def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for batched generation")
    parser.add_argument("--max_seq_length", type=int, default=config.get("max_seq_length", 8192), help="Max sequence length for inference")
    args = parser.parse_args()

    # Save run manifest
    results_dir = str(get_drive_path("results", f"baseline_{args.tier}"))
    save_run_manifest(results_dir, vars(args))

    # Load dataset
    logger.info("Loading unified dataset...")
    splits = load_processed_dataset()
    test_data = splits["test"]
    if args.max_samples:
        test_data = test_data.select(range(args.max_samples))

    # Load base model
    logger.info(f"Loading baseline model for tier: {args.tier}...")
    model, tokenizer, model_info = load_model_for_inference(
        tier=args.tier,
        max_seq_length=args.max_seq_length
    )

    # Run inference
    logger.info(f"Running baseline batched inference (batch_size={args.batch_size})...")
    output_path = str(get_drive_path("results", model_info["short_name"], "baseline", "inference_raw.jsonl"))
    ensure_dir(Path(output_path).parent)
    results = run_inference_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_data,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_path=output_path,
    )

    raw_predictions = [res["raw_output"] for res in results]
    references = [build_ground_truth_dict(res["sample"]) for res in results]
    
    # Ensure 1-to-1 alignment between results and images.
    # Auto-resume can cause the `results` list to be in a different order 
    # than the original `test_data`. We must map them safely by image_id.
    if "image" in test_data.column_names:
        image_map = {str(sample["image_id"]): sample["image"] for sample in test_data}
        images = [image_map.get(str(res.get("image_id"))) for res in results]
    else:
        images = None
    
    # Run evaluation
    logger.info("Running evaluation...")
    eval_results = run_full_evaluation(raw_predictions, references, images=images)

    # Save results
    output_dir = ensure_dir(get_drive_path("results", model_info["short_name"], "baseline"))
    
    # Save aggregated metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(eval_results["metrics"], f, indent=2)
        
    # Save the raw predictions, latency, and parsing data so you can analyze it later
    inference_log = []
    for i, res in enumerate(results):
        inference_log.append({
            "image_id": res["image_id"],
            "raw_output": res["raw_output"],
            "latency_seconds": res.get("latency_seconds", 0.0),
            "parsed_output": eval_results["parsed_predictions"][i],
            "ground_truth": references[i]
        })
        
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(inference_log, f, indent=2)
        
    logger.info(f"Baseline run complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()