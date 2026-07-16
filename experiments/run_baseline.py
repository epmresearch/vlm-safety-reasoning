"""
Entry point: runs baseline (no fine-tuning) inference + evaluation for the unified task.
Usage: python experiments/run_baseline.py --tier 2b
"""
import argparse
import json

from core.constants import DEFAULT_MODEL_TIER
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from data.loader import load_processed_dataset
from data.preprocessor import build_ground_truth_dict
from models.model_loader import load_model_for_inference
from models.inference import run_inference
from evaluation.evaluator import run_full_evaluation

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=DEFAULT_MODEL_TIER, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length for inference")
    args = parser.parse_args()

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
    logger.info("Running baseline inference...")
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=test_data,
        max_samples=args.max_samples,
    )

    raw_predictions = [res["raw_output"] for res in results]
    references = [build_ground_truth_dict(res["sample"]) for res in results]
    # inference.py strips the image to save RAM, so we pull it directly from the dataset split
    images = test_data["image"] if "image" in test_data.column_names else None
    
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