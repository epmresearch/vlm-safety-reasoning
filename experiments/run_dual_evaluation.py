"""
Dual-Pass Evaluation Pipeline: Strict vs. Valid.
This script calculates metrics twice:
1. Strict: Across all samples. Parse/Schema failures are scored as 0.0 (empty strings/dicts).
2. Valid: Across only parse-successful samples. The references and images are subsetted accordingly.
"""
import argparse
import json
import subprocess
import shutil as sh
from pathlib import Path
from typing import Dict, List, Any

from core.config import load_config
from core.io import ensure_dir, get_drive_path
from core.logging import get_logger, attach_file_logger
from core.run_manifest import save_run_manifest
from data.loader import load_processed_dataset
from data.preprocessor import build_ground_truth_dict

from evaluation.output_parser import parse_model_output, validate_unified_output
from evaluation.metrics_captioning import compute_all_caption_metrics, _check_java_available
from evaluation.metrics_grounding import compute_grounding_metrics
from evaluation.metrics_violations import compute_violation_metrics
from evaluation.metrics_structural import compute_structural_metrics
from evaluation.metrics_reasoning import batch_score_reasoning
from evaluation.spice_cache import restore_spice_cache, save_spice_cache

logger = get_logger(__name__)


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def ensure_java8_active():
    verify = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
    if "1.8" in verify:
        logger.info("Java 8 already active.")
        return

    listing = subprocess.run(
        ["update-alternatives", "--list", "java"], capture_output=True, text=True
    ).stdout
    java8_candidates = [line.strip() for line in listing.splitlines() if "java-8" in line]
    if not java8_candidates:
        logger.warning(
            "Java 8 not found via update-alternatives. SPICE may fail on this "
            "JRE version. Install with: apt-get install -y openjdk-8-jdk-headless"
        )
        return

    subprocess.run(["update-alternatives", "--set", "java", java8_candidates[0]], check=True)
    verify = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
    if "1.8" in verify:
        logger.info(f"Switched active java to Java 8: {java8_candidates[0]}")
    else:
        logger.warning(f"Attempted Java 8 switch but verification failed: {verify}")


def run_dual_pass(raw_predictions: List[str], references: List[Dict[str, Any]], images: List[Any]):
    logger.info("Parsing and validating outputs...")
    
    parsed_preds = []
    failures = []
    
    # 1. The Parsing Gate (FIXED SIGNATURES)
    for idx, raw in enumerate(raw_predictions):
        image_id = references[idx].get("image_id", f"unknown_{idx}")
        
        parsed = parse_model_output(raw)
        if parsed is None:
            failures.append({"image_id": image_id, "error_type": "json_parse_error", "raw_prediction": raw})
            parsed_preds.append(None)
            continue
            
        validated = validate_unified_output(parsed)
        if validated is None:
            failures.append({"image_id": image_id, "error_type": "schema_validation_error", "raw_prediction": raw})
            parsed_preds.append(None)
            continue
            
        parsed_preds.append(parsed)

    # Calculate Structural Metrics (FIXED SIGNATURE)
    structural_metrics = compute_structural_metrics(raw_predictions)
    
    # 2. Split the Lists
    # Strict
    pred_captions_strict = [p.get("caption", "") if p else "empty" for p in parsed_preds]
    pred_objects_strict = [p if p else {} for p in parsed_preds]
    pred_violations_strict = [p if p else {} for p in parsed_preds]
    
    gt_captions_strict = [r.get("caption", "") for r in references]
    gt_objects_strict = references
    gt_violations_strict = references
    images_strict = images

    # Valid
    pred_captions_valid = []
    pred_objects_valid = []
    pred_violations_valid = []
    
    gt_captions_valid = []
    gt_objects_valid = []
    gt_violations_valid = []
    images_valid = []
    
    for i, p in enumerate(parsed_preds):
        if p is not None:
            pred_captions_valid.append(p.get("caption", ""))
            pred_objects_valid.append(p)
            pred_violations_valid.append(p)
            
            gt_captions_valid.append(references[i].get("caption", ""))
            gt_objects_valid.append(references[i])
            gt_violations_valid.append(references[i])
            
            if images and i < len(images):
                images_valid.append(images[i])
            else:
                images_valid.append(None)

    logger.info(f"Strict Pass: {len(pred_captions_strict)} samples.")
    logger.info(f"Valid Pass: {len(pred_captions_valid)} samples.")

    # 3. Pass 1: Strict
    logger.info("Running STRICT metrics pass...")
    strict_metrics = {}
    if len(pred_captions_strict) > 0:
        strict_metrics.update(compute_all_caption_metrics(pred_captions_strict, gt_captions_strict, images_strict, prefix="captioning_"))
        strict_metrics.update(compute_grounding_metrics(pred_objects_strict, gt_objects_strict))
        strict_metrics.update(compute_violation_metrics(pred_violations_strict, gt_violations_strict))
        strict_metrics.update(batch_score_reasoning(pred_violations_strict, gt_violations_strict, images=images_strict)) # Added missing images parameter

    # 4. Pass 2: Valid
    if not failures:
        logger.info("No parse or schema failures detected. VALID pass is identical to STRICT pass. Copying metrics to save time...")
        import copy
        valid_metrics = copy.deepcopy(strict_metrics)
    else:
        logger.info("Running VALID metrics pass...")
        valid_metrics = {}
        if len(pred_captions_valid) > 0:
            valid_metrics.update(compute_all_caption_metrics(pred_captions_valid, gt_captions_valid, images_valid, prefix="captioning_"))
            valid_metrics.update(compute_grounding_metrics(pred_objects_valid, gt_objects_valid))
            valid_metrics.update(compute_violation_metrics(pred_violations_valid, gt_violations_valid))
            valid_metrics.update(batch_score_reasoning(pred_violations_valid, gt_violations_valid, images=images_valid))

    return {
        "metrics": {
            "structural_metrics": structural_metrics,
            "strict_metrics": strict_metrics,
            "valid_metrics": valid_metrics
        },
        "failures": failures,
        "parsed_predictions": parsed_preds
    }


def main():
    config = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_java_switch", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions_path not found: {predictions_path}")

    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else predictions_path.parent)

    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"dual_evaluation_{output_dir.name}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    if not args.skip_java_switch:
        logger.info("Checking Java setup for SPICE...")
        ensure_java8_active()

    if not _check_java_available():
        raise RuntimeError("Java is required on PATH.")

    SPICE_CACHE_DIR = str(get_drive_path("tools", "spice_corenlp_cache"))
    restore_spice_cache(SPICE_CACHE_DIR)

    run_config = {
        "experiment": f"dual_evaluation_{output_dir.name}",
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "max_samples": args.max_samples,
    }
    save_run_manifest(str(output_dir), run_config)

    logger.info(f"Loading predictions from {predictions_path}...")
    records = load_jsonl(str(predictions_path))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    
    raw_predictions = [r["raw_output"] for r in records]
    references = [build_ground_truth_dict(r["sample"]) for r in records]

    logger.info("Loading processed dataset to re-attach images...")
    splits = load_processed_dataset()
    test_data = splits["test"]
    image_map = {str(sample["image_id"]): sample["image"] for sample in test_data}
    images = [image_map.get(str(r.get("image_id"))) for r in records]

    # --- Run Dual Evaluation ---
    eval_results = run_dual_pass(raw_predictions, references, images)

    # --- Save Nested Metrics ---
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["metrics"], f, indent=2, ensure_ascii=False)
    logger.info(f"Nested Metrics saved to: {metrics_path}")

    # --- Save Failures ---
    parse_failures_path = output_dir / "json_parse_failures.json"
    schema_failures_path = output_dir / "schema_validation_failures.json"
    
    parse_failures = [f for f in eval_results.get("failures", []) if f.get("error_type") == "json_parse_error"]
    schema_failures = [f for f in eval_results.get("failures", []) if f.get("error_type") == "schema_validation_error"]
    
    with open(parse_failures_path, "w", encoding="utf-8") as f:
        json.dump(parse_failures, f, indent=2)
    with open(schema_failures_path, "w", encoding="utf-8") as f:
        json.dump(schema_failures, f, indent=2)

    parsed_path = output_dir / "parsed_predictions.json"
    with open(parsed_path, "w", encoding="utf-8") as f:
        valid_preds = [p for p in eval_results.get("parsed_predictions", []) if p is not None]
        json.dump(valid_preds, f, indent=2)

    save_spice_cache(SPICE_CACHE_DIR)
    
    # --- W&B Logging ---
    if args.wandb_project:
        try:
            import wandb
            logger.info("Initializing W&B logging...")
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=run_config
            )
            
            def flatten_dict(d: dict, parent_key: str = '', sep: str = '/') -> dict:
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
                
            flat_metrics = flatten_dict(eval_results["metrics"])
            flat_metrics["failures/json_parse"] = len(parse_failures)
            flat_metrics["failures/schema_validation"] = len(schema_failures)
            
            wandb.log(flat_metrics)
            
            artifact = wandb.Artifact(name=f"eval_results_{output_dir.name}", type="evaluation")
            artifact.add_file(str(metrics_path))
            artifact.add_file(str(parse_failures_path))
            artifact.add_file(str(schema_failures_path))
            wandb.log_artifact(artifact)
            
            wandb.finish()
        except ImportError:
            logger.error("wandb is not installed. Please install it (pip install wandb) to use W&B logging.")

    logger.info("Dual-pass evaluation complete.")


if __name__ == "__main__":
    main()
