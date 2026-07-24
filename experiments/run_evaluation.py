"""
Entry point: runs the full evaluation pipeline (structural, captioning,
grounding, violation, reasoning metrics) against an existing predictions.jsonl
file — the kind produced by experiments/run_inference.py. Does NOT run
inference itself.

Requires a JRE on PATH for METEOR/CIDEr-D/SPICE (run_full_evaluation fails
fast with a clear error if Java is missing — see evaluation/metrics_captioning.py).
SPICE also downloads ~2GB of CoreNLP models on first-ever call; this script
restores/saves that cache to Drive automatically via evaluation/spice_cache.py.

Usage:
    python -m experiments.run_evaluation --predictions_path /path/to/predictions.jsonl

    # Explicit output dir (defaults to the same dir as predictions_path)
    python -m experiments.run_evaluation \
        --predictions_path /path/to/predictions.jsonl \
        --output_dir /path/to/results/dir

    # Limit samples for a quick smoke test (evaluates only the first N records
    # found in the predictions file, not a re-run of inference)
    python -m experiments.run_evaluation --predictions_path ... --max_samples 32
"""
import argparse
import json
import subprocess
import shutil as sh
from pathlib import Path

from core.config import load_config
from core.io import ensure_dir, get_drive_path
from core.logging import get_logger, attach_file_logger
from core.run_manifest import save_run_manifest
from data.loader import load_processed_dataset
from data.preprocessor import build_ground_truth_dict
from evaluation.evaluator import run_full_evaluation
from evaluation.spice_cache import restore_spice_cache, save_spice_cache
from evaluation.metrics_captioning import _check_java_available

logger = get_logger(__name__)


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def ensure_java8_active():
    """Best-effort: if java-8 is installed but not the active `java`, switch to it.
    SPICE's bundled CoreNLP 3.6.0 requires Java 8 specifically (Java 17 fails
    with a JSON parser error.
    Does nothing if java-8 isn't installed — the caller should have provisioned
    it (e.g. `apt-get install -y openjdk-8-jdk-headless`) before running this.
    """
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


def main():
    config = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_path", required=True,
        help="Path to a predictions.jsonl file (from run_inference.py or a legacy baseline run)."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save metrics.json / failures / parsed predictions. "
             "Defaults to the same directory as --predictions_path."
    )
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Only evaluate the first N records in the predictions file.")
    parser.add_argument("--skip_java_switch", action="store_true",
                         help="Don't attempt to auto-switch to Java 8 for SPICE.")
    parser.add_argument("--wandb_project", type=str, default=None,
                         help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                         help="Weights & Biases run name")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions_path not found: {predictions_path}")

    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else predictions_path.parent)

    # --- File logging ---
    logs_dir = ensure_dir(get_drive_path(config.get("paths", {}).get("logs_subdir", "logs")))
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"evaluation_{output_dir.name}_{timestamp}.txt"
    attach_file_logger(str(log_file))

    # --- Java / SPICE setup ---
    if not args.skip_java_switch:
        logger.info("Checking Java setup for METEOR/CIDEr-D/SPICE...")
        ensure_java8_active()

    if not _check_java_available():
        raise RuntimeError(
            "Java is not on PATH at all. Install a JRE before running evaluation, "
            "e.g.: apt-get install -y openjdk-8-jdk-headless"
        )

    SPICE_CACHE_DIR = str(get_drive_path("tools", "spice_corenlp_cache"))
    cache_restored = restore_spice_cache(SPICE_CACHE_DIR)
    logger.info(f"SPICE/CoreNLP cache restored from Drive: {cache_restored}")
    if not cache_restored:
        logger.info(
            "First-ever run: SPICE will download ~2GB of CoreNLP models on its "
            "first call. This cache will be saved to Drive at the end of this run."
        )

    # --- Manifest ---
    run_config = {
        "experiment": f"evaluation_{output_dir.name}",
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "max_samples": args.max_samples,
    }
    save_run_manifest(str(output_dir), run_config)
    logger.info(json.dumps(run_config, indent=2))

    # --- Load predictions ---
    logger.info(f"Loading predictions from {predictions_path}...")
    records = load_jsonl(str(predictions_path))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    logger.info(f"Loaded {len(records)} prediction records.")

    raw_predictions = [r["raw_output"] for r in records]
    references = [build_ground_truth_dict(r["sample"]) for r in records]

    # --- Load dataset and build image_id -> PIL image map ---
    # Predictions were saved without images (see run_inference_batched), so we
    # re-attach them here by image_id — this also makes the mapping robust to
    # auto-resume having written records out of original dataset order.
    logger.info("Loading processed dataset to re-attach images by image_id...")
    splits = load_processed_dataset()
    test_data = splits["test"]
    image_map = {str(sample["image_id"]): sample["image"] for sample in test_data}
    images = [image_map.get(str(r.get("image_id"))) for r in records]

    missing_images = sum(1 for img in images if img is None)
    if missing_images:
        logger.warning(
            f"{missing_images} / {len(images)} records had no matching image_id "
            f"in the test split — CLIPScore/reasoning metrics for those will be affected."
        )

    # --- Run evaluation ---
    logger.info("Running full evaluation pipeline...")
    eval_results = run_full_evaluation(
        raw_predictions, references, images=images
    )

    # --- Save outputs ---
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["metrics"], f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Define separate paths for each error type (changed extensions to .json)
    parse_failures_path = output_dir / "json_parse_failures.json"
    schema_failures_path = output_dir / "schema_validation_failures.json"
    
    # Separate the failures into two lists in memory
    parse_failures = [
        f for f in eval_results.get("failures", []) 
        if f.get("error_type") == "json_parse_error"
    ]
    schema_failures = [
        f for f in eval_results.get("failures", []) 
        if f.get("error_type") == "schema_validation_error"
    ]
    
    parse_count = len(parse_failures)
    schema_count = len(schema_failures)
    
    # Write the lists to standard JSON files
    with open(parse_failures_path, "w", encoding="utf-8") as f_parse:
        json.dump(parse_failures, f_parse, indent=2)
        
    with open(schema_failures_path, "w", encoding="utf-8") as f_schema:
        json.dump(schema_failures, f_schema, indent=2)
                
    # Log the separate counts
    logger.info(f"JSON Parse failures logged to: {parse_failures_path} ({parse_count} failures)")
    logger.info(f"Schema Validation failures logged to: {schema_failures_path} ({schema_count} failures)")

    # 2. Save parsed predictions as a standard JSON array
    parsed_path = output_dir / "parsed_predictions.json"
    with open(parsed_path, "w", encoding="utf-8") as f:
        # Filter out the None values into a new list first
        valid_preds = [
            pred for pred in eval_results.get("parsed_predictions", []) 
            if pred is not None
        ]
        # Dump the filtered list
        json.dump(valid_preds, f, indent=2)
        
    logger.info(f"Parsed predictions saved to: {parsed_path}")

    # Combined record (raw + parsed + ground truth) for convenience when reviewing results.
    combined_log = []
    for i, r in enumerate(records):
        combined_log.append({
            "image_id": r["image_id"],
            "raw_output": r["raw_output"],
            "latency_seconds": r.get("latency_seconds", 0.0),
            "parsed_output": eval_results["parsed_predictions"][i],
            "ground_truth": references[i],
        })
    combined_path = output_dir / "predictions_with_eval.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_log, f, indent=2)
    logger.info(f"Combined predictions+eval saved to: {combined_path}")

    # --- Save SPICE cache for future runs (no-op if already up to date) ---
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
            
            # The metrics in run_evaluation.py are already flat, but we add failure counts
            metrics_to_log = dict(eval_results["metrics"])
            metrics_to_log["failures/json_parse"] = parse_count
            metrics_to_log["failures/schema_validation"] = schema_count
            
            wandb.log(metrics_to_log)
            wandb.finish()
        except ImportError:
            logger.error("wandb is not installed. Please install it (pip install wandb) to use W&B logging.")

    logger.info(f"Evaluation complete. All artifacts saved to: {output_dir}")
    logger.info(f"Total metrics tracked: {len(eval_results['metrics'])}")


if __name__ == "__main__":
    main()