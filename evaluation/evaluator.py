"""
Main evaluator orchestrator.
Combines all metric functions into a unified evaluation pipeline.
"""
from typing import Dict, List, Any

from evaluation.output_parser import parse_model_output, validate_unified_output
from evaluation.metrics_captioning import compute_all_caption_metrics
from evaluation.metrics_grounding import compute_grounding_metrics
from evaluation.metrics_violations import compute_violation_metrics
from evaluation.metrics_structural import compute_structural_metrics
from evaluation.metrics_reasoning import batch_score_reasoning
from core.logging import get_logger

logger = get_logger(__name__)

def run_full_evaluation(
    raw_predictions: List[str],
    references: List[Dict[str, Any]],
    images: List[Any] = None,
) -> Dict[str, Any]:
    """
    Runs the complete evaluation pipeline.
    raw_predictions: list of raw string responses from the model.
    references: list of ground truth UnifiedOutput dictionaries.
    images: list of PIL Images (for CLIPScore).
    """
    logger.info("Starting full evaluation pipeline...")
    
    # C2 fail-fast: METEOR, CIDEr-D, and SPICE require Java.
    # Detect missing Java upfront instead of silently producing metrics.json
    # with missing keys that are indistinguishable from zero-score metrics.
    from evaluation.metrics_captioning import _check_java_available
    if not _check_java_available():
        raise RuntimeError(
            "Java is required for METEOR/CIDEr-D/SPICE evaluation but was "
            "not found on PATH. Install a JRE before running evaluation "
            "(e.g., `apt-get install -y default-jre` on Colab/Linux)."
        )
    
    if len(raw_predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(raw_predictions)} predictions vs {len(references)} references"
        )
    
    # 1. Structural metrics & Parsing
    structural_metrics = compute_structural_metrics(raw_predictions)
        
    # Parse predictions and capture failures
    parsed_preds = []
    failures = []
    for i, raw_str in enumerate(raw_predictions):
        image_id = references[i].get("image_id", f"unknown_{i}")
            
        # 1. JSON Parse
        parsed = parse_model_output(raw_str)
        if parsed is None:
            parsed_preds.append(None)
            failures.append({
                "image_id": image_id,
                "error_type": "json_parse_error",
                "raw_prediction": raw_str
            })
            continue
                
        # 2. Schema Validation
        validated = validate_unified_output(parsed)
        if validated is None:
            parsed_preds.append(None)
            failures.append({
                "image_id": image_id,
                "error_type": "schema_validation_error",
                "raw_prediction": raw_str
            })
            continue
                
        # Valid JSON and Valid Schema
        parsed_preds.append(parsed)
    
    # Extract components
    pred_captions = [p.get("caption", "") if p else "" for p in parsed_preds]
    gt_captions = [r.get("caption", "") for r in references]
    
    pred_objects = [p if p else {} for p in parsed_preds]
    gt_objects = references
    
    pred_violations = [p if p else {} for p in parsed_preds]
    gt_violations = references
    
    # 2. Captioning metrics
    logger.info("Computing captioning metrics...")
    caption_metrics = compute_all_caption_metrics(pred_captions, gt_captions, images=images, prefix="captioning_")
    
    # 3. Grounding metrics
    logger.info("Computing grounding metrics...")
    grounding_metrics = compute_grounding_metrics(pred_objects, gt_objects)
    
    # 4. Violation metrics
    logger.info("Computing safety violation metrics...")
    violation_metrics = compute_violation_metrics(pred_violations, gt_violations)
    
    # 5. Reasoning metrics
    logger.info("Computing reasoning metrics (Captioning Suite)...")
    reasoning_metrics = batch_score_reasoning(pred_violations, gt_violations, images=images)
    
    # Combine all results
    all_metrics = {}
    all_metrics.update(structural_metrics)
    all_metrics.update(caption_metrics)
    all_metrics.update(grounding_metrics)
    all_metrics.update(violation_metrics)
    all_metrics.update(reasoning_metrics)
    
    logger.info(f"Evaluation complete. {len(failures)} schema failures logged.")
    
    return {
        "metrics": all_metrics,
        "parsed_predictions": parsed_preds,
        "failures": failures
    }