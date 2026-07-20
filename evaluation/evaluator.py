"""
Main evaluator orchestrator.
Combines all metric functions into a unified evaluation pipeline.
"""
from typing import Dict, List, Any, Optional
import json
import os

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
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the complete evaluation pipeline.
    raw_predictions: list of raw string responses from the model.
    references: list of ground truth UnifiedOutput dictionaries.
    images: optional list of PIL Images (for CLIPScore).
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
        
    # Checkpoint state
    ckpt = {
        "structural_metrics": None,
        "caption_metrics": None,
        "grounding_metrics": None,
        "violation_metrics": None,
        "reasoning_metrics": None,
        "parsed_predictions": None,
        "failures": None
    }
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            ckpt.update(json.load(f))
            
    def save_checkpoint():
        if checkpoint_path:
            with open(checkpoint_path, 'w') as f:
                json.dump(ckpt, f, indent=2)
    
    # 1. Structural metrics & Parsing
    if ckpt["structural_metrics"] is None:
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
            
        ckpt["structural_metrics"] = structural_metrics
        ckpt["parsed_predictions"] = parsed_preds
        ckpt["failures"] = failures
        save_checkpoint()
    
    structural_metrics = ckpt["structural_metrics"]
    parsed_preds = ckpt["parsed_predictions"]
    failures = ckpt["failures"]
    
    # Extract components
    pred_captions = [p.get("caption", "") if p else "" for p in parsed_preds]
    gt_captions = [r.get("caption", "") for r in references]
    
    pred_objects = [p if p else {} for p in parsed_preds]
    gt_objects = references
    
    pred_violations = [p if p else {} for p in parsed_preds]
    gt_violations = references
    
    # 2. Captioning metrics
    if ckpt["caption_metrics"] is None:
        logger.info("Computing captioning metrics...")
        ckpt["caption_metrics"] = compute_all_caption_metrics(pred_captions, gt_captions, images=images, prefix="captioning_")
        save_checkpoint()
    caption_metrics = ckpt["caption_metrics"]
    
    # 3. Grounding metrics
    if ckpt["grounding_metrics"] is None:
        logger.info("Computing grounding metrics...")
        ckpt["grounding_metrics"] = compute_grounding_metrics(pred_objects, gt_objects)
        save_checkpoint()
    grounding_metrics = ckpt["grounding_metrics"]
    
    # 4. Violation metrics
    if ckpt["violation_metrics"] is None:
        logger.info("Computing safety violation metrics...")
        ckpt["violation_metrics"] = compute_violation_metrics(pred_violations, gt_violations)
        save_checkpoint()
    violation_metrics = ckpt["violation_metrics"]
    
    # 5. Reasoning metrics
    if ckpt["reasoning_metrics"] is None:
        logger.info("Computing reasoning metrics (Captioning Suite)...")
        ckpt["reasoning_metrics"] = batch_score_reasoning(pred_violations, gt_violations)
        save_checkpoint()
    reasoning_metrics = ckpt["reasoning_metrics"]
    
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