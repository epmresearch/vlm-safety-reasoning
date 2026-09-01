"""
Main evaluator orchestrator.
Combines all metric functions into a unified evaluation pipeline.
"""
from typing import Dict, List, Any

from evaluation.output_parser import parse_output_for_task, validate_output_for_task
from evaluation.metrics_captioning import compute_all_caption_metrics
from evaluation.metrics_grounding import compute_grounding_metrics
from evaluation.metrics_violations import compute_violation_metrics
from evaluation.metrics_structural import compute_structural_metrics
from evaluation.metrics_reasoning import batch_score_reasoning
from core.tasks import CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS, task_has
from core.logging import get_logger

logger = get_logger(__name__)

def run_full_evaluation(
    raw_predictions: List[str],
    references: List[Dict[str, Any]],
    images: List[Any] = None,
    skip_spice: bool = False,
    spice_only: bool = False,
    task: str = "unified",
) -> Dict[str, Any]:
    """
    Runs the complete evaluation pipeline.
    raw_predictions: list of raw string responses from the model.
    references: list of ground truth dicts from data.preprocessor.build_gt_dict.
    images: list of PIL Images (for CLIPScore).
    task: any task registered in core/tasks.py. Which metric families run is
        decided by that task's capabilities, NOT by a task-name comparison:
        captioning needs a `caption` field, grounding needs the object classes,
        violations and reasoning need the rule_N_violation fields. A task whose
        output does not contain a family's fields would otherwise be scored
        against ground truth it was never asked to predict.
    """
    logger.info("Starting full evaluation pipeline...")

    wants_caption = task_has(task, CAP_CAPTION)
    wants_objects = task_has(task, CAP_OBJECTS)
    wants_violations = task_has(task, CAP_VIOLATIONS)

    # The text-scoring families are the ones that need pixels (CLIPScore) and a JVM
    # (METEOR/CIDEr-D/SPICE): captioning directly, and reasoning because it scores
    # violation reasons through the captioning suite.
    needs_text_scoring = wants_caption or wants_violations

    if needs_text_scoring and images is None:
        raise ValueError(
            "run_full_evaluation requires `images` (aligned list of PIL Images) "
            "for CLIPScore-based caption/reasoning metrics. Pass images=..., "
            "or drop CLIPScore from the pipeline if you truly want text-only eval."
        )

    # C2 fail-fast: METEOR, CIDEr-D, and SPICE require Java.
    # Detect missing Java upfront instead of silently producing metrics.json
    # with missing keys that are indistinguishable from zero-score metrics.
    # Gated: object_only scores no text at all and must not need a JVM.
    if needs_text_scoring:
        from evaluation.metrics_captioning import _check_java_available
        if not _check_java_available():
            raise RuntimeError(
                "Java is required for METEOR/CIDEr-D/SPICE evaluation but was "
                "not found on PATH. These metrics will be omitted from the results."
            )

    if len(raw_predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(raw_predictions)} predictions vs {len(references)} references"
        )
    
    # 1. Structural metrics & Parsing
    structural_metrics = {}
    if not spice_only:
        structural_metrics = compute_structural_metrics(raw_predictions, task=task)
        
    # Parse predictions and capture failures
    parsed_preds = []
    failures = []
    for i, raw_str in enumerate(raw_predictions):
        image_id = references[i].get("image_id", f"unknown_{i}")
            
        # 1. JSON Parse
        parsed = parse_output_for_task(raw_str, task=task)
        if parsed is None:
            parsed_preds.append(None)
            failures.append({
                "image_id": image_id,
                "error_type": "json_parse_error",
                "raw_prediction": raw_str
            })
            continue
                
        # 2. Schema Validation
        validated = validate_output_for_task(parsed, task=task)
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
    
    # Pass parse/schema failures through as None (NOT {}). compute_violation_metrics
    # distinguishes them from a genuine "no violation" prediction, so an unparseable
    # output can no longer be credited as a rule_0 true negative.
    pred_violations = list(parsed_preds)
    gt_violations = references
    
    # 2. Captioning metrics — tasks producing a `caption` field
    caption_metrics = {}
    if wants_caption:
        logger.info("Computing captioning metrics...")
        caption_metrics = compute_all_caption_metrics(
            pred_captions, gt_captions, images=images, 
            include_spice=not skip_spice, spice_only=spice_only, prefix="captioning_"
        )
    
    grounding_metrics = {}
    violation_metrics = {}
    reasoning_metrics = {}
    
    if not spice_only:
        # 3. Grounding metrics — tasks producing the object classes
        if wants_objects:
            logger.info("Computing grounding metrics...")
            grounding_metrics = compute_grounding_metrics(pred_objects, gt_objects)
        
        # 4. Violation metrics — tasks producing rule_N_violation
        if wants_violations:
            logger.info("Computing safety violation metrics...")
            violation_metrics = compute_violation_metrics(pred_violations, gt_violations)
        
            # 5. Reasoning metrics — scores the violation `reason` strings, so it is
            #    gated on the same capability.
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