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
from evaluation.metrics_think import compute_think_metrics, task_expects_think_block
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
    use_llm_judge: bool = False,
    model_texts: List[str] = None,
) -> Dict[str, Any]:
    """
    Runs the complete evaluation pipeline.
    raw_predictions: list of raw string responses from the model.
    references: list of ground truth dicts from data.preprocessor.build_gt_dict.
    images: list of PIL Images (for CLIPScore).
    model_texts: optional, aligned with raw_predictions — what the MODEL emitted,
        before structural repair rewrote it. Only used by the think_* diagnostics,
        which are skipped (keys absent, not zero) when it is None. It is a separate
        argument because raw_predictions comes from the REPAIRED file by design, and
        repair replaces raw_output with re-serialized JSON for every record it fixed,
        stripping the <think> block along with it.
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
    #
    # TWO families are emitted, and the difference between them IS the repair.
    #
    #   structural_*       scored on `raw_predictions` -- the REPAIRED file, which is
    #                      what the fixed inference -> repair -> evaluation chain feeds
    #                      in. Unchanged; every historical number keeps its meaning.
    #   structural_*_raw   scored on `model_texts` -- what the MODEL actually emitted,
    #                      before structural_repair.py rewrote it (ADDED 2026-10-28,
    #                      README_v2.md P0-3).
    #
    # WHY: the post-repair number alone is badly misleading. Measured on
    # `vo-baseline-2b-v2` (n=3004), the four gates separate like this:
    #
    #     structural_json_validity_rate          0.9810   (repaired)
    #     structural_json_validity_rate_raw      0.9597   (model's own text)
    #     structural_schema_adherence_rate       0.9810   (repaired)
    #     structural_schema_adherence_rate_raw   0.2011   (model's own text)
    #
    # READ THE SECOND PAIR. 0.2011 is the honest answer to 'how often did this model
    # emit exactly the contract, unaided', and it is 4.9x below the number the file
    # reported before this change.
    #
    # AND NOTE THE MECHANISM, because it is easy to get wrong and this repo's own notes
    # had it wrong: those 2,400 records do NOT fail to parse. 95.97% of them parse
    # cleanly; they fail SCHEMA VALIDATION. Only 57 records (1.9%) are unparseable even
    # after repair. Verified by decomposing the gates on the saved predictions:
    #
    #     strict parse 95.97% | strict parse + schema 20.11%
    #     tolerant parse 95.97% | tolerant parse + schema 20.11%
    #
    # The strict and tolerant parsers agree exactly here, so for this task the whole
    # 75.9-point gap is the schema, not the JSON. Describing it as a parse failure
    # points debugging at the wrong layer (token caps and repetition loops) instead of
    # the right one (missing/ill-typed rule fields).
    #
    # `structural_schema_adherence_rate_raw` reproduces repair_report.json's `valid_raw`
    # share EXACTLY (0.2011 both ways) -- that status is defined as 'passed the strict
    # gate AND validated, with zero repair applied', which is the same conjunction. So
    # the number README_v2.md quotes by hand from repair_stats.csv is now a first-class
    # key in metrics.json, carried into the results index and every comparison table
    # automatically.
    #
    # The remaining movement -- raw schema adherence 20.11% -> 99.93% -- is the single
    # biggest thing SFT does to this model, and it was invisible here until now.
    #
    # SEMANTICS, deliberately chosen: the _raw family is the SAME function on a
    # different input. Nothing is filtered and nothing is scored differently -- an
    # unparseable output counts in the DENOMINATOR as a failure, exactly as it already
    # does post-repair (compute_structural_metrics divides both rates by the full
    # sample count). That is the only choice that makes the pair comparable: if the
    # raw family excluded format failures it would be a format-compliance rate that
    # excludes non-compliance, i.e. it would measure nothing, and any gap between the
    # two keys would conflate 'repair fixed it' with 'we changed the denominator'.
    # Because both come from one function, the gap is attributable to repair alone.
    #
    # SCOPE: reporting only. No other metric family changes -- the violation, grounding
    # and reasoning numbers are still scored on the repaired file exactly as before,
    # and an unparseable prediction still surfaces there as
    # violation_prediction_failure_{count,rate}. This adds keys; it reshapes nothing.
    #
    # When `model_texts` is None (spice_only, or a caller that did not pass it) the
    # _raw keys are ABSENT, never zero -- same convention as the LLM judge, so a
    # missing measurement can never be read as a measured failure. Scoring an
    # UNREPAIRED file is also well defined: run_evaluation.py's fallback makes
    # model_texts == raw_predictions there, so the two families agree, correctly.
    structural_metrics = {}
    if not spice_only:
        structural_metrics = compute_structural_metrics(raw_predictions, task=task)
        if model_texts is not None:
            raw_structural = compute_structural_metrics(model_texts, task=task)
            structural_metrics.update(
                {f"{k}_raw": v for k, v in raw_structural.items()}
            )
        
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
    judge_metrics, judge_details, judge_status = {}, None, None
    
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

            # 6. LLM-as-a-judge reasoning scores (dataset paper, Stage 3) -- OPT-IN, and
            #    added alongside the text-similarity suite above, never instead of it.
            #    Gated on the same capability, so object_only/caption_only skip it
            #    silently. When off, its keys are absent from metrics.json, never zeros.
            #    run_llm_judge is fail-soft: a judge that cannot load logs an ERROR and
            #    returns no keys rather than costing the rest of this evaluation.
            if use_llm_judge:
                from evaluation.metrics_llm_judge import run_llm_judge
                logger.info("Computing LLM-as-a-judge reasoning scores...")
                judge_metrics, judge_details, judge_status = run_llm_judge(
                    pred_violations, gt_violations
                )
    
    # Combine all results
    # 7. <think>-block diagnostics — only for a task whose prompt asks for a block.
    #
    # Gated on the PROMPT, not on a task name (core/tasks.py forbids task-name
    # literals) and not on whether blocks happen to appear: for violations_think the
    # keys are therefore always present, so an all-zero result reads as "the model
    # stopped emitting blocks" rather than "the metric never ran". For every other task
    # the keys are absent entirely, never zero.
    think_metrics = {}
    if not spice_only and task_expects_think_block(task):
        if model_texts is None:
            # Deliberately SKIPPED rather than computed from raw_predictions. On the
            # repaired file, `raw_output` is the re-serialized JSON for every record
            # that repair fixed (structural_repair.py:1526) and the model's original
            # text for every record it did not -- so a fallback would produce a
            # plausible-looking block-presence rate that is partly measuring the repair
            # stage. This repo has already paid for one metric that silently described
            # post-repair output (structural_json_validity_rate); absent keys are the
            # honest alternative.
            logger.warning(
                "Task %r expects a <think> block but model_texts was not provided; "
                "skipping think_* metrics. Pass the pre-repair text "
                "(original_raw_output, falling back to raw_output) to measure them.",
                task,
            )
        else:
            logger.info("Computing <think>-block diagnostics...")
            think_metrics = compute_think_metrics(model_texts, parsed_preds)

    all_metrics = {}
    all_metrics.update(structural_metrics)
    all_metrics.update(caption_metrics)
    all_metrics.update(grounding_metrics)
    all_metrics.update(violation_metrics)
    all_metrics.update(reasoning_metrics)
    all_metrics.update(judge_metrics)
    all_metrics.update(think_metrics)
    
    logger.info(f"Evaluation complete. {len(failures)} schema failures logged.")
    
    return {
        "metrics": all_metrics,
        "parsed_predictions": parsed_preds,
        "failures": failures,
        # None unless the LLM judge actually ran for this task (see step 6 above).
        "llm_judge_details": judge_details,
        "llm_judge_status": judge_status,
    }