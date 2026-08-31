"""
Builds the Base vs SFT vs SFT+GSPO comparison table for a task.
Usage: python experiments/compare_results.py --tier 2b
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from core.config import load_config
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger
from core.naming import task_prefix
from models.model_loader import get_model_info

logger = get_logger(__name__)

def load_eval_json(run_name: str) -> dict:
    """Load metrics.json for one inference run.

    Real on-disk layout, as written by the hpc_*.sh chain
    (run_inference -> structural_repair -> run_evaluation):

        results/inference/<run_name>/repair_applied/evaluation_results/metrics.json

    This previously looked in results/<short_name>/<variant>/metrics.json, which the
    pipeline has never written.
    """
    base = get_drive_path("results", "inference", run_name)
    candidates = [
        base / "repair_applied" / "evaluation_results" / "metrics.json",
        base / "evaluation_results" / "metrics.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    logger.warning(f"Missing results file for run '{run_name}'; looked in: "
                   + ", ".join(str(c) for c in candidates))
    return {}

def flatten_metrics(metrics: dict, label: str) -> dict:
    """Pick the headline metrics out of a FLAT metrics.json.

    run_evaluation.py writes a flat dict (it dumps eval_results["metrics"] directly).
    This function used to read metrics["strict_metrics"] / ["valid_metrics"] /
    ["structural_metrics"], a nested shape that is no longer produced — so every value
    resolved to None and the whole table came out empty.
    """
    if not metrics:
        return {"Model": label}

    m = metrics
    flat = {"Model": label}

    # Structural
    flat["Valid_JSON_%"] = m.get("structural_json_validity_rate", 0.0) * 100
    flat["Schema_Adherence_%"] = m.get("structural_schema_adherence_rate", 0.0) * 100
    flat["Prediction_Failure_%"] = m.get("violation_prediction_failure_rate", 0.0) * 100

    # Captioning (absent for violations_only)
    flat["BERTScore_F1"] = m.get("captioning_bertscore_f1")
    flat["CLIPScore"] = m.get("captioning_clipscore")
    flat["METEOR"] = m.get("captioning_meteor")
    flat["CIDEr-D"] = m.get("captioning_ciderd")

    # Object grounding (absent for violations_only)
    flat["Grounding_IoU_Macro_Mask"] = m.get("grounding_mask_iou_all_macro_mean_tn0")
    flat["Grounding_IoU_Macro_Greedy"] = m.get("grounding_greedy_iou_all_macro_mean_tn0")

    # Violation identification
    flat["Violation_F1_Micro"] = m.get("violation_identification_f1_micro")
    flat["Violation_Precision_Micro"] = m.get("violation_identification_precision_micro")
    flat["Violation_Recall_Micro"] = m.get("violation_identification_recall_micro")
    flat["Violation_F1_Macro"] = m.get("violation_identification_f1_macro")
    flat["Violation_Precision_Macro"] = m.get("violation_identification_precision_macro")
    flat["Violation_Recall_Macro"] = m.get("violation_identification_recall_macro")

    # Safe-image guardrail: recall on rule_0 is 1 - false-alarm-rate on safe images.
    flat["Rule0_Recall_(1-FalseAlarm)"] = m.get("violation_identification_recall_rule_0")
    flat["Rule0_Precision"] = m.get("violation_identification_precision_rule_0")

    # IoU-conditioned identification (the strict variant)
    flat["Violation_F1_Micro_IoUCond"] = m.get("violation_identification_iou_conditioned_f1_micro")

    # Violation grounding
    flat["Violation_Grounding_IoU_Mask"] = m.get("violation_grounding_mask_iou_macro_tn0")
    flat["Violation_Grounding_IoU_Greedy"] = m.get("violation_grounding_greedy_iou_macro_tn0")

    # Reasoning
    flat["Reasoning_BERTScore_F1_Macro"] = m.get("reasoning_text_similarity_bertscore_f1_macro")
    flat["Reasoning_BERTScore_F1_Micro"] = m.get("reasoning_text_similarity_bertscore_f1_micro")

    return flat

def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    parser.add_argument("--task", default="unified", help="Task name: 'unified' or 'violations_only'")
    parser.add_argument(
        "--version", required=True,
        help="Run version tag of the results to compare (e.g. v4, v5). Must match "
             "the version the baseline/SFT/GRPO runs were produced under."
    )
    args = parser.parse_args()

    model_info = get_model_info(args.tier)
    short_name = model_info["short_name"]
    prefix = args.task
    tier = args.tier
    version = args.version

    # SFT is evaluated from the 'best' checkpoint (lowest eval_loss) — that is the
    # checkpoint the merge step feeds to GRPO. GRPO has no best/, so it stays '_final'.
    if prefix == "unified":
        baseline_metrics = load_eval_json(f"baseline_{tier}_{version}")
        if not baseline_metrics:
            baseline_metrics = load_eval_json("baseline")
        sft_metrics = load_eval_json(f"unified-sft-{tier}-{version}_best")
        grpo_metrics = load_eval_json(f"unified-grpo-{tier}-{version}_final")
    else:
        # e.g. violations_only -> vo-sft-{version}
        short_prefix = task_prefix(prefix)
        # Baseline might be prefixed or not depending on how it was run
        baseline_metrics = load_eval_json(f"{short_prefix}-baseline-{tier}-{version}")
        if not baseline_metrics:
            baseline_metrics = load_eval_json(f"{prefix}-baseline")
        sft_metrics = load_eval_json(f"{short_prefix}-sft-{tier}-{version}_best")
        grpo_metrics = load_eval_json(f"{short_prefix}-grpo-{tier}-{version}_final")

    summary_rows = [
        flatten_metrics(baseline_metrics, "Base"),
        flatten_metrics(sft_metrics, "SFT"),
        flatten_metrics(grpo_metrics, "SFT+GRPO"),
    ]

    comparison_df = pd.DataFrame(summary_rows)
    out_path = get_drive_path("results", short_name, "comparison_table.csv")
    ensure_dir(out_path.parent)
    comparison_df.to_csv(out_path, index=False)

    logger.info(f"Comparison table saved to {out_path}")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()