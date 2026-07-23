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
from models.model_loader import get_model_info

logger = get_logger(__name__)

def load_eval_json(model_short_name: str, variant: str) -> dict:
    path = get_drive_path("results", model_short_name, variant) / "metrics.json"
    if not path.exists():
        logger.warning(f"Missing results file: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def flatten_metrics(metrics: dict, label: str) -> dict:
    if not metrics:
        return {"Model": label}
    flat = {"Model": label}
    # Caption metrics
    flat["BERTScore_F1"] = metrics.get("captioning_bertscore_f1")
    flat["CLIPScore"] = metrics.get("captioning_clipscore")
    flat["METEOR"] = metrics.get("captioning_meteor")
    flat["CIDEr-D"] = metrics.get("captioning_ciderd")
    
    # Grounding metrics (using _tn0 as default per object-level grounding convention)
    flat["Grounding_IoU_Macro_Mask"] = metrics.get("grounding_mask_iou_all_macro_mean_tn0")
    flat["Grounding_IoU_Macro_Greedy"] = metrics.get("grounding_greedy_iou_all_macro_mean_tn0")
    
    # Violation metrics
    flat["Violation_F1_Micro"] = metrics.get("violation_identification_f1_micro")
    flat["Violation_Precision_Micro"] = metrics.get("violation_identification_precision_micro")
    flat["Violation_Recall_Micro"] = metrics.get("violation_identification_recall_micro")
    flat["Violation_F1_Macro"] = metrics.get("violation_identification_f1_macro")
    flat["Violation_Precision_Macro"] = metrics.get("violation_identification_precision_macro")
    flat["Violation_Recall_Macro"] = metrics.get("violation_identification_recall_macro")
    flat["Violation_Grounding_IoU_Mask"] = metrics.get("violation_grounding_mask_iou_macro_tn0")
    flat["Violation_Grounding_IoU_Greedy"] = metrics.get("violation_grounding_greedy_iou_macro_tn0")
    # Structural metrics
    flat["Valid_JSON_%"] = (metrics.get("structural_json_validity_rate", 0.0)) * 100
    flat["Schema_Adherence_%"] = (metrics.get("structural_schema_adherence_rate", 0.0)) * 100
    # Reasoning metrics
    flat["Reasoning_BERTScore_F1_Macro"] = metrics.get("reasoning_text_similarity_bertscore_f1_macro")
    flat["Reasoning_BERTScore_F1_Micro"] = metrics.get("reasoning_text_similarity_bertscore_f1_micro")
    return flat

def main():
    config = load_config()
    default_tier = config.get("active_tier", "2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=default_tier, help="Model tier (e.g., 2b, 4b, 8b)")
    args = parser.parse_args()

    model_info = get_model_info(args.tier)
    short_name = model_info["short_name"]

    baseline_metrics = load_eval_json(short_name, "baseline")
    sft_metrics = load_eval_json(short_name, "unified-sft-v1")
    grpo_metrics = load_eval_json(short_name, "unified-grpo-v1")

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