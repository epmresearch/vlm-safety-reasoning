"""
Builds the Base vs SFT vs SFT+GSPO comparison table for a task.
Usage: python experiments/compare_results.py --tier 2b
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from core.constants import DEFAULT_MODEL_TIER
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
    cap = metrics.get("captioning", {})
    flat["BERTScore_F1"] = cap.get("bert_f1", None)
    flat["CLIPScore"] = cap.get("clip_score", None)
    
    # Grounding metrics
    grnd = metrics.get("grounding", {})
    flat["Grounding_IoU"] = grnd.get("mean_iou", None)
    flat["Grounding_F1"] = grnd.get("f1", None)
    
    # Violation metrics
    viol = metrics.get("violations", {})
    flat["Violation_F1"] = viol.get("f1", None)
    
    # Structural metrics
    struct = metrics.get("structural", {})
    flat["Valid_JSON_%"] = struct.get("valid_json_ratio", 0.0) * 100
    flat["Complete_Format_%"] = struct.get("complete_format_ratio", 0.0) * 100
    
    return flat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=DEFAULT_MODEL_TIER, help="Model tier (e.g., 2b, 4b, 8b)")
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