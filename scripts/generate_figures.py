"""
Generates comprehensive paper figures for the unified model outputs.
Usage: python scripts/generate_figures.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from core.io import get_drive_path, ensure_dir
from core.config import load_config
from models.model_loader import get_model_info

def load_metrics(model_short_name: str, variant: str) -> dict:
    path = get_drive_path("results", model_short_name, variant) / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def generate_radar_chart():
    # Placeholder for radar chart logic
    pass

def generate_bar_charts(out_dir):
    config = load_config()
    tier = config.get("active_tier", "2b")
    short_name = get_model_info(tier)["short_name"]
    base_metrics = load_metrics(short_name, "baseline")
    sft_metrics = load_metrics(short_name, "unified-sft-v1")
    
    if not base_metrics or not sft_metrics:
        print("Metrics not found. Run evaluation first.")
        return

    data = {
        "Metric": ["Valid JSON (%)", "Format Completeness (%)", "Caption BERTScore", "Grounding IoU", "Violation F1"],
        "Baseline": [
            base_metrics.get("structural", {}).get("valid_json_ratio", 0) * 100,
            base_metrics.get("structural", {}).get("complete_format_ratio", 0) * 100,
            base_metrics.get("captioning", {}).get("bert_f1", 0) * 100,
            base_metrics.get("grounding", {}).get("mean_iou", 0) * 100,
            base_metrics.get("violations", {}).get("f1", 0) * 100
        ],
        "SFT": [
            sft_metrics.get("structural", {}).get("valid_json_ratio", 0) * 100,
            sft_metrics.get("structural", {}).get("complete_format_ratio", 0) * 100,
            sft_metrics.get("captioning", {}).get("bert_f1", 0) * 100,
            sft_metrics.get("grounding", {}).get("mean_iou", 0) * 100,
            sft_metrics.get("violations", {}).get("f1", 0) * 100
        ]
    }
    
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars="Metric", var_name="Model", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title(f"Baseline vs SFT Performance ({tier.upper()})")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = out_dir / "performance_comparison.png"
    plt.savefig(out_path)
    print(f"Saved figure: {out_path}")

def main():
    out_dir = ensure_dir(get_drive_path("results", "figures"))
    generate_bar_charts(out_dir)

if __name__ == "__main__":
    main()