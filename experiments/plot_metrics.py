import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

# Set seaborn style for beautiful, academic paper-ready plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

MODELS = ["baseline", "sft", "grpo"]
RESULTS_DIR = Path("evaluation_results")
PLOTS_DIR = RESULTS_DIR / "plots"

def load_metrics():
    metrics_data = {}
    repair_data = {}
    for model in MODELS:
        metrics_file = RESULTS_DIR / model / "metrics.json"
        repair_file = RESULTS_DIR / model / "repair_report.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data[model] = json.load(f)
        else:
            print(f"Warning: {metrics_file} not found.")
            
        if repair_file.exists():
            with open(repair_file, "r") as f:
                repair_data[model] = json.load(f)
        else:
            print(f"Warning: {repair_file} not found.")
            
    return metrics_data, repair_data

def plot_formatting(metrics_data, repair_data):
    models = list(metrics_data.keys())
    if not models:
        return

    json_validity = [metrics_data[m].get("structural_json_validity_rate", 0) * 100 for m in models]
    schema_adherence = [metrics_data[m].get("structural_schema_adherence_rate", 0) * 100 for m in models]
    
    # Repairs
    repairs = []
    for m in models:
        if m in repair_data:
            report = repair_data[m]
            repaired_count = report.get('total_repaired', report.get('samples_repaired', 0))
            if 'total_repaired' not in report and 'summary' in report:
                repaired_count = report['summary'].get('total_repaired', 0)
            repairs.append(repaired_count)
        else:
            repairs.append(0)
            
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    rects1 = ax1.bar(x - width/2, json_validity, width, label='Valid JSON (%)', color=sns.color_palette("muted")[0])
    rects2 = ax1.bar(x + width/2, schema_adherence, width, label='Valid Schema (%)', color=sns.color_palette("muted")[1])
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Structural Formatting & Adherence by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 105)
    
    # Add a twin axis for the repair count line
    if any(repairs):
        ax2 = ax1.twinx()
        ax2.plot(x, repairs, color=sns.color_palette("dark")[3], marker='o', linewidth=2, markersize=8, label='Repairs Needed')
        ax2.set_ylabel('Total Fallback Repairs Triggered')
        ax2.set_ylim(0, max(repairs) * 1.2 if max(repairs) > 0 else 10)
        ax2.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "formatting_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_grounding(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    
    categories = ["Excavator", "Rebar", "Worker w/ Hard Hat", "Mean (Macro)"]
    keys = [
        "grounding_mask_iou_all_micro_excavator",
        "grounding_mask_iou_all_micro_rebar",
        "grounding_mask_iou_all_micro_worker_with_white_hard_hat",
        "grounding_mask_iou_all_macro_mean_tn0"
    ]
    
    data = []
    for m in models:
        for cat, key in zip(categories, keys):
            val = metrics_data[m].get(key, 0)
            data.append({"Model": m.upper(), "Category": cat, "IoU Score": val})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Category", y="IoU Score", hue="Model", palette="deep")
    plt.title("Object Grounding (IoU) Comparison")
    plt.ylim(0, 1.0)
    plt.ylabel("Intersection over Union (IoU)")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "grounding_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_safety_violations(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    
    rules = ["Rule 0", "Rule 1", "Rule 2", "Rule 3", "Rule 4", "Macro F1"]
    keys = [
        "violation_identification_f1_rule_0",
        "violation_identification_f1_rule_1",
        "violation_identification_f1_rule_2",
        "violation_identification_f1_rule_3",
        "violation_identification_f1_rule_4",
        "violation_identification_f1_macro"
    ]
    
    data = []
    for m in models:
        for rule, key in zip(rules, keys):
            val = metrics_data[m].get(key, 0)
            data.append({"Model": m.upper(), "Rule": rule, "F1 Score": val})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Rule", y="F1 Score", hue="Model", palette="pastel")
    plt.title("Safety Violation Identification (F1 Score)")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "safety_violations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_reasoning_captioning(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    
    metrics = ["Reasoning BERTScore F1", "Reasoning Meteor", "Reasoning CLIPScore", "Captioning CLIPScore", "Captioning Meteor"]
    keys = [
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro",
        "reasoning_text_similarity_clipscore_macro",
        "captioning_clipscore",
        "captioning_meteor"
    ]
    
    data = []
    for m in models:
        for metric, key in zip(metrics, keys):
            val = metrics_data[m].get(key, 0)
            data.append({"Model": m.upper(), "Metric": metric, "Score": val})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette="colorblind")
    plt.title("Reasoning & Captioning Quality")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reasoning_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading metrics...")
    metrics_data, repair_data = load_metrics()
    if not metrics_data:
        print("No metrics found in evaluation_results/")
        return
        
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Plotting formatting metrics...")
    plot_formatting(metrics_data, repair_data)
    
    print("Plotting grounding metrics...")
    plot_grounding(metrics_data)
    
    print("Plotting safety violations...")
    plot_safety_violations(metrics_data)
    
    print("Plotting reasoning & captioning...")
    plot_reasoning_captioning(metrics_data)
    
    print(f"All plots saved to {PLOTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
