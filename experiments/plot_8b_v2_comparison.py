import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from math import pi
import numpy as np

# Set seaborn style for beautiful, academic paper-ready plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

RESULTS_DIR = Path("evaluation_results_v2")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PHASES = ["baseline", "sft", "grpo"]

def load_metrics():
    """Loads 8B metrics into a dictionary: data[phase]"""
    data = {}
    missing_files = []
    
    for phase in PHASES:
        folder_name = f"vo_{phase}_8b"
        metrics_file = RESULTS_DIR / folder_name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data[phase] = json.load(f)
        else:
            data[phase] = {}
            missing_files.append(str(metrics_file))
                
    if missing_files:
        print("Note: The following metrics files are missing:")
        for m in missing_files:
            print(f"  - {m}")
            
    return data

def plot_formatting(metrics_data):
    models = list(metrics_data.keys())
    if not models: return

    json_validity = [metrics_data[m].get("structural_json_validity_rate", 0) * 100 for m in models]
    schema_adherence = [metrics_data[m].get("structural_schema_adherence_rate", 0) * 100 for m in models]
            
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    rects1 = ax1.bar(x - width/2, json_validity, width, label='Valid JSON (%)', color=sns.color_palette("muted")[0])
    rects2 = ax1.bar(x + width/2, schema_adherence, width, label='Valid Schema (%)', color=sns.color_palette("muted")[1])
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Structural Formatting & Adherence by Model (8B v2)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 105)
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_formatting_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_safety_violations_f1(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = ["Rule 0", "Rule 1", "Rule 2", "Rule 3", "Rule 4", "Macro F1"]
    keys = [f"violation_identification_f1_rule_{i}" for i in range(5)] + ["violation_identification_f1_macro"]
    data = [{"Model": m.upper(), "Rule": rule, "F1 Score": metrics_data[m].get(key, 0)} for m in models for rule, key in zip(rules, keys)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(data), x="Rule", y="F1 Score", hue="Model", palette="pastel")
    plt.title("Safety Violation Identification (F1 Score)")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_safety_violations_f1.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_iou_conditioned_violations(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = ["Rule 1", "Rule 2", "Macro F1"]
    keys = ["violation_identification_iou_conditioned_f1_rule_1", "violation_identification_iou_conditioned_f1_rule_2", "violation_identification_iou_conditioned_f1_macro"]
    data = [{"Model": m.upper(), "Rule": rule, "F1 Score": metrics_data[m].get(key, 0)} for m in models for rule, key in zip(rules, keys)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(data), x="Rule", y="F1 Score", hue="Model", palette="dark")
    plt.title("Strict Safety Violation Identification (F1 Score Conditioned on IoU)")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_iou_conditioned_violations.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_violation_precision_recall(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Precision (Macro)", "Recall (Macro)", "IoU Cond. Precision", "IoU Cond. Recall"]
    keys = ["violation_identification_precision_macro", "violation_identification_recall_macro", "violation_identification_iou_conditioned_precision_macro", "violation_identification_iou_conditioned_recall_macro"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Violation Identification Precision vs Recall")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_violation_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_violation_grounding(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Mask IoU (Macro TN0)", "Greedy IoU (Macro TN0)", "Mask IoU (Micro Mean)"]
    keys = ["violation_grounding_mask_iou_macro_tn0", "violation_grounding_greedy_iou_macro_tn0", "violation_grounding_mask_iou_micro_mean"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="coolwarm")
    plt.title("Violation Grounding (IoU) Comparison")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_violation_grounding.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_reasoning_by_rule(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = ["Rule 1", "Rule 2", "Macro"]
    keys = ["reasoning_text_similarity_bertscore_f1_rule_1", "reasoning_text_similarity_bertscore_f1_rule_2", "reasoning_text_similarity_bertscore_f1_macro"]
    data = [{"Model": m.upper(), "Rule": rule, "BERTScore F1": metrics_data[m].get(key, 0)} for m in models for rule, key in zip(rules, keys)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(data), x="Rule", y="BERTScore F1", hue="Model", palette="husl")
    plt.title("Reasoning Explanation Quality by Rule (BERTScore)")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_reasoning_by_rule.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_summary(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    
    metrics = ["Format\nValid", "Grounding\nIoU", "Grounding\nF1", "Violation\nF1", "Reasoning\nBERT"]
    keys = [
        "structural_json_validity_rate",
        "grounding_mask_iou_all_macro_mean_tn0",
        "grounding_presence_f1_macro",
        "violation_identification_f1_macro",
        "reasoning_text_similarity_bertscore_f1_macro"
    ]
    valid_indices = [i for i, k in enumerate(keys) if any(m_data.get(k) is not None for m_data in metrics_data.values())]
    if not valid_indices: return
    metrics = [metrics[i] for i in valid_indices]
    keys = [keys[i] for i in valid_indices]
    
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metrics)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1.0)
    
    colors = sns.color_palette("deep", len(models))
    for idx, model in enumerate(models):
        values = [metrics_data[model].get(k, 0) for k in keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model.upper(), color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Radar Summary of Macro Metrics (8B v2)")
    plt.savefig(PLOTS_DIR / "11_radar_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_rule_prf1(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = ["rule_0", "rule_1", "rule_2", "rule_3", "rule_4"] 
    metrics = ["precision", "recall", "f1"]
    
    data = []
    for m in models:
        for rule in rules:
            for metric in metrics:
                key = f"violation_identification_{metric}_{rule}"
                val = metrics_data[m].get(key, 0)
                data.append({"Model": m.upper(), "Rule/Metric": f"{rule.replace('_',' ').title()} {metric.title()}", "Score": val})
                
    df = pd.DataFrame(data)
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df, x="Rule/Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Per-Rule Violation Precision/Recall/F1 Breakdown")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "16_per_rule_prf1.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_reasoning_metrics(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["BERTScore P", "BERTScore R", "BERTScore F1", "METEOR", "CIDEr", "CLIPScore"]
    keys = ["reasoning_text_similarity_bertscore_precision_macro", "reasoning_text_similarity_bertscore_recall_macro", "reasoning_text_similarity_bertscore_f1_macro", "reasoning_text_similarity_meteor_macro", "reasoning_text_similarity_ciderd_macro", "reasoning_text_similarity_clipscore_macro"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="magma")
    plt.title("Full Reasoning Explanations Metrics (Macro)")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "17_full_reasoning_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_overall_heatmap(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    
    selected_keys = [
        "structural_json_validity_rate",
        "violation_identification_f1_macro",
        "violation_identification_recall_macro",
        "violation_identification_precision_macro",
        "violation_grounding_mask_iou_macro_tn0",
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro"
    ]
    selected_keys = [k for k in selected_keys if any(m_data.get(k) is not None for m_data in metrics_data.values())]
    if not selected_keys: return
    
    data_matrix = []
    for k in selected_keys:
        row = [metrics_data[m].get(k, 0) for m in models]
        data_matrix.append(row)
        
    df = pd.DataFrame(data_matrix, columns=[m.upper() for m in models], index=[k.replace('_', ' ').title() for k in selected_keys])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title("Overall Metrics Heatmap (8B v2)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "18_overall_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_rule_heatmap(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = [1, 2, 3, 4]
    matrix = []
    row_labels = []
    
    for m in models:
        row = []
        for r in rules:
            key = f"violation_identification_f1_rule_{r}"
            row.append(metrics_data[m].get(key, 0))
        matrix.append(row)
        row_labels.append(m.upper())
            
    df = pd.DataFrame(matrix, index=row_labels, columns=[f"Rule {r}" for r in rules])
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title("Per-Rule Violation F1 Score Heatmap (8B v2)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_per_rule_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Initializing V2 8B Plotting Suite with all metrics...")
    data = load_metrics()
    
    if not data:
        print("Error: No data loaded.")
        return
        
    print("\nGenerating comprehensive plots...")
    
    plot_formatting(data)
    plot_safety_violations_f1(data)
    plot_iou_conditioned_violations(data)
    plot_violation_precision_recall(data)
    plot_violation_grounding(data)
    plot_reasoning_by_rule(data)
    plot_radar_summary(data)
    plot_per_rule_prf1(data)
    plot_full_reasoning_metrics(data)
    plot_overall_heatmap(data)
    plot_rule_heatmap(data)
    
    print(f"\nAll plots successfully saved to: {PLOTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
