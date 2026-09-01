import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from math import pi

# Set seaborn style for beautiful, academic paper-ready plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

MODELS = ["baseline", "sft", "grpo"]
RESULTS_DIR = Path("evaluation_results")
PLOTS_DIR = RESULTS_DIR / "plots"

def load_metrics(task: str = "unified", tier: str = "", version: str = ""):
    metrics_data = {}
    repair_data = {}

    suffix = f"_{tier}" if tier and tier != "2b" else ""

    # Folder names come from core.naming.results_dir_names — the same helper the
    # comparison table uses, so the two can never disagree. The old task ==
    # "unified" branch existed only for the legacy unprefixed baseline folder,
    # which is gone.
    from core.naming import results_dir_names
    models = ["baseline", "sft", "grpo"]
    names = results_dir_names(task, tier, version)
    variants = [names["baseline"], names["sft"], names["grpo"]]
        
    for model_key, variant in zip(models, variants):
        metrics_file = RESULTS_DIR / variant / "metrics.json"
        repair_file = RESULTS_DIR / variant / "repair_report.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data[model_key] = json.load(f)
        else:
            print(f"Warning: {metrics_file} not found.")
            
        if repair_file.exists():
            with open(repair_file, "r") as f:
                repair_data[model_key] = json.load(f)
        else:
            print(f"Warning: {repair_file} not found.")
            
    return metrics_data, repair_data

def plot_formatting(metrics_data, repair_data):
    models = list(metrics_data.keys())
    if not models: return

    json_validity = [metrics_data[m].get("structural_json_validity_rate", 0) * 100 for m in models]
    schema_adherence = [metrics_data[m].get("structural_schema_adherence_rate", 0) * 100 for m in models]
    
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
    
    if any(repairs):
        ax2 = ax1.twinx()
        ax2.plot(x, repairs, color=sns.color_palette("dark")[3], marker='o', linewidth=2, markersize=8, label='Repairs Needed')
        ax2.set_ylabel('Total Fallback Repairs Triggered')
        ax2.set_ylim(0, max(repairs) * 1.2 if max(repairs) > 0 else 10)
        ax2.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_formatting_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_grounding_iou(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    categories = ["Excavator", "Rebar", "Worker w/ Hard Hat", "Mean (Macro)"]
    keys = ["grounding_mask_iou_all_micro_excavator", "grounding_mask_iou_all_micro_rebar", "grounding_mask_iou_all_micro_worker_with_white_hard_hat", "grounding_mask_iou_all_macro_mean_tn0"]
    data = [{"Model": m.upper(), "Category": cat, "IoU Score": metrics_data[m].get(key, 0)} for m in models for cat, key in zip(categories, keys)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(data), x="Category", y="IoU Score", hue="Model", palette="deep")
    plt.title("Object Grounding (IoU) Comparison")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_grounding_iou.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_grounding_presence(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    categories = ["Excavator", "Rebar", "Worker w/ Hard Hat", "Mean (Macro)"]
    keys = ["grounding_presence_f1_excavator", "grounding_presence_f1_rebar", "grounding_presence_f1_worker_with_white_hard_hat", "grounding_presence_f1_macro"]
    data = [{"Model": m.upper(), "Category": cat, "F1 Score": metrics_data[m].get(key, 0)} for m in models for cat, key in zip(categories, keys)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(data), x="Category", y="F1 Score", hue="Model", palette="muted")
    plt.title("Object Grounding Presence (Detection F1 Score)")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_grounding_presence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_grounding_tn0_vs_tn1(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Absent Objects (TN0)", "Present Objects (TN1)"]
    keys = ["grounding_mask_iou_all_macro_mean_tn0", "grounding_mask_iou_all_macro_mean_tn1"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="Set3")
    plt.title("Grounding IoU: Absent vs Present Objects")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_grounding_tn0_vs_tn1.png", dpi=300, bbox_inches='tight')
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
    
def plot_reasoning_vs_captioning(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Reasoning BERTScore", "Captioning BERTScore", "Reasoning CLIPScore", "Captioning CLIPScore"]
    keys = ["reasoning_text_similarity_bertscore_f1_macro", "captioning_bertscore_f1", "reasoning_text_similarity_clipscore_macro", "captioning_clipscore"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="colorblind")
    plt.title("General Captioning vs Safety Reasoning Explanations")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_reasoning_vs_captioning.png", dpi=300, bbox_inches='tight')
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

# ----- NEW 9 CHARTS -----

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
    # Filter out missing keys to gracefully support violations_only task
    valid_indices = [i for i, k in enumerate(keys) if any(m_data.get(k) is not None for m_data in metrics_data.values())]
    if not valid_indices: return
    metrics = [metrics[i] for i in valid_indices]
    keys = [keys[i] for i in valid_indices]
    
    # number of variable
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
    plt.title("Radar Summary of Macro Metrics")
    plt.savefig(PLOTS_DIR / "11_radar_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_captioning_breakdown(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["BERTScore P", "BERTScore R", "BERTScore F1", "METEOR", "CIDEr-D", "CLIPScore"]
    keys = ["captioning_bertscore_precision", "captioning_bertscore_recall", "captioning_bertscore_f1", "captioning_meteor", "captioning_ciderd", "captioning_clipscore"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="Set1")
    plt.title("Full Captioning Metrics Breakdown")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_full_captioning_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_grounding_counts(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    classes = ["excavator", "rebar", "worker_with_white_hard_hat"]
    stat_types = ["true_positives", "false_positives", "false_negatives"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = sns.color_palette("muted", len(models))
    
    for i, cls in enumerate(classes):
        data = []
        for m in models:
            for stat in stat_types:
                val = metrics_data[m].get(f"grounding_{stat}_count_{cls}", 0)
                data.append({"Model": m.upper(), "Count Type": stat.replace("_", " ").title(), "Count": val})
        df = pd.DataFrame(data)
        sns.barplot(data=df, x="Count Type", y="Count", hue="Model", ax=axes[i], palette="muted")
        axes[i].set_title(cls.replace("_", " ").title())
        axes[i].set_xlabel("")
        if i > 0: axes[i].get_legend().remove()
    
    plt.suptitle("Grounding TP/FP/FN Counts per Class")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_grounding_counts.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_mask_vs_greedy_iou(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    classes = ["excavator", "rebar", "worker_with_white_hard_hat"]
    
    data = []
    for m in models:
        for cls in classes:
            mask_val = metrics_data[m].get(f"grounding_mask_iou_all_micro_{cls}", 0)
            greedy_val = metrics_data[m].get(f"grounding_greedy_iou_all_micro_{cls}", 0)
            data.append({"Model": m.upper(), "Class": cls.replace("_", " ").title(), "IoU Type": "Mask IoU", "Score": mask_val})
            data.append({"Model": m.upper(), "Class": cls.replace("_", " ").title(), "IoU Type": "Greedy IoU", "Score": greedy_val})
            
    df = pd.DataFrame(data)
    plt.figure(figsize=(14, 6))
    sns.catplot(data=df, x="Class", y="Score", hue="Model", col="IoU Type", kind="bar", height=5, aspect=1.2, palette="deep")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_mask_vs_greedy_iou.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_exist_vs_all_iou(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Mask IoU (Exist Macro)", "Mask IoU (All Macro)"]
    keys = ["grounding_mask_iou_exist_macro_mean", "grounding_mask_iou_all_macro_mean_tn1"]
    data = [{"Model": m.upper(), "Metric": metric, "Score": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Grounding IoU: Objects that Exist vs All Images")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_exist_vs_all_iou.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_rule_prf1(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    rules = ["rule_0", "rule_1", "rule_2"] # rule 3/4 usually zero
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
    
    # Select a diverse set of top 15-20 metrics
    selected_keys = [
        "structural_json_validity_rate",
        "captioning_bertscore_f1",
        "captioning_clipscore",
        "grounding_mask_iou_all_macro_mean_tn0",
        "grounding_presence_f1_macro",
        "violation_identification_f1_macro",
        "violation_identification_recall_macro",
        "violation_identification_precision_macro",
        "violation_grounding_mask_iou_macro_tn0",
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro"
    ]
    # Filter out missing keys for task awareness
    selected_keys = [k for k in selected_keys if any(m_data.get(k) is not None for m_data in metrics_data.values())]
    if not selected_keys: return
    
    data_matrix = []
    for k in selected_keys:
        row = [metrics_data[m].get(k, 0) for m in models]
        data_matrix.append(row)
        
    df = pd.DataFrame(data_matrix, columns=[m.upper() for m in models], index=[k.replace('_', ' ').title() for k in selected_keys])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title("Overall Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "18_overall_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_caption_word_stats(metrics_data):
    models = list(metrics_data.keys())
    if not models: return
    metrics = ["Avg Words (Captioning)", "Avg Words (Reasoning)"]
    keys = ["captioning_avg_words_per_caption", "reasoning_text_similarity_avg_words_per_caption_micro"]
    data = [{"Model": m.upper(), "Metric": metric, "Word Count": metrics_data[m].get(key, 0)} for m in models for metric, key in zip(metrics, keys)]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Word Count", hue="Model", palette="Pastel1")
    plt.title("Average Word Count per Caption/Explanation")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_caption_word_stats.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    from core.constants import VALID_TASKS
    parser.add_argument("--task", default="unified", choices=VALID_TASKS, help="Task name")
    parser.add_argument("--tier", default="", help="Model tier (e.g., 2b, 4b, 8b). Empty defaults to 2b folder structure.")
    parser.add_argument(
        "--version", required=True,
        help="Run version tag of the results to plot (e.g. v4, v5)."
    )
    args = parser.parse_args()

    global PLOTS_DIR
    if args.tier and args.tier != "2b":
        PLOTS_DIR = RESULTS_DIR / f"plots_{args.tier}"

    print(f"Loading metrics for tier: {args.tier or '2b'}...")
    metrics_data, repair_data = load_metrics(args.task, args.tier, args.version)
    if not metrics_data:
        print("No metrics found in evaluation_results/")
        return
        
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Comprehensive Plot Suite in {PLOTS_DIR}...")

    # Which plot families are meaningful is decided by the task's capabilities, not
    # by a task-name comparison — see core/tasks.py. The old `!= "violations_only"`
    # gates would have drawn empty grounding plots for caption_only and empty
    # captioning plots for object_only, and drew violation plots for both.
    from core.tasks import CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS, task_has
    has_caption = task_has(args.task, CAP_CAPTION)
    has_objects = task_has(args.task, CAP_OBJECTS)
    has_violations = task_has(args.task, CAP_VIOLATIONS)

    plot_formatting(metrics_data, repair_data)

    if has_objects:
        plot_grounding_iou(metrics_data)
        plot_grounding_presence(metrics_data)
        plot_grounding_tn0_vs_tn1(metrics_data)
        plot_grounding_counts(metrics_data)
        plot_mask_vs_greedy_iou(metrics_data)
        plot_exist_vs_all_iou(metrics_data)

    if has_violations:
        plot_safety_violations_f1(metrics_data)
        plot_iou_conditioned_violations(metrics_data)
        plot_violation_precision_recall(metrics_data)
        plot_violation_grounding(metrics_data)
        plot_reasoning_by_rule(metrics_data)

    if has_caption:
        plot_full_captioning_breakdown(metrics_data)
        plot_caption_word_stats(metrics_data)

    # Compares violation-reason quality against caption quality — needs both.
    if has_caption and has_violations:
        plot_reasoning_vs_captioning(metrics_data)

    plot_radar_summary(metrics_data)
    plot_per_rule_prf1(metrics_data)
    plot_full_reasoning_metrics(metrics_data)
    plot_overall_heatmap(metrics_data)
    
    print(f"Plots successfully saved to {PLOTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
