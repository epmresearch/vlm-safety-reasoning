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

RESULTS_DIR = Path("evaluation_results")

TIERS = ["2b", "4b", "8b"]
PHASES = ["baseline", "sft", "grpo"]

# Set by main() from --version; module-level so the plot_* functions (which
# only take `data`) don't all need a version parameter threaded through them.
VERSION = None
PLOTS_DIR = None
TASK = None

def load_all_metrics():
    """Loads all metrics into a nested dictionary: data[tier][phase]"""
    data = {tier: {} for tier in TIERS}
    missing_files = []

    from core.naming import results_dir_names
    for tier in TIERS:
        names = results_dir_names(TASK, tier, VERSION)
        for phase in PHASES:
            # Matches exactly how the pipeline creates the folders, via the shared
            # name builder rather than a hardcoded 'vo-' prefix.
            folder_name = names[phase]
            folder = RESULTS_DIR / folder_name
            folder.mkdir(parents=True, exist_ok=True) # Create empty folder if user hasn't yet
            
            metrics_file = folder / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data[tier][phase] = json.load(f)
            else:
                data[tier][phase] = {}
                missing_files.append(str(metrics_file))
                
    if missing_files:
        print("Note: The following metrics files are missing (showing empty/0 in plots):")
        for m in missing_files:
            print(f"  - {m}")
            
    return data

def plot_grouped_bar(data, metric_key, title, ylabel, filename, y_min=0, y_max=1.0):
    rows = []
    has_data = False
    for tier in TIERS:
        for phase in PHASES:
            val = data[tier][phase].get(metric_key, 0)
            if val > 0: has_data = True
            rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Score": val})
            
    if not has_data:
        print(f"Skipping {filename}: No data found for {metric_key}")
        return
        
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model Size", y="Score", hue="Phase", palette="deep")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(y_min, y_max)
    plt.legend(title="Phase", loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scaling_line(data):
    rows = []
    for tier in TIERS:
        for phase in PHASES:
            val = data[tier][phase].get("violation_identification_f1_macro", 0)
            rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Macro F1": val})
            
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Model Size", y="Macro F1", hue="Phase", style="Phase", 
                 markers=True, markersize=10, linewidth=2.5, palette="dark")
    plt.title("Scaling Behavior: Safety Violation F1 by Model Size")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1.0)
    plt.legend(title="Phase", loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_scale_vs_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_rule_heatmap(data):
    rules = [1, 2, 3, 4]
    matrix = []
    row_labels = []
    
    for tier in TIERS:
        for phase in PHASES:
            row = []
            for r in rules:
                key = f"violation_identification_f1_rule_{r}"
                row.append(data[tier][phase].get(key, 0))
            matrix.append(row)
            row_labels.append(f"{tier.upper()} {phase.upper()}")
            
    df = pd.DataFrame(matrix, index=row_labels, columns=[f"Rule {r}" for r in rules])
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title("Per-Rule Violation F1 Score Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_per_rule_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_charts(data):
    metrics = ["Format\nValidity", "Violation\nF1", "Strict\nIoU F1", "Grounding\nIoU", "Reasoning\nBERT"]
    keys = [
        "structural_schema_adherence_rate",
        "violation_identification_f1_macro",
        "violation_identification_iou_conditioned_f1_macro",
        "violation_grounding_mask_iou_macro_tn0",
        "reasoning_text_similarity_bertscore_f1_macro"
    ]
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    for tier in TIERS:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], metrics)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
        plt.ylim(0, 1.0)
        
        colors = sns.color_palette("deep", len(PHASES))
        for idx, phase in enumerate(PHASES):
            values = [data[tier][phase].get(k, 0) for k in keys]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=phase.upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f"Safety Radar Summary - {tier.upper()} Model")
        plt.savefig(PLOTS_DIR / f"07_radar_summary_{tier}.png", dpi=300, bbox_inches='tight')
        plt.close('all')

def plot_iou_conditioned_violations(data):
    rows = []
    has_data = False
    for tier in TIERS:
        for phase in PHASES:
            for rule in [1, 2, "macro"]:
                key = f"violation_identification_iou_conditioned_f1_rule_{rule}" if rule != "macro" else "violation_identification_iou_conditioned_f1_macro"
                val = data[tier][phase].get(key, 0)
                if val > 0: has_data = True
                rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Rule": f"Rule {rule}" if rule != "macro" else "Macro F1", "F1 Score": val})
                
    if not has_data: return
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df, x="Rule", y="F1 Score", hue="Phase", col="Model Size", kind="bar", palette="dark", height=5, aspect=1)
    g.set(ylim=(0, 1.0))
    plt.savefig(PLOTS_DIR / "08_strict_iou_conditioned_f1.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_violation_precision_recall(data):
    rows = []
    has_data = False
    for tier in TIERS:
        for phase in PHASES:
            metrics = ["Precision (Macro)", "Recall (Macro)", "IoU Cond. Precision", "IoU Cond. Recall"]
            keys = ["violation_identification_precision_macro", "violation_identification_recall_macro", "violation_identification_iou_conditioned_precision_macro", "violation_identification_iou_conditioned_recall_macro"]
            for metric, key in zip(metrics, keys):
                val = data[tier][phase].get(key, 0)
                if val > 0: has_data = True
                rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Metric": metric, "Score": val})
                
    if not has_data: return
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df, x="Metric", y="Score", hue="Phase", col="Model Size", kind="bar", palette="Set2", height=5, aspect=1.2)
    g.set(ylim=(0, 1.0))
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.savefig(PLOTS_DIR / "09_precision_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_per_rule_prf1(data):
    rows = []
    has_data = False
    rules = [1, 2, 3, 4]
    metrics = ["precision", "recall", "f1"]
    for tier in TIERS:
        for phase in PHASES:
            for rule in rules:
                for metric in metrics:
                    key = f"violation_identification_{metric}_rule_{rule}"
                    val = data[tier][phase].get(key, 0)
                    if val > 0: has_data = True
                    rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Rule/Metric": f"R{rule} {metric.title()}", "Score": val})
                    
    if not has_data: return
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df, x="Rule/Metric", y="Score", hue="Phase", row="Model Size", kind="bar", palette="tab10", height=4, aspect=3)
    g.set(ylim=(0, 1.0))
    plt.savefig(PLOTS_DIR / "10_per_rule_precision_recall_f1.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_reasoning_by_rule(data):
    rows = []
    has_data = False
    for tier in TIERS:
        for phase in PHASES:
            for rule in [1, 2, "macro"]:
                key = f"reasoning_text_similarity_bertscore_f1_rule_{rule}" if rule != "macro" else "reasoning_text_similarity_bertscore_f1_macro"
                val = data[tier][phase].get(key, 0)
                if val > 0: has_data = True
                rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Rule": f"Rule {rule}" if rule != "macro" else "Macro F1", "BERTScore F1": val})
                
    if not has_data: return
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df, x="Rule", y="BERTScore F1", hue="Phase", col="Model Size", kind="bar", palette="husl", height=5, aspect=1)
    g.set(ylim=(0, 1.0))
    plt.savefig(PLOTS_DIR / "11_reasoning_bertscore_by_rule.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_full_reasoning_metrics(data):
    rows = []
    has_data = False
    metrics = ["BERT P", "BERT R", "BERT F1", "METEOR", "CIDEr", "CLIP"]
    keys = ["reasoning_text_similarity_bertscore_precision_macro", "reasoning_text_similarity_bertscore_recall_macro", "reasoning_text_similarity_bertscore_f1_macro", "reasoning_text_similarity_meteor_macro", "reasoning_text_similarity_ciderd_macro", "reasoning_text_similarity_clipscore_macro"]
    for tier in TIERS:
        for phase in PHASES:
            for metric, key in zip(metrics, keys):
                val = data[tier][phase].get(key, 0)
                if val > 0: has_data = True
                rows.append({"Model Size": tier.upper(), "Phase": phase.upper(), "Metric": metric, "Score": val})
                
    if not has_data: return
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df, x="Metric", y="Score", hue="Phase", col="Model Size", kind="bar", palette="magma", height=5, aspect=1.2)
    g.set(ylim=(0, 1.0))
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.savefig(PLOTS_DIR / "12_full_reasoning_linguistics.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_master_heatmap(data):
    columns = [f"{tier.upper()} {phase.upper()}" for tier in TIERS for phase in PHASES]
    selected_keys = [
        "structural_schema_adherence_rate",
        "violation_identification_f1_macro",
        "violation_identification_recall_macro",
        "violation_identification_precision_macro",
        "violation_identification_iou_conditioned_f1_macro",
        "violation_grounding_mask_iou_macro_tn0",
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro",
        "reasoning_text_similarity_ciderd_macro"
    ]
    
    has_data = False
    for k in selected_keys:
        if any(data[tier][phase].get(k, 0) > 0 for tier in TIERS for phase in PHASES):
            has_data = True
            break
            
    if not has_data: return
    
    matrix = []
    for k in selected_keys:
        row = [data[tier][phase].get(k, 0) for tier in TIERS for phase in PHASES]
        matrix.append(row)
        
    df = pd.DataFrame(matrix, columns=columns, index=[k.replace('_', ' ').title() for k in selected_keys])
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title("Master Metrics Heatmap Across All Tiers & Phases")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_master_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close('all')


def main():
    import argparse
    from core.constants import VALID_TASKS
    from core.naming import task_prefix
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", required=True,
        help="Run version tag of the results to plot (e.g. v1, v2). Determines "
             "both the result folders read (<prefix>-<phase>-<tier>-<version>) and "
             "the output directory (plots_<prefix>_<version>)."
    )
    parser.add_argument(
        "--task", default="violations_only", choices=VALID_TASKS,
        help="Task whose results to plot. Every plot in this suite is "
             "violation-specific, so only violation-capable tasks are meaningful "
             "here; use experiments/plot_metrics.py for the capability-gated suite.",
    )
    args = parser.parse_args()

    global VERSION, PLOTS_DIR, TASK
    VERSION = args.version
    TASK = args.task
    PLOTS_DIR = RESULTS_DIR / f"plots_{task_prefix(TASK)}_{VERSION}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    from core.tasks import CAP_VIOLATIONS, task_has
    if not task_has(TASK, CAP_VIOLATIONS):
        raise SystemExit(
            f"plot_metrics_vo.py only plots violation metrics, which task {TASK!r} "
            "does not produce. Use experiments/plot_metrics.py --task "
            f"{TASK} instead."
        )

    print(f"Initializing {task_prefix(TASK)} 3x3 Evaluation Suite (version={VERSION})...")
    data = load_all_metrics()
    
    print("\nGenerating comprehensive cross-model and cross-phase plots...")
    
    # 1. Format Validity
    plot_grouped_bar(data, "structural_schema_adherence_rate", 
                     "Format & Schema Validity Rate", "Percentage (%)", 
                     "01_format_validity.png")
                     
    # 2. Violation Detection F1
    plot_grouped_bar(data, "violation_identification_f1_macro", 
                     "Safety Violation Identification (Macro F1)", "F1 Score", 
                     "02_violation_f1.png")
                     
    # 3. Violation Grounding IoU
    plot_grouped_bar(data, "violation_grounding_mask_iou_macro_tn0", 
                     "Violation Grounding (Macro Mask IoU)", "IoU Score", 
                     "03_grounding_iou.png")
                     
    # 4. Reasoning Explanation Quality
    plot_grouped_bar(data, "reasoning_text_similarity_bertscore_f1_macro", 
                     "Safety Reasoning Quality (BERTScore F1)", "BERTScore", 
                     "04_reasoning_bertscore.png")
                     
    # 5. Scale vs Performance
    plot_scaling_line(data)
    
    # 6. Heatmap
    plot_rule_heatmap(data)
    
    # 7. Radar Charts
    plot_radar_charts(data)
    
    # 8. Strict IoU Conditioned F1
    plot_iou_conditioned_violations(data)
    
    # 9. Precision vs Recall Breakdown
    plot_violation_precision_recall(data)
    
    # 10. Per Rule PRF1
    plot_per_rule_prf1(data)
    
    # 11. Reasoning BERTScore by Rule
    plot_reasoning_by_rule(data)
    
    # 12. Full Reasoning Linguistics (BERT, METEOR, CIDEr)
    plot_full_reasoning_metrics(data)
    
    # 13. Master Heatmap Matrix
    plot_master_heatmap(data)
    
    print(f"\nAll 13 plots saved to: {PLOTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
