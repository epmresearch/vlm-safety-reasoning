"""
Generate deep CSV comparison tables for Violations-Only (VO) v4 pipeline.

For each model size (2B, 4B, 8B), produces a single CSV file with:
  - All metric values for Baseline, SFT, and GRPO
  - Absolute delta (SFT - Baseline) and (GRPO - Baseline)
  - Percentage change (SFT vs Baseline) and (GRPO vs Baseline)
  - A "Best Phase" column highlighting the winner

Also produces a combined "master" CSV with all 9 runs side by side.

Usage:
    python experiments/generate_comparison_csv.py
"""

import json
import csv
import os
from pathlib import Path

RESULTS_DIR = Path("evaluation_results")
CSV_DIR = RESULTS_DIR / "csv_comparisons_v4"
CSV_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["2b", "4b", "8b"]
PHASES = ["baseline", "sft", "grpo"]

# ---------------------------------------------------------------------------
# Metric groups — organised by evaluation dimension for readability
# ---------------------------------------------------------------------------
METRIC_GROUPS = {
    "Structural Validity": [
        ("JSON Validity Rate", "structural_json_validity_rate"),
        ("Schema Adherence Rate", "structural_schema_adherence_rate"),
        ("Valid JSON Count", "structural_valid_json_count"),
        ("Valid Schema Count", "structural_valid_schema_count"),
        ("Total Samples", "structural_total_samples_count"),
    ],
    "Violation Identification — Micro": [
        ("Precision (Micro)", "violation_identification_precision_micro"),
        ("Recall (Micro)", "violation_identification_recall_micro"),
        ("F1 (Micro)", "violation_identification_f1_micro"),
    ],
    "Violation Identification — Per Rule": [
        ("Rule 0 (Safe) Precision", "violation_identification_precision_rule_0"),
        ("Rule 0 (Safe) Recall", "violation_identification_recall_rule_0"),
        ("Rule 0 (Safe) F1", "violation_identification_f1_rule_0"),
        ("Rule 1 (PPE) Precision", "violation_identification_precision_rule_1"),
        ("Rule 1 (PPE) Recall", "violation_identification_recall_rule_1"),
        ("Rule 1 (PPE) F1", "violation_identification_f1_rule_1"),
        ("Rule 2 (Harness) Precision", "violation_identification_precision_rule_2"),
        ("Rule 2 (Harness) Recall", "violation_identification_recall_rule_2"),
        ("Rule 2 (Harness) F1", "violation_identification_f1_rule_2"),
        ("Rule 3 (Edge) Precision", "violation_identification_precision_rule_3"),
        ("Rule 3 (Edge) Recall", "violation_identification_recall_rule_3"),
        ("Rule 3 (Edge) F1", "violation_identification_f1_rule_3"),
        ("Rule 4 (Blind Spot) Precision", "violation_identification_precision_rule_4"),
        ("Rule 4 (Blind Spot) Recall", "violation_identification_recall_rule_4"),
        ("Rule 4 (Blind Spot) F1", "violation_identification_f1_rule_4"),
    ],
    "Violation Identification — Macro": [
        ("Precision (Macro)", "violation_identification_precision_macro"),
        ("Recall (Macro)", "violation_identification_recall_macro"),
        ("F1 (Macro)", "violation_identification_f1_macro"),
    ],
    "Strict IoU-Conditioned Violation ID — Micro": [
        ("IoU Threshold", "violation_identification_iou_threshold"),
        ("Strict Precision (Micro)", "violation_identification_iou_conditioned_precision_micro"),
        ("Strict Recall (Micro)", "violation_identification_iou_conditioned_recall_micro"),
        ("Strict F1 (Micro)", "violation_identification_iou_conditioned_f1_micro"),
    ],
    "Strict IoU-Conditioned Violation ID — Per Rule": [
        ("Strict Rule 1 Precision", "violation_identification_iou_conditioned_precision_rule_1"),
        ("Strict Rule 1 Recall", "violation_identification_iou_conditioned_recall_rule_1"),
        ("Strict Rule 1 F1", "violation_identification_iou_conditioned_f1_rule_1"),
        ("Strict Rule 2 Precision", "violation_identification_iou_conditioned_precision_rule_2"),
        ("Strict Rule 2 Recall", "violation_identification_iou_conditioned_recall_rule_2"),
        ("Strict Rule 2 F1", "violation_identification_iou_conditioned_f1_rule_2"),
        ("Strict Rule 3 Precision", "violation_identification_iou_conditioned_precision_rule_3"),
        ("Strict Rule 3 Recall", "violation_identification_iou_conditioned_recall_rule_3"),
        ("Strict Rule 3 F1", "violation_identification_iou_conditioned_f1_rule_3"),
        ("Strict Rule 4 Precision", "violation_identification_iou_conditioned_precision_rule_4"),
        ("Strict Rule 4 Recall", "violation_identification_iou_conditioned_recall_rule_4"),
        ("Strict Rule 4 F1", "violation_identification_iou_conditioned_f1_rule_4"),
    ],
    "Strict IoU-Conditioned Violation ID — Macro": [
        ("Strict Precision (Macro)", "violation_identification_iou_conditioned_precision_macro"),
        ("Strict Recall (Macro)", "violation_identification_iou_conditioned_recall_macro"),
        ("Strict F1 (Macro)", "violation_identification_iou_conditioned_f1_macro"),
    ],
    "Violation Grounding — Mask IoU": [
        ("Rule 1 Mask IoU (TN0)", "violation_grounding_mask_iou_rule_1_tn0"),
        ("Rule 1 Mask IoU (TN1)", "violation_grounding_mask_iou_rule_1_tn1"),
        ("Rule 1 Mask IoU (Micro)", "violation_grounding_mask_iou_rule_1_micro"),
        ("Rule 2 Mask IoU (TN0)", "violation_grounding_mask_iou_rule_2_tn0"),
        ("Rule 2 Mask IoU (TN1)", "violation_grounding_mask_iou_rule_2_tn1"),
        ("Rule 2 Mask IoU (Micro)", "violation_grounding_mask_iou_rule_2_micro"),
        ("Rule 3 Mask IoU (TN0)", "violation_grounding_mask_iou_rule_3_tn0"),
        ("Rule 3 Mask IoU (TN1)", "violation_grounding_mask_iou_rule_3_tn1"),
        ("Rule 3 Mask IoU (Micro)", "violation_grounding_mask_iou_rule_3_micro"),
        ("Rule 4 Mask IoU (TN0)", "violation_grounding_mask_iou_rule_4_tn0"),
        ("Rule 4 Mask IoU (TN1)", "violation_grounding_mask_iou_rule_4_tn1"),
        ("Rule 4 Mask IoU (Micro)", "violation_grounding_mask_iou_rule_4_micro"),
        ("Macro Mask IoU (TN0)", "violation_grounding_mask_iou_macro_tn0"),
        ("Macro Mask IoU (TN1)", "violation_grounding_mask_iou_macro_tn1"),
        ("Micro Mean Mask IoU", "violation_grounding_mask_iou_micro_mean"),
    ],
    "Violation Grounding — Greedy IoU": [
        ("Rule 1 Greedy IoU (TN0)", "violation_grounding_greedy_iou_rule_1_tn0"),
        ("Rule 2 Greedy IoU (TN0)", "violation_grounding_greedy_iou_rule_2_tn0"),
        ("Rule 3 Greedy IoU (TN0)", "violation_grounding_greedy_iou_rule_3_tn0"),
        ("Rule 4 Greedy IoU (TN0)", "violation_grounding_greedy_iou_rule_4_tn0"),
        ("Macro Greedy IoU (TN0)", "violation_grounding_greedy_iou_macro_tn0"),
        ("Micro Mean Greedy IoU", "violation_grounding_greedy_iou_micro_mean"),
    ],
    "Violation Grounding — TN Counts": [
        ("Rule 1 TN Count", "violation_grounding_tn_count_rule_1"),
        ("Rule 2 TN Count", "violation_grounding_tn_count_rule_2"),
        ("Rule 3 TN Count", "violation_grounding_tn_count_rule_3"),
        ("Rule 4 TN Count", "violation_grounding_tn_count_rule_4"),
    ],
    "Reasoning Quality — Micro Aggregates": [
        ("BERTScore Precision (Micro)", "reasoning_text_similarity_bertscore_precision_micro"),
        ("BERTScore Recall (Micro)", "reasoning_text_similarity_bertscore_recall_micro"),
        ("BERTScore F1 (Micro)", "reasoning_text_similarity_bertscore_f1_micro"),
        ("METEOR (Micro)", "reasoning_text_similarity_meteor_micro"),
        ("CIDEr-D (Micro)", "reasoning_text_similarity_ciderd_micro"),
        ("CLIPScore (Micro)", "reasoning_text_similarity_clipscore_micro"),
        ("Avg Words/Explanation (Micro)", "reasoning_text_similarity_avg_words_per_caption_micro"),
        ("Min Words (Micro)", "reasoning_text_similarity_min_words_micro"),
        ("Max Words (Micro)", "reasoning_text_similarity_max_words_micro"),
    ],
    "Reasoning Quality — Per Rule BERTScore F1": [
        ("Rule 1 BERTScore F1", "reasoning_text_similarity_bertscore_f1_rule_1"),
        ("Rule 2 BERTScore F1", "reasoning_text_similarity_bertscore_f1_rule_2"),
        ("Rule 3 BERTScore F1", "reasoning_text_similarity_bertscore_f1_rule_3"),
        ("Rule 4 BERTScore F1", "reasoning_text_similarity_bertscore_f1_rule_4"),
    ],
    "Reasoning Quality — Per Rule METEOR": [
        ("Rule 1 METEOR", "reasoning_text_similarity_meteor_rule_1"),
        ("Rule 2 METEOR", "reasoning_text_similarity_meteor_rule_2"),
        ("Rule 3 METEOR", "reasoning_text_similarity_meteor_rule_3"),
        ("Rule 4 METEOR", "reasoning_text_similarity_meteor_rule_4"),
    ],
    "Reasoning Quality — Per Rule CIDEr-D": [
        ("Rule 1 CIDEr-D", "reasoning_text_similarity_ciderd_rule_1"),
        ("Rule 2 CIDEr-D", "reasoning_text_similarity_ciderd_rule_2"),
        ("Rule 3 CIDEr-D", "reasoning_text_similarity_ciderd_rule_3"),
        ("Rule 4 CIDEr-D", "reasoning_text_similarity_ciderd_rule_4"),
    ],
    "Reasoning Quality — Per Rule CLIPScore": [
        ("Rule 1 CLIPScore", "reasoning_text_similarity_clipscore_rule_1"),
        ("Rule 2 CLIPScore", "reasoning_text_similarity_clipscore_rule_2"),
        ("Rule 3 CLIPScore", "reasoning_text_similarity_clipscore_rule_3"),
        ("Rule 4 CLIPScore", "reasoning_text_similarity_clipscore_rule_4"),
    ],
    "Reasoning Quality — Macro Aggregates": [
        ("BERTScore F1 (Macro)", "reasoning_text_similarity_bertscore_f1_macro"),
        ("METEOR (Macro)", "reasoning_text_similarity_meteor_macro"),
        ("CIDEr-D (Macro)", "reasoning_text_similarity_ciderd_macro"),
        ("CLIPScore (Macro)", "reasoning_text_similarity_clipscore_macro"),
    ],
    "Reasoning Quality — Per Rule Avg Words": [
        ("Rule 1 Avg Words", "reasoning_text_similarity_avg_words_per_caption_rule_1"),
        ("Rule 2 Avg Words", "reasoning_text_similarity_avg_words_per_caption_rule_2"),
        ("Rule 3 Avg Words", "reasoning_text_similarity_avg_words_per_caption_rule_3"),
        ("Rule 4 Avg Words", "reasoning_text_similarity_avg_words_per_caption_rule_4"),
    ],
}


def load_metrics(tier, phase):
    """Load a single metrics.json file for a given tier and phase."""
    folder = RESULTS_DIR / f"vo-{phase}-{tier}-v4"
    metrics_file = folder / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    else:
        print(f"  WARNING: {metrics_file} not found — will show 0 for all metrics.")
        return {}


def fmt_val(val):
    """Format a numeric value for the CSV cell."""
    if val is None:
        return "N/A"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if abs(val) >= 10:
            return f"{val:.4f}"
        return f"{val:.6f}"
    return str(val)


def fmt_delta(delta):
    """Format a delta with sign."""
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    if isinstance(delta, int):
        return f"{sign}{delta}"
    return f"{sign}{delta:.6f}"


def fmt_pct(baseline_val, new_val):
    """Calculate and format percentage change."""
    if baseline_val is None or new_val is None:
        return "N/A"
    if baseline_val == 0:
        if new_val == 0:
            return "0.00%"
        return "+∞%" if new_val > 0 else "-∞%"
    pct = ((new_val - baseline_val) / abs(baseline_val)) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def best_phase(baseline_val, sft_val, grpo_val):
    """Determine which phase has the best (highest) value."""
    vals = {"BASELINE": baseline_val, "SFT": sft_val, "GRPO": grpo_val}
    # Filter out None values
    valid = {k: v for k, v in vals.items() if v is not None}
    if not valid:
        return "N/A"
    winner = max(valid, key=valid.get)
    return winner


def generate_tier_csv(tier):
    """Generate a single deep comparison CSV for one model tier."""
    print(f"\n{'='*60}")
    print(f"  Generating comparison CSV for Qwen3-VL-{tier.upper()}")
    print(f"{'='*60}")

    data = {}
    for phase in PHASES:
        data[phase] = load_metrics(tier, phase)

    output_path = CSV_DIR / f"comparison_{tier}_v4.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            f"Qwen3-VL-{tier.upper()} — VO v4 Deep Comparison",
            "", "", "", "", "", "", ""
        ])
        writer.writerow([])
        writer.writerow([
            "Metric Group",
            "Metric",
            "BASELINE",
            "SFT",
            "GRPO",
            "Δ SFT−Base",
            "% SFT vs Base",
            "Δ GRPO−Base",
            "% GRPO vs Base",
            "Δ GRPO−SFT",
            "% GRPO vs SFT",
            "Best Phase"
        ])

        total_metrics = 0
        grpo_wins = 0
        sft_wins = 0
        baseline_wins = 0

        for group_name, metrics_list in METRIC_GROUPS.items():
            # Section separator
            writer.writerow([])
            writer.writerow([f"--- {group_name} ---"])

            for display_name, key in metrics_list:
                b_val = data["baseline"].get(key)
                s_val = data["sft"].get(key)
                g_val = data["grpo"].get(key)

                # Compute deltas
                delta_sb = (s_val - b_val) if (s_val is not None and b_val is not None) else None
                delta_gb = (g_val - b_val) if (g_val is not None and b_val is not None) else None
                delta_gs = (g_val - s_val) if (g_val is not None and s_val is not None) else None

                winner = best_phase(b_val, s_val, g_val)

                # Count winners for summary (only for float metrics)
                if isinstance(b_val, (int, float)) and isinstance(s_val, (int, float)) and isinstance(g_val, (int, float)):
                    # Skip count-type metrics from the winner tally
                    if "count" not in key.lower() and "threshold" not in key.lower() and "words" not in key.lower() and "samples" not in key.lower():
                        total_metrics += 1
                        if winner == "GRPO":
                            grpo_wins += 1
                        elif winner == "SFT":
                            sft_wins += 1
                        else:
                            baseline_wins += 1

                writer.writerow([
                    group_name,
                    display_name,
                    fmt_val(b_val),
                    fmt_val(s_val),
                    fmt_val(g_val),
                    fmt_delta(delta_sb),
                    fmt_pct(b_val, s_val),
                    fmt_delta(delta_gb),
                    fmt_pct(b_val, g_val),
                    fmt_delta(delta_gs),
                    fmt_pct(s_val, g_val),
                    winner,
                ])

        # Summary footer
        writer.writerow([])
        writer.writerow([])
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Total Scored Metrics", total_metrics])
        writer.writerow(["BASELINE Wins", baseline_wins, f"({baseline_wins/total_metrics*100:.1f}%)" if total_metrics > 0 else ""])
        writer.writerow(["SFT Wins", sft_wins, f"({sft_wins/total_metrics*100:.1f}%)" if total_metrics > 0 else ""])
        writer.writerow(["GRPO Wins", grpo_wins, f"({grpo_wins/total_metrics*100:.1f}%)" if total_metrics > 0 else ""])

    print(f"  ✓ Saved: {output_path}")
    return data


def generate_master_csv(all_data):
    """Generate a master CSV with all 9 runs side by side."""
    print(f"\n{'='*60}")
    print(f"  Generating MASTER comparison CSV (all tiers)")
    print(f"{'='*60}")

    output_path = CSV_DIR / "master_comparison_all_tiers_v4.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Metric Group", "Metric"]
        for tier in TIERS:
            for phase in PHASES:
                header.append(f"{tier.upper()} {phase.upper()}")
        header.append("Best Overall")
        writer.writerow(header)

        for group_name, metrics_list in METRIC_GROUPS.items():
            writer.writerow([])
            writer.writerow([f"--- {group_name} ---"])

            for display_name, key in metrics_list:
                row = [group_name, display_name]
                vals = {}
                for tier in TIERS:
                    for phase in PHASES:
                        val = all_data[tier][phase].get(key)
                        row.append(fmt_val(val))
                        label = f"{tier.upper()} {phase.upper()}"
                        if val is not None:
                            vals[label] = val

                # Determine best overall
                if vals and all(isinstance(v, (int, float)) for v in vals.values()):
                    if "count" not in key.lower() and "threshold" not in key.lower() and "words" not in key.lower() and "samples" not in key.lower():
                        best = max(vals, key=vals.get)
                        row.append(best)
                    else:
                        row.append("")
                else:
                    row.append("")

                writer.writerow(row)

    print(f"  ✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("  VO v4 Deep CSV Comparison Generator")
    print("=" * 60)

    all_data = {}

    # Generate per-tier CSVs
    for tier in TIERS:
        tier_data = generate_tier_csv(tier)
        all_data[tier] = tier_data

    # Generate master CSV
    generate_master_csv(all_data)

    print(f"\n{'='*60}")
    print(f"  All CSVs saved to: {CSV_DIR.absolute()}")
    print(f"{'='*60}")
    print(f"\n  Files generated:")
    print(f"    1. comparison_2b_v4.csv")
    print(f"    2. comparison_4b_v4.csv")
    print(f"    3. comparison_8b_v4.csv")
    print(f"    4. master_comparison_all_tiers_v4.csv")


if __name__ == "__main__":
    main()
