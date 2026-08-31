import os
import json
from pathlib import Path

MODELS = ["baseline", "sft", "grpo"]
RESULTS_DIR = Path("evaluation_results")
OUTPUT_MD = RESULTS_DIR / "qualitative_examples.md"

def load_predictions():
    data = {}
    for model in MODELS:
        pred_file = RESULTS_DIR / model / "predictions_with_eval.json"
        if pred_file.exists():
            with open(pred_file, "r") as f:
                data[model] = json.load(f)
        else:
            print(f"Warning: {pred_file} not found.")
    return data

def build_image_map(predictions_data):
    # Map image_id -> {model: data}
    image_map = {}
    for model, preds in predictions_data.items():
        for item in preds:
            img_id = item.get("image_id", "unknown")
            if img_id not in image_map:
                image_map[img_id] = {}
            image_map[img_id][model] = item
    return image_map

def _rule_set(violation_dict):
    """Set of rules flagged in a flat output dict, using the shared reward predicate."""
    from core.constants import RULES
    from rewards.reward_utils import _is_violation_present

    d = violation_dict or {}
    return {r for r in RULES if _is_violation_present(d.get(f"{r}_violation"))}


def _sample_f1(pred, gt):
    """Per-sample F1 over the flagged-rule set. 1.0 when both agree the site is safe."""
    p, g = _rule_set(pred), _rule_set(gt)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def extract_examples(image_map):
    """Rank images by per-sample violation-identification F1.

    The previous heuristic read 'reasoning_text_similarity_bertscore_f1' off each
    prediction record, but that key only ever exists AGGREGATED in metrics.json — never
    per sample. Every lookup returned the 0 default, so no triumph or failure could ever
    be selected. This scores each sample from data the file actually carries
    (parsed_output vs ground_truth).
    """
    triumphs = []
    failures = []

    for img_id, models_data in image_map.items():
        grpo_data = models_data.get("grpo")
        baseline_data = models_data.get("baseline")
        if not (grpo_data and baseline_data):
            continue

        gt = grpo_data.get("ground_truth") or baseline_data.get("ground_truth")
        grpo_score = _sample_f1(grpo_data.get("parsed_output"), gt)
        base_score = _sample_f1(baseline_data.get("parsed_output"), gt)

        # Triumph: GRPO got the rule set right where the baseline did not.
        if grpo_score > 0.6 and base_score < 0.3:
            triumphs.append((img_id, base_score, grpo_score, models_data))

        # Failure: GRPO got the rule set wrong.
        if grpo_score < 0.1:
            failures.append((img_id, grpo_score, models_data))

    triumphs.sort(key=lambda x: x[2] - x[1], reverse=True)  # by largest gain
    failures.sort(key=lambda x: x[1])                       # by lowest score

    return triumphs[:5], failures[:5]

def generate_markdown(triumphs, failures, image_map):
    md = ["# Qualitative Examples\n"]
    
    md.append("## GRPO Triumphs (GRPO Succeeded, Baseline Failed)\n")
    if not triumphs:
        md.append("*No data available to compare GRPO vs Baseline yet.*\n")
    else:
        for img_id, b_score, g_score, data in triumphs:
            md.append(f"### Image ID: `{img_id}`")
            md.append(f"**Baseline Score:** {b_score:.3f} | **GRPO Score:** {g_score:.3f}\n")
            md.append("#### Baseline Prediction:")
            md.append(f"```json\n{json.dumps(data['baseline'].get('parsed_output', {}), indent=2)}\n```\n")
            md.append("#### GRPO Prediction:")
            md.append(f"```json\n{json.dumps(data['grpo'].get('parsed_output', {}), indent=2)}\n```\n")
            md.append("---\n")
            
    md.append("## Catastrophic Failures (GRPO Failed)\n")
    if not failures:
        md.append("*No GRPO failure data available yet.*\n")
    else:
        for img_id, g_score, data in failures:
            md.append(f"### Image ID: `{img_id}`")
            md.append(f"**GRPO Score:** {g_score:.3f}\n")
            md.append("#### GRPO Prediction:")
            md.append(f"```json\n{json.dumps(data['grpo'].get('parsed_output', {}), indent=2)}\n```\n")
            md.append("---\n")
            
    # Also just list the top 3 overall random examples to compare SFT vs Baseline if GRPO is missing
    md.append("## General Comparisons (Baseline vs SFT vs GRPO)\n")
    count = 0
    for img_id, data in image_map.items():
        if count >= 3: break
        md.append(f"### Image ID: `{img_id}`\n")
        for model in MODELS:
            if model in data:
                md.append(f"#### {model.upper()} Prediction:")
                md.append(f"```json\n{json.dumps(data[model].get('parsed_output', {}), indent=2)}\n```\n")
        md.append("---\n")
        count += 1
            
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(md))
        
def main():
    print("Loading predictions...")
    data = load_predictions()
    if not data:
        print("No prediction files found.")
        return
        
    image_map = build_image_map(data)
    triumphs, failures = extract_examples(image_map)
    
    print("Generating qualitative report...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_markdown(triumphs, failures, image_map)
    
    print(f"Qualitative report saved to {OUTPUT_MD.absolute()}")

if __name__ == "__main__":
    main()
