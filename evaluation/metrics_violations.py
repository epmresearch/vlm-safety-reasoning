"""
Metrics for safety violation evaluation.
"""
from typing import Dict, List, Any, Set
import pandas as pd

from core.constants import RULES
from data.box_utils import compute_mask_union_iou, greedy_multibox_iou, scale_1000_to_01, clean_boxes, normalize_boxes
from core.logging import get_logger
# Single source of truth for "is this payload a violation?", shared with the GRPO rewards.
# Metrics previously used plain truthiness here, which disagreed with the reward on
# {"bounding_box": [], "reason": ""} — a violation to the metric, safe to the reward.
from rewards.reward_utils import _is_violation_present

logger = get_logger(__name__)

# IoU threshold for IoU-conditioned violation identification.
# A TP only counts if the greedy IoU between predicted and GT bounding boxes
# meets this threshold. 0.25 is deliberately lenient — for construction safety
# images where workers are small in frame, this means "the model at least knows
# which region of the image the violation is in."
IOU_CONDITIONED_THRESHOLD = 0.25

def compute_violation_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes rule identification (F1, Precision, Recall) and grounding IoU for safety violations.
    predictions: List of flat output dictionaries.
    references: List of flat output dictionaries.
    """
    if not predictions or not references:
        raise ValueError(
            "compute_violation_metrics requires non-empty predictions and references lists."
        )
    if len(predictions) != len(references):
        raise ValueError(
            f"compute_violation_metrics: length mismatch — "
            f"{len(predictions)} predictions vs {len(references)} references."
        )

    # We add rule_0 (no violation) for explicit tracking
    ALL_RULES = RULES + ["rule_0"]
    global_tp, global_fp, global_fn = 0, 0, 0
    rule_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ALL_RULES}
    
    # Mask Trackers
    rule_iou_tn0_mask = {r: [] for r in RULES}
    rule_inter_total_mask = {r: 0.0 for r in RULES}
    rule_union_total_mask = {r: 0.0 for r in RULES}

    # Greedy Trackers
    rule_iou_tn0_greedy = {r: [] for r in RULES}
    rule_inter_total_greedy = {r: 0.0 for r in RULES}
    rule_union_total_greedy = {r: 0.0 for r in RULES}

    rule_tn_count = {r: 0 for r in RULES}

    # IoU-conditioned violation identification counters
    # A TP only counts if greedy IoU >= IOU_CONDITIONED_THRESHOLD
    iou_cond_global_tp, iou_cond_global_fp, iou_cond_global_fn = 0, 0, 0
    iou_cond_rule_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in RULES}
    
    # Predictions that failed JSON parse / schema validation arrive as None.
    # They are NOT "the model said safe" — scoring them as such inflated rule_0 TP
    # (the true-negative / no-violation class) with what are really output failures.
    prediction_failures = 0

    for pred_dict, gt_dict in zip(predictions, references):
        prediction_failed = pred_dict is None
        if prediction_failed:
            prediction_failures += 1
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}

        pred_rules = set()
        pred_by_rule = {}
        gt_rules = set()
        gt_by_rule = {}
        
        for r in RULES:
            p_v = pred_dict.get(f"{r}_violation")
            if _is_violation_present(p_v):
                pred_rules.add(r)
                pred_by_rule[r] = p_v if isinstance(p_v, dict) else {}

            g_v = gt_dict.get(f"{r}_violation")
            if _is_violation_present(g_v):
                gt_rules.add(r)
                gt_by_rule[r] = g_v if isinstance(g_v, dict) else {}
        
        # Rule 0 tracking ("correctly reported no violation").
        # A failed prediction never earns rule_0 TP: emitting unparseable output is not
        # the same as correctly identifying a safe site. When GT is safe it is charged as
        # a rule_0 FN (the model failed to produce the correct safe answer), which keeps
        # violation_identification_recall_rule_0 honest as the over-flagging guardrail.
        if prediction_failed:
            if not gt_rules:
                rule_counts["rule_0"]["fn"] += 1
            # GT-has-violation + failure: the missed rules are already charged as FN
            # below. No rule_0 FP, because the model never asserted "safe".
        elif not gt_rules and not pred_rules:
            rule_counts["rule_0"]["tp"] += 1
        elif not gt_rules and pred_rules:
            rule_counts["rule_0"]["fn"] += 1
        elif gt_rules and not pred_rules:
            rule_counts["rule_0"]["fp"] += 1
            
        # Global counts
        tp = len(pred_rules & gt_rules)
        fp = len(pred_rules - gt_rules)
        fn = len(gt_rules - pred_rules)
        
        global_tp += tp
        global_fp += fp
        global_fn += fn
        
        # Per-rule counts
        for r in RULES:
            in_pred, in_gt = r in pred_rules, r in gt_rules
            if in_pred and in_gt:
                rule_counts[r]["tp"] += 1
            elif in_pred and not in_gt:
                rule_counts[r]["fp"] += 1
            elif not in_pred and in_gt:
                rule_counts[r]["fn"] += 1
                
        common_rules = pred_rules & gt_rules

        # --- IoU-conditioned identification ---
        # For rules predicted AND in GT (common_rules): check if grounding is good enough
        # For rules only in pred (FP) or only in GT (FN): unconditionally count them
        for r in RULES:
            in_pred, in_gt = r in pred_rules, r in gt_rules
            if in_pred and in_gt:
                # Compute greedy IoU to gate the TP
                p_boxes_1000 = normalize_boxes(pred_by_rule[r].get("bounding_box", []))
                g_boxes_01 = normalize_boxes(gt_by_rule[r].get("bounding_box", []))
                p_boxes_01 = [scale_1000_to_01(b) for b in p_boxes_1000]
                p_boxes_01 = clean_boxes(p_boxes_01)
                g_boxes_01 = clean_boxes(g_boxes_01)
                greedy_iou, _, _ = greedy_multibox_iou(p_boxes_01, g_boxes_01)

                if greedy_iou >= IOU_CONDITIONED_THRESHOLD:
                    iou_cond_rule_counts[r]["tp"] += 1
                    iou_cond_global_tp += 1
                else:
                    # Correctly identified the rule, but box is in the wrong place.
                    # Counts as both FP (bad prediction) and FN (missed the real location).
                    iou_cond_rule_counts[r]["fp"] += 1
                    iou_cond_rule_counts[r]["fn"] += 1
                    iou_cond_global_fp += 1
                    iou_cond_global_fn += 1
            elif in_pred and not in_gt:
                iou_cond_rule_counts[r]["fp"] += 1
                iou_cond_global_fp += 1
            elif not in_pred and in_gt:
                iou_cond_rule_counts[r]["fn"] += 1
                iou_cond_global_fn += 1

        # --- Grounding IoU for common rules (existing logic) ---
        for r in common_rules:
            pred_boxes_1000 = pred_by_rule[r].get("bounding_box", [])
            gt_boxes_01 = gt_by_rule[r].get("bounding_box", [])

            # Normalize first to handle flat lists
            pred_boxes_1000 = normalize_boxes(pred_boxes_1000)
            gt_boxes_01 = normalize_boxes(gt_boxes_01)

            # Scale pred to [0, 1] FIRST
            pred_boxes_01 = [scale_1000_to_01(b) for b in pred_boxes_1000]

            # Then clean both using the exact same [0, 1] scale threshold
            pred_boxes_01 = clean_boxes(pred_boxes_01)
            gt_boxes_01 = clean_boxes(gt_boxes_01)

            # 1. Mask-Union IoU
            mask_result = compute_mask_union_iou(pred_boxes_01, gt_boxes_01)
            mask_iou = mask_result["iou"] if mask_result["iou"] is not None else 0.0

            # 2. Greedy IoU
            greedy_iou_val, greedy_inter, greedy_union = greedy_multibox_iou(pred_boxes_01, gt_boxes_01)

            if not pred_boxes_01 and not gt_boxes_01:
                # Both sides flagged this rule but neither localized it — reachable via a
                # reason-only violation on both sides, which structural_repair.py actively
                # produces (a bare `true` becomes {"reason": "", "bounding_box": []}, a bare
                # reason string becomes {"reason": s, "bounding_box": []}).
                #
                # Scored 0.0: for a safety-grounding task, "we both agree something is wrong
                # but neither of us says where" deserves no grounding credit. The old `tn1`
                # convention scored this 1.0 — full credit for not localizing — and was
                # emitted as a parallel set of keys. Those keys were bit-identical to tn0 in
                # all 27 historical metrics.json (the branch never fired on real data), so
                # they carried no information while doubling every grounding row in the
                # comparison CSV. Removed.
                rule_iou_tn0_mask[r].append(0.0)
                rule_iou_tn0_greedy[r].append(0.0)
                rule_tn_count[r] += 1
            else:
                rule_iou_tn0_mask[r].append(mask_iou)
                rule_iou_tn0_greedy[r].append(greedy_iou_val)

            rule_inter_total_mask[r] += mask_result["intersection"]
            rule_union_total_mask[r] += mask_result["union"]

            rule_inter_total_greedy[r] += greedy_inter
            rule_union_total_greedy[r] += greedy_union

    metrics = {}

    # Unparseable / schema-invalid predictions, excluded from rule_0 true positives.
    metrics["violation_prediction_failure_count"] = prediction_failures
    metrics["violation_prediction_failure_rate"] = (
        prediction_failures / len(predictions) if predictions else 0.0
    )

    # Global (pooled) precision/recall/F1 — this is MICRO-averaging
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics["violation_identification_precision_micro"] = precision
    metrics["violation_identification_recall_micro"] = recall
    metrics["violation_identification_f1_micro"] = f1
    
    rule_precisions, rule_recalls, rule_f1s = [], [], []
    for r in ALL_RULES:
        tp, fp, fn = rule_counts[r]["tp"], rule_counts[r]["fp"], rule_counts[r]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        r_f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        
        metrics[f"violation_identification_precision_{r}"] = p
        metrics[f"violation_identification_recall_{r}"] = rec
        metrics[f"violation_identification_f1_{r}"] = r_f1
        
        if r in RULES:
            rule_precisions.append(p)
            rule_recalls.append(rec)
            rule_f1s.append(r_f1)
    
    metrics["violation_identification_precision_macro"] = (
        sum(rule_precisions) / len(rule_precisions) if rule_precisions else 0.0
    )
    metrics["violation_identification_recall_macro"] = (
        sum(rule_recalls) / len(rule_recalls) if rule_recalls else 0.0
    )
    metrics["violation_identification_f1_macro"] = (
        sum(rule_f1s) / len(rule_f1s) if rule_f1s else 0.0
    )
        
    # Grounding IoU per rule and global macro/micro
    # The `_tn1` counterparts of every key below were removed: they scored an
    # un-localized-on-both-sides rule as IoU 1.0, which is the wrong incentive for a
    # grounding task, and they were bit-identical to `_tn0` in all historical results.
    # The `_tn0` suffix is kept on the key names so existing dashboards/CSVs keep working.
    tn0_macros_mask, tn0_macros_greedy = [], []
    total_inter_mask, total_union_mask = 0.0, 0.0
    total_inter_greedy, total_union_greedy = 0.0, 0.0

    for r in RULES:
        metrics[f"violation_grounding_tn_count_{r}"] = rule_tn_count[r]

        # Mask-union IoU
        tn0_val_mask = sum(rule_iou_tn0_mask[r]) / len(rule_iou_tn0_mask[r]) if rule_iou_tn0_mask[r] else 0.0
        metrics[f"violation_grounding_mask_iou_{r}_tn0"] = tn0_val_mask
        tn0_macros_mask.append(tn0_val_mask)

        inter_r_mask = rule_inter_total_mask[r]
        union_r_mask = rule_union_total_mask[r]
        metrics[f"violation_grounding_mask_iou_{r}_micro"] = inter_r_mask / union_r_mask if union_r_mask > 0 else 0.0
        total_inter_mask += inter_r_mask
        total_union_mask += union_r_mask

        # Greedy multi-box IoU
        tn0_val_greedy = sum(rule_iou_tn0_greedy[r]) / len(rule_iou_tn0_greedy[r]) if rule_iou_tn0_greedy[r] else 0.0
        metrics[f"violation_grounding_greedy_iou_{r}_tn0"] = tn0_val_greedy
        tn0_macros_greedy.append(tn0_val_greedy)

        inter_r_greedy = rule_inter_total_greedy[r]
        union_r_greedy = rule_union_total_greedy[r]
        metrics[f"violation_grounding_greedy_iou_{r}_micro"] = inter_r_greedy / union_r_greedy if union_r_greedy > 0 else 0.0
        total_inter_greedy += inter_r_greedy
        total_union_greedy += union_r_greedy

    # Global grounding IoU aggregates
    metrics["violation_grounding_mask_iou_macro_tn0"] = sum(tn0_macros_mask) / len(tn0_macros_mask) if tn0_macros_mask else 0.0
    metrics["violation_grounding_mask_iou_micro_mean"] = total_inter_mask / total_union_mask if total_union_mask > 0 else 0.0
    metrics["violation_grounding_greedy_iou_macro_tn0"] = sum(tn0_macros_greedy) / len(tn0_macros_greedy) if tn0_macros_greedy else 0.0
    metrics["violation_grounding_greedy_iou_micro_mean"] = total_inter_greedy / total_union_greedy if total_union_greedy > 0 else 0.0

    # -----------------------------------------------------------------------
    # IoU-Conditioned Rule Identification (Metric 3)
    # A TP only counts if greedy IoU >= IOU_CONDITIONED_THRESHOLD.
    # rule_0 is excluded — it has no bounding box by definition.
    # -----------------------------------------------------------------------
    metrics["violation_identification_iou_threshold"] = IOU_CONDITIONED_THRESHOLD

    # Micro (pooled across all rules)
    iou_cond_p_micro = iou_cond_global_tp / (iou_cond_global_tp + iou_cond_global_fp) if (iou_cond_global_tp + iou_cond_global_fp) > 0 else 0.0
    iou_cond_r_micro = iou_cond_global_tp / (iou_cond_global_tp + iou_cond_global_fn) if (iou_cond_global_tp + iou_cond_global_fn) > 0 else 0.0
    iou_cond_f1_micro = 2 * iou_cond_p_micro * iou_cond_r_micro / (iou_cond_p_micro + iou_cond_r_micro) if (iou_cond_p_micro + iou_cond_r_micro) > 0 else 0.0
    metrics["violation_identification_iou_conditioned_precision_micro"] = iou_cond_p_micro
    metrics["violation_identification_iou_conditioned_recall_micro"] = iou_cond_r_micro
    metrics["violation_identification_iou_conditioned_f1_micro"] = iou_cond_f1_micro

    # Per-rule + macro (only RULES, not rule_0)
    iou_cond_precisions, iou_cond_recalls, iou_cond_f1s = [], [], []
    for r in RULES:
        tp_c = iou_cond_rule_counts[r]["tp"]
        fp_c = iou_cond_rule_counts[r]["fp"]
        fn_c = iou_cond_rule_counts[r]["fn"]
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        metrics[f"violation_identification_iou_conditioned_precision_{r}"] = p_c
        metrics[f"violation_identification_iou_conditioned_recall_{r}"] = r_c
        metrics[f"violation_identification_iou_conditioned_f1_{r}"] = f1_c
        iou_cond_precisions.append(p_c)
        iou_cond_recalls.append(r_c)
        iou_cond_f1s.append(f1_c)

    metrics["violation_identification_iou_conditioned_precision_macro"] = sum(iou_cond_precisions) / len(iou_cond_precisions) if iou_cond_precisions else 0.0
    metrics["violation_identification_iou_conditioned_recall_macro"] = sum(iou_cond_recalls) / len(iou_cond_recalls) if iou_cond_recalls else 0.0
    metrics["violation_identification_iou_conditioned_f1_macro"] = sum(iou_cond_f1s) / len(iou_cond_f1s) if iou_cond_f1s else 0.0
    
    return metrics