"""
Reasoning evaluation.

Scores the "reason" field in safety violations using the standard captioning
metrics suite (BERTScore, METEOR, CIDEr, CLIPScore).
"""
from typing import Any, Dict, List
from tqdm import tqdm

from core.logging import get_logger
from evaluation.metrics_captioning import compute_all_caption_metrics

logger = get_logger(__name__)

def batch_score_reasoning(
    pred_violations: List[Dict[str, Any]], 
    gt_violations: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Batched reasoning evaluation using the captioning metrics suite, broken down per rule."""
    from core.constants import RULES
    
    # Track globally for macro averages
    all_pred_reasons = []
    all_gt_reasons = []
    
    # Track separately per rule
    rule_pred_reasons = {r: [] for r in RULES}
    rule_gt_reasons = {r: [] for r in RULES}
    
    for i, (pred_dict, gt_dict) in enumerate(tqdm(zip(pred_violations, gt_violations), desc="Reasoning Eval", total=len(pred_violations))):
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}
        
        pred_by_rule = {r: pred_dict.get(f"{r}_violation", {}).get("reason", "") for r in RULES if pred_dict.get(f"{r}_violation")}
        gt_by_rule = {r: gt_dict.get(f"{r}_violation", {}).get("reason", "") for r in RULES if gt_dict.get(f"{r}_violation")}
        
        common_rules = set(pred_by_rule.keys()) & set(gt_by_rule.keys())
        
        for r in common_rules:
            pred_reason = pred_by_rule[r]
            gt_reason = gt_by_rule[r]
            
            # Sanitize empty strings to avoid tokenizer crashes
            pred_reason = pred_reason if pred_reason and str(pred_reason).strip() else "empty"
            gt_reason = gt_reason if gt_reason and str(gt_reason).strip() else "empty"
            
            # Add to global trackers
            all_pred_reasons.append(pred_reason)
            all_gt_reasons.append(gt_reason)
                
            # Add to rule trackers
            rule_pred_reasons[r].append(pred_reason)
            rule_gt_reasons[r].append(gt_reason)
            
    result = {}
    
    # 1. Compute global (macro) reasoning metrics
    if all_pred_reasons:
        logger.info(f"Computing global reasoning metrics over {len(all_pred_reasons)} valid reasons...")
        caption_res = compute_all_caption_metrics(all_pred_reasons, all_gt_reasons, include_spice=False)
        for k, v in caption_res.items():
            result[f"reasoning_text_similarity_{k}_macro"] = v
    else:
        result["reasoning_text_similarity_bertscore_f1_macro"] = 0.0
        result["reasoning_text_similarity_meteor_macro"] = 0.0
        result["reasoning_text_similarity_ciderd_macro"] = 0.0
        
    # 2. Compute per-rule reasoning metrics
    for r in RULES:
        if rule_pred_reasons[r]:
            logger.info(f"Computing reasoning metrics for {r} over {len(rule_pred_reasons[r])} valid reasons...")
            rule_res = compute_all_caption_metrics(
                rule_pred_reasons[r], 
                rule_gt_reasons[r],
                include_spice=False
            )
            for k, v in rule_res.items():
                result[f"reasoning_text_similarity_{k}_{r}"] = v
        else:
            result[f"reasoning_text_similarity_bertscore_f1_{r}"] = 0.0
            result[f"reasoning_text_similarity_meteor_{r}"] = 0.0
            result[f"reasoning_text_similarity_ciderd_{r}"] = 0.0
            
    return result
