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
    gt_violations: List[Dict[str, Any]],
    images: List[Any] = None
) -> Dict[str, float]:
    """Batched reasoning evaluation using the captioning metrics suite."""
    all_pred_reasons = []
    all_gt_reasons = []
    reasoning_images = []
    
    for i, (pred_dict, gt_dict) in enumerate(tqdm(zip(pred_violations, gt_violations), desc="Reasoning Eval", total=len(pred_violations))):
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}
        
        from core.constants import RULES
        pred_by_rule = {r: pred_dict.get(f"{r}_violation", {}).get("reason", "") for r in RULES if pred_dict.get(f"{r}_violation")}
        gt_by_rule = {r: gt_dict.get(f"{r}_violation", {}).get("reason", "") for r in RULES if gt_dict.get(f"{r}_violation")}
        
        common_rules = set(pred_by_rule.keys()) & set(gt_by_rule.keys())
        
        for r in common_rules:
            pred_reason = pred_by_rule[r]
            gt_reason = gt_by_rule[r]
            
            # Sanitize empty strings
            pred_reason = pred_reason if pred_reason and str(pred_reason).strip() else "empty"
            gt_reason = gt_reason if gt_reason and str(gt_reason).strip() else "empty"
            
            all_pred_reasons.append(pred_reason)
            all_gt_reasons.append(gt_reason)
            if images and i < len(images):
                reasoning_images.append(images[i])
            
    result = {}
    if all_pred_reasons:
        logger.info(f"Computing reasoning metrics over {len(all_pred_reasons)} valid reasons...")
        # compute_all_caption_metrics handles the standard suite
        caption_res = compute_all_caption_metrics(all_pred_reasons, all_gt_reasons, images=reasoning_images if images else None)
        
        # Prefix keys with "reasoning_"
        for k, v in caption_res.items():
            result[f"reasoning_{k}"] = v
    else:
        result["reasoning_bertscore_f1"] = 0.0
        result["reasoning_meteor"] = 0.0
        result["reasoning_cider"] = 0.0
        result["reasoning_clipscore"] = 0.0
        
    return result
