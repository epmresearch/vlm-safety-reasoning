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
    images: List[Any]  # <-- ADDED: Mandatory visual context
) -> Dict[str, float]:
    """Batched reasoning evaluation using the captioning metrics suite, broken down per rule."""
    if not pred_violations or not gt_violations:
        raise ValueError(
            "batch_score_reasoning requires non-empty predictions and references lists."
        )
    if images is None:
        raise ValueError(
            "batch_score_reasoning requires `images`; pass the image list aligned "
            "with pred_violations/gt_violations."
        )
    # ADDED: Ensure images match predictions and ground truth
    if len(pred_violations) != len(gt_violations) or len(pred_violations) != len(images):
        raise ValueError(
            "batch_score_reasoning: length mismatch between predictions, references, and images."
        )

    from core.constants import RULES
    
    # Track globally for macro averages
    all_pred_reasons = []
    all_gt_reasons = []
    all_images = [] 
    
    # Track separately per rule
    rule_pred_reasons = {r: [] for r in RULES}
    rule_gt_reasons = {r: [] for r in RULES}
    rule_images = {r: [] for r in RULES}  
    
    for i, (pred_dict, gt_dict) in enumerate(tqdm(zip(pred_violations, gt_violations), desc="Reasoning Eval", total=len(pred_violations))):
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}
        current_image = images[i] 
        
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
            all_images.append(current_image) 
                
            # Add to rule trackers
            rule_pred_reasons[r].append(pred_reason)
            rule_gt_reasons[r].append(gt_reason)
            rule_images[r].append(current_image)
            
    result = {}
    
    # 1. Compute global (micro/pooled) reasoning metrics
    if all_pred_reasons:
        logger.info(f"Computing global reasoning metrics over {len(all_pred_reasons)} valid reasons...")
        # ADDED: Pass images=all_images
        caption_res = compute_all_caption_metrics(all_pred_reasons, all_gt_reasons, images=all_images, include_spice=False)
        for k, v in caption_res.items():
            result[f"reasoning_text_similarity_{k}_micro"] = v
    else:
        result["reasoning_text_similarity_bertscore_precision_micro"] = 0.0
        result["reasoning_text_similarity_bertscore_recall_micro"] = 0.0
        result["reasoning_text_similarity_bertscore_f1_micro"] = 0.0
        result["reasoning_text_similarity_meteor_micro"] = 0.0
        result["reasoning_text_similarity_ciderd_micro"] = 0.0
        result["reasoning_text_similarity_clipscore_micro"] = 0.0
        result["reasoning_text_similarity_avg_words_per_caption_micro"] = 0.0
        result["reasoning_text_similarity_min_words_micro"] = 0.0
        result["reasoning_text_similarity_max_words_micro"] = 0.0
        
    # 2. Compute per-rule reasoning metrics
    for r in RULES:
        if rule_pred_reasons[r]:
            logger.info(f"Computing reasoning metrics for {r} over {len(rule_pred_reasons[r])} valid reasons...")
            rule_res = compute_all_caption_metrics(
                rule_pred_reasons[r], 
                rule_gt_reasons[r],
                images=rule_images[r], 
                include_spice=False
            )
            for k, v in rule_res.items():
                result[f"reasoning_text_similarity_{k}_{r}"] = v
        else:
            result[f"reasoning_text_similarity_bertscore_precision_{r}"] = 0.0
            result[f"reasoning_text_similarity_bertscore_recall_{r}"] = 0.0
            result[f"reasoning_text_similarity_bertscore_f1_{r}"] = 0.0
            result[f"reasoning_text_similarity_meteor_{r}"] = 0.0
            result[f"reasoning_text_similarity_ciderd_{r}"] = 0.0
            result[f"reasoning_text_similarity_clipscore_{r}"] = 0.0
            result[f"reasoning_text_similarity_avg_words_per_caption_{r}"] = 0.0
            result[f"reasoning_text_similarity_min_words_{r}"] = 0.0
            result[f"reasoning_text_similarity_max_words_{r}"] = 0.0
            
    # 3. Compute true macro reasoning metrics
    metrics_keys = ["bertscore_f1", "meteor", "ciderd", "clipscore"]  
    for k in metrics_keys:
        rule_scores = [result.get(f"reasoning_text_similarity_{k}_{r}", 0.0) for r in RULES]
        result[f"reasoning_text_similarity_{k}_macro"] = sum(rule_scores) / len(rule_scores)
            
    return result