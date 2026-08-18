"""
Reasoning text quality (TP-conditioned). Uses semantic + lexical similarity with length calibration.
"""

import math
from typing import List, Dict
from collections import defaultdict
from rewards.reward_utils import (
    _strict_parse, _safe_batch_reward, _is_violation_present,
    _embed_texts, _cosine_sim_batch, _ngram_f1,
)
from core.constants import RULES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_batch_reward
def compute_reward(completions: List[str], ground_truths: List[dict], **kwargs) -> List[float]:
    rewards = [0.0] * len(completions)
    
    all_pred_reasons = []
    all_gt_reasons = []
    task_mapping = [] 
    
    for i, (completion, ground_truth) in enumerate(zip(completions, ground_truths)):
        parsed = _strict_parse(completion)
        if parsed is None:
            continue
            
        pred_rules = set()
        gt_rules = set()
        common_rules = []
        
        for r in RULES:
            has_pred = _is_violation_present(parsed.get(f"{r}_violation"))
            has_gt = _is_violation_present(ground_truth.get(f"{r}_violation"))
            
            if has_pred: pred_rules.add(r)
            if has_gt: gt_rules.add(r)
            if has_pred and has_gt:
                common_rules.append(r)
                
        # Perfect True Negative
        if not pred_rules and not gt_rules:
            rewards[i] = 1.0
            continue
            
        if not common_rules:
            continue
            
        for r in common_rules:
            pv = parsed.get(f"{r}_violation", {}) or {}
            gv = ground_truth.get(f"{r}_violation", {}) or {}
            
            pred_reason = str((pv.get("reason", "") if isinstance(pv, dict) else "") or "").strip()
            gt_reason = str((gv.get("reason", "") if isinstance(gv, dict) else "") or "").strip()
            
            if pred_reason and gt_reason:
                all_pred_reasons.append(pred_reason)
                all_gt_reasons.append(gt_reason)
                task_mapping.append((i, r))

    if not all_pred_reasons:
        return rewards
        
    # Batch embed everything across all prompts and rules
    pred_embs = _embed_texts(all_pred_reasons)
    ref_embs = _embed_texts(all_gt_reasons)
    
    # Vectorized cosine similarity
    semantics = _cosine_sim_batch(pred_embs, ref_embs).tolist()
    
    scores_per_completion = defaultdict(list)
    for (i, r), pred_reason, gt_reason, sem in zip(task_mapping, all_pred_reasons, all_gt_reasons, semantics):
        semantic = max(0.0, sem)
        lexical = _ngram_f1(pred_reason, gt_reason, n_range=(1, 2))
        
        len_pred = len(pred_reason.split())
        len_gt = len(gt_reason.split())
        sigma = max(len_gt * 0.6, 3.0)
        length_factor = math.exp(-0.5 * ((len_pred - len_gt) / sigma) ** 2)
        
        content = 0.6 * semantic + 0.4 * lexical
        scores_per_completion[i].append(content * length_factor)
        
    for i, scores in scores_per_completion.items():
        if scores:
            rewards[i] = sum(scores) / len(scores)
            
    return rewards

compute_reward.__name__ = "reward_reasoning"
compute_reward.is_batched = True
