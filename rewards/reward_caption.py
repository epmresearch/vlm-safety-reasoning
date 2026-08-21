"""
Computes the caption quality reward for GRPO.
Uses semantic similarity, lexical overlap, and length calibration.
"""

import math
from typing import List, Dict
from rewards.reward_utils import _strict_parse_for_task, _safe_batch_reward, _embed_texts, _cosine_sim_batch, _ngram_f1
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_batch_reward
def compute_reward(completions: List[str], ground_truths: List[dict], **kwargs) -> List[float]:
    rewards = [0.0] * len(completions)
    
    valid_indices = []
    pred_caps = []
    gt_caps = []
    
    for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
        task = kwargs.get("task", "unified")
        parsed = _strict_parse_for_task(completion, task=task)
        if parsed is None:
            continue
            
        pred_cap = str(parsed.get("caption", "") or "").strip()
        gt_cap = str(gt.get("caption", "") or "").strip()
        
        if pred_cap and gt_cap:
            valid_indices.append(i)
            pred_caps.append(pred_cap)
            gt_caps.append(gt_cap)
            
    if not valid_indices:
        return rewards
        
    # Batch embed everything at once
    pred_embs = _embed_texts(pred_caps)
    ref_embs = _embed_texts(gt_caps)
    
    # Vectorized cosine similarity
    semantics = _cosine_sim_batch(pred_embs, ref_embs).tolist()
    
    for idx, pred_cap, gt_cap, sem in zip(valid_indices, pred_caps, gt_caps, semantics):
        semantic = max(0.0, sem)
        lexical = _ngram_f1(pred_cap, gt_cap, n_range=(1, 2))
        
        len_pred = len(pred_cap.split())
        len_gt = len(gt_cap.split())
        sigma = max(len_gt * 0.6, 5.0)
        length_factor = math.exp(-0.5 * ((len_pred - len_gt) / sigma) ** 2)
        
        content = 0.6 * semantic + 0.4 * lexical
        rewards[idx] = content * length_factor
        
    return rewards

compute_reward.__name__ = "reward_caption"
compute_reward.is_batched = True
