"""
Reasoning text quality (TP-conditioned). Uses semantic + lexical similarity with length calibration.
"""

import math
from rewards.reward_utils import (
    _strict_parse, _safe_reward, _is_violation_present,
    _embed_texts, _cosine_sim, _ngram_f1,
)
from core.constants import RULES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    parsed = _strict_parse(completion)
    if parsed is None:
        return 0.0

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

    # Perfect True Negative: Both correctly agree there are no violations
    if not pred_rules and not gt_rules:
        return 1.0

    if not common_rules:
        return 0.0

    rule_scores = []
    for r in common_rules:
        pv = parsed.get(f"{r}_violation", {}) or {}
        gv = ground_truth.get(f"{r}_violation", {}) or {}

        pred_reason = str((pv.get("reason", "") if isinstance(pv, dict) else "") or "").strip()
        gt_reason = str((gv.get("reason", "") if isinstance(gv, dict) else "") or "").strip()

        if not pred_reason or not gt_reason:
            rule_scores.append(0.0)
            continue

        pred_emb = _embed_texts([pred_reason])
        ref_emb = _embed_texts([gt_reason])
        semantic = max(0.0, _cosine_sim(pred_emb[0], ref_emb[0]))

        lexical = _ngram_f1(pred_reason, gt_reason, n_range=(1, 2))

        len_pred = len(pred_reason.split())
        len_gt = len(gt_reason.split())
        sigma = max(len_gt * 0.6, 3.0)  # Relaxed tolerance for natural language variation
        length_factor = math.exp(-0.5 * ((len_pred - len_gt) / sigma) ** 2)

        content = 0.6 * semantic + 0.4 * lexical
        rule_scores.append(content * length_factor)

    return sum(rule_scores) / len(rule_scores) if rule_scores else 0.0

compute_reward.__name__ = "reward_reasoning"
