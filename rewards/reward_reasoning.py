"""
Reasoning text quality (TP-conditioned). Uses semantic + lexical similarity with length calibration.
"""

import math
from rewards.reward_utils import (
    _strict_parse, _safe_reward, _is_violation_present,
    _get_embed_model, _cosine_sim, _ngram_f1,
)
from core.constants import RULES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    parsed = _strict_parse(completion)
    if parsed is None:
        return 0.0

    common_rules = []
    for r in RULES:
        if _is_violation_present(parsed.get(f"{r}_violation")) and \
           _is_violation_present(ground_truth.get(f"{r}_violation")):
            common_rules.append(r)

    if not common_rules:
        return 0.0

    model = _get_embed_model()
    rule_scores = []
    for r in common_rules:
        pv = parsed.get(f"{r}_violation", {}) or {}
        gv = ground_truth.get(f"{r}_violation", {}) or {}

        pred_reason = str(pv.get("reason", "") if isinstance(pv, dict) else "").strip()
        gt_reason = str(gv.get("reason", "") if isinstance(gv, dict) else "").strip()

        if not pred_reason or not gt_reason:
            rule_scores.append(0.0)
            continue

        pred_emb = model.encode([pred_reason], convert_to_tensor=True)
        ref_emb = model.encode([gt_reason], convert_to_tensor=True)
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
