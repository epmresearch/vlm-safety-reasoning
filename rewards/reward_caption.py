"""
Computes the caption quality reward for GRPO.
Uses semantic similarity, lexical overlap, and length calibration.
"""

import math
from rewards.reward_utils import _strict_parse, _safe_reward, _get_embed_model, _cosine_sim, _ngram_f1
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    parsed = _strict_parse(completion)
    if parsed is None:
        return 0.0

    pred_cap = str(parsed.get("caption", "") or "").strip()
    gt_cap = str(ground_truth.get("caption", "") or "").strip()

    if not pred_cap or not gt_cap:
        return 0.0

    # Semantic similarity
    model = _get_embed_model()
    pred_emb = model.encode([pred_cap], convert_to_tensor=True)
    ref_emb = model.encode([gt_cap], convert_to_tensor=True)
    semantic = max(0.0, _cosine_sim(pred_emb[0], ref_emb[0]))

    # Lexical overlap
    lexical = _ngram_f1(pred_cap, gt_cap, n_range=(1, 2))

    # Length calibration (Gaussian centered on THIS sample's GT length)
    len_pred = len(pred_cap.split())
    len_gt = len(gt_cap.split())
    sigma = max(len_gt * 0.6, 5.0)
    length_factor = math.exp(-0.5 * ((len_pred - len_gt) / sigma) ** 2)

    content = 0.6 * semantic + 0.4 * lexical
    return content * length_factor

compute_reward.__name__ = "reward_caption"
