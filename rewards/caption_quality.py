"""
Caption quality reward for GRPO training.

Uses sentence-transformers cosine similarity between predicted and reference
captions. Fast, no external API calls, deterministic.
"""
from typing import Any, Dict

from core.logging import get_logger
from rewards.json_validity import try_parse_json

logger = get_logger(__name__)

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_reward(prediction: str, ground_truth: dict) -> float:
    """Reward function for caption quality via cosine similarity.

    Args:
        prediction: Raw model output string (fenced JSON).
        ground_truth: Ground truth dict with 'caption' key.

    Returns:
        Cosine similarity in [0, 1] (clamped at 0).
    """
    parsed = try_parse_json(prediction)
    if parsed is None:
        return 0.0

    pred_caption = parsed.get("caption", "")
    ref_caption = ground_truth.get("caption", "")

    if not pred_caption or not ref_caption:
        return 0.0

    model = _get_model()
    pred_emb = model.encode([pred_caption], convert_to_tensor=True)
    ref_emb = model.encode([ref_caption], convert_to_tensor=True)

    from sentence_transformers.util import cos_sim
    similarity = float(cos_sim(pred_emb, ref_emb)[0][0])

    return max(0.0, similarity)