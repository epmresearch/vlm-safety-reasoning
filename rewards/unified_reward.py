"""
Unified composite reward for GRPO training (v2 — 6-component design).

Parses the unified JSON output once via a shared strict-parse gate, then
scores six orthogonal aspects of the construction safety inspection output:

    1. Format validity  (schema compliance)
    2. Caption quality   (semantic + lexical + length calibration)
    3. Object grounding  (mask-union IoU with TN fix)
    4. Violation ID       (F-beta, β=2, recall-weighted)
    5. Violation grounding (TP-conditioned mask-union IoU)
    6. Reasoning quality   (TP-conditioned semantic + lexical + length)

Two integration modes:
    - get_reward_funcs_and_weights(): returns separate functions + weight list
      for TRL GRPOTrainer's native multi-reward support (preferred if available)
    - compute_reward(): single function returning the weighted composite score
      (fallback for older TRL versions without reward_weights support)
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logging import get_logger
from rewards.reward_utils import _strict_parse, _has_repetition_pathology

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Import the six reward components
# ---------------------------------------------------------------------------
from rewards.reward_format import compute_reward as _reward_format
from rewards.reward_caption import compute_reward as _reward_caption
from rewards.reward_grounding import compute_reward as _reward_grounding
from rewards.reward_violation_id import compute_reward as _reward_violation_id
from rewards.reward_violation_grounding import compute_reward as _reward_violation_grounding
from rewards.reward_reasoning import compute_reward as _reward_reasoning

# ---------------------------------------------------------------------------
# Component registry: ordered list of (name, function, weight)
# Weights sum to 1.0 by design so total reward ∈ [0, 1].
# ---------------------------------------------------------------------------
REWARD_COMPONENTS: List[Tuple[str, Callable, float]] = [
    ("reward_format",              _reward_format,              0.05),
    ("reward_caption",             _reward_caption,             0.15),
    ("reward_grounding",           _reward_grounding,           0.25),
    ("reward_violation_id",        _reward_violation_id,        0.30),
    ("reward_violation_grounding", _reward_violation_grounding, 0.15),
    ("reward_reasoning",           _reward_reasoning,           0.10),
]

# Repetition pathology penalty factor (applied multiplicatively to final score)
REPETITION_PENALTY_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Mode 1: Separate reward functions for TRL's native multi-reward support
# ---------------------------------------------------------------------------

def _make_batch_reward(reward_fn: Callable) -> Callable:
    """Wraps a single-instance reward function into a batch function.

    TRL's GRPOTrainer calls reward functions with the signature:
        (completions: List[str], **kwargs) -> List[float]

    Our component functions have the signature:
        (completion: str, ground_truth: dict, **kwargs) -> float

    This wrapper bridges the gap by iterating over the batch.
    """
    def batch_fn(prompts=None, completions=None, ground_truth=None, **kwargs):
        # Handle TRL >= 0.9 where prompts is the first positional argument
        if completions is None and prompts is not None:
            completions = prompts
        if completions is None:
            return []
        if ground_truth is None:
            return [0.0] * len(completions)
        
        import json
        parsed_gts = [json.loads(gt) if isinstance(gt, str) else gt for gt in ground_truth]
        return [reward_fn(c, gt) for c, gt in zip(completions, parsed_gts)]
    batch_fn.__name__ = getattr(reward_fn, '__name__', 'reward_fn')
    return batch_fn


def get_reward_funcs_and_weights() -> Tuple[List[Callable], List[float]]:
    """Returns the list of batch-compatible reward functions and their weights.

    For use with TRL GRPOTrainer's reward_funcs + reward_weights interface:
        funcs, weights = get_reward_funcs_and_weights()
        trainer = GRPOTrainer(..., reward_funcs=funcs, reward_weights=weights)

    Each returned function has the TRL-compatible batch signature:
        (completions: List[str], **kwargs) -> List[float]
    """
    funcs = [_make_batch_reward(fn) for _, fn, _ in REWARD_COMPONENTS]
    weights = [w for _, _, w in REWARD_COMPONENTS]
    return funcs, weights


# ---------------------------------------------------------------------------
# Mode 2: Single composite reward (fallback for older TRL versions)
# ---------------------------------------------------------------------------

def compute_reward(
    prediction: str,
    ground_truth: dict,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite reward: weighted sum of six components + repetition penalty.

    If JSON parsing fails at the shared strict-parse gate, the entire reward
    is 0.0 (not just the format component — all others would also be 0.0
    since there's no valid content to score).

    Args:
        prediction: Raw model output string (with ```json fences).
        ground_truth: Ground truth dict with keys:
            caption, rule_X_violation, and object classes.
        weights: Optional weight overrides. Keys must match component names.
            If None, uses the default weights from REWARD_COMPONENTS.

    Returns:
        Weighted sum of rewards in [0, 1], with repetition penalty applied.
    """
    # Build weight map
    w = {name: weight for name, _, weight in REWARD_COMPONENTS}
    if weights is not None:
        w.update(weights)

    # Compute each component
    component_scores: Dict[str, float] = {}
    weighted_sum = 0.0

    for name, reward_fn, _ in REWARD_COMPONENTS:
        score = reward_fn(prediction, ground_truth)
        component_scores[name] = score
        weighted_sum += score * w[name]

    # Apply repetition pathology penalty
    parsed = _strict_parse(prediction)
    if parsed is not None and _has_repetition_pathology(parsed):
        weighted_sum *= REPETITION_PENALTY_FACTOR
        logger.debug("Repetition pathology detected — penalty applied")

    logger.debug(
        "Component scores: %s → total=%.4f",
        component_scores,
        weighted_sum,
    )

    return weighted_sum


def compute_reward_with_breakdown(
    prediction: str,
    ground_truth: dict,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Like compute_reward but also returns per-component scores.

    Useful for logging during training.

    Returns:
        Dict with keys for each component score plus 'total' and
        'repetition_penalty_applied' (bool as 0.0/1.0).
    """
    w = {name: weight for name, _, weight in REWARD_COMPONENTS}
    if weights is not None:
        w.update(weights)

    result: Dict[str, float] = {}
    weighted_sum = 0.0

    for name, reward_fn, _ in REWARD_COMPONENTS:
        score = reward_fn(prediction, ground_truth)
        result[name] = score
        weighted_sum += score * w[name]

    # Repetition check
    parsed = _strict_parse(prediction)
    has_rep = parsed is not None and _has_repetition_pathology(parsed)
    result["repetition_penalty_applied"] = 1.0 if has_rep else 0.0

    if has_rep:
        weighted_sum *= REPETITION_PENALTY_FACTOR

    result["total"] = weighted_sum
    return result


# ---------------------------------------------------------------------------
# GRPOTrainer-compatible wrapper (single function mode)
# ---------------------------------------------------------------------------

def build_grpo_reward_fn(
    weights: Optional[Dict[str, float]] = None,
) -> Callable[[List[str]], List[float]]:
    """Build a single reward function matching TRL GRPOTrainer's signature.

    TRL's GRPOTrainer expects reward functions with signature:
        (completions: List[str], **kwargs) -> List[float]

    where kwargs includes ground_truth (list of dicts aligned with
    completions).

    Args:
        weights: Optional per-component weight overrides.

    Returns:
        A callable matching the GRPOTrainer reward function interface.
    """

    def unified_reward_fn(
        prompts=None,
        completions: List[str] = None,
        ground_truth: List[Dict] = None,
        **kwargs,
    ) -> List[float]:
        if completions is None and prompts is not None:
            completions = prompts
        if completions is None:
            return []
        if ground_truth is None:
            return [0.0] * len(completions)

        import json
        scores = []
        for completion, gt in zip(completions, ground_truth):
            parsed_gt = json.loads(gt) if isinstance(gt, str) else gt
            score = compute_reward(completion, parsed_gt, weights=weights)
            scores.append(score)
        return scores

    unified_reward_fn.__name__ = "reward_unified"
    return unified_reward_fn
