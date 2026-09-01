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

ONE integration path: get_reward_funcs_for_task(task) returns the task's active
components as TRL-compatible batch functions plus their weights, for
GRPOTrainer(reward_funcs=..., reward_weights=...). TRL then forms the weighted sum
itself, which is what gives per-component logging in W&B.

A second "composite reward" mode used to exist as a fallback for older TRL
versions. It has been deleted: it ignored the task's `reward_components` entirely
and always scored all six components at *unified* weights, so any non-unified task
that ever fell into it would have trained silently against the wrong objective.
`trl==0.23.0` supports reward_weights natively, so the fallback was unreachable
dead weight with a live footgun in it.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logging import get_logger
from rewards.reward_utils import _strict_parse_for_task, _has_repetition_pathology, reward_constant

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
# Task-aware reward assembly
# ---------------------------------------------------------------------------

# Full registry of all available reward components (name -> (function, default_weight))
ALL_REWARD_COMPONENTS = {
    name: (fn, weight) for name, fn, weight in REWARD_COMPONENTS
}


def _apply_repetition_penalty(scores, completions, task: str):
    """Scales a component's scores down on completions showing box repetition.

    Applied per component rather than to the total, which is mathematically
    identical because TRL forms the weighted sum linearly:

        sum_k w_k * (f * r_k)  ==  f * sum_k w_k * r_k

    This penalty used to live only in the deleted Mode-2 composite path, so it
    **never fired in production** — the live path had no repetition check at all.
    That matters because truncated generation is a known failure mode here (see
    the repetition_penalty ablations and _has_repetition_pathology): a model that
    loops on one box emits schema-valid JSON and can score well on IoU while
    saying nothing.

    Trigger: >5 occurrences of one identical box tuple, pooled across every box
    field the task owns. Tunable per task via `repetition_penalty_factor`
    (default 0.5); set it to 1.0 to disable. caption_only parses to a caption with
    no boxes, so it can never fire there.
    """
    factor = float(reward_constant(task, "repetition_penalty_factor", 0.5))
    if factor >= 1.0:
        return scores

    out = []
    for score, completion in zip(scores, completions):
        parsed = _strict_parse_for_task(completion, task=task)
        if parsed is not None and _has_repetition_pathology(parsed):
            out.append(score * factor)
        else:
            out.append(score)
    return out


def _make_task_aware_batch_reward(reward_fn: Callable, task: str) -> Callable:
    """Wraps a reward function for task-aware batch execution.

    Every component receives ``task=`` so it resolves the right schema and wire
    format through _strict_parse_for_task. Batched components (reward_caption,
    reward_reasoning) take it via kwargs; per-sample ones take it as a keyword.
    """
    def batch_fn(prompts=None, completions=None, ground_truth=None, **kwargs):
        if completions is None and prompts is not None:
            completions = prompts
        if completions is None:
            return []
        if ground_truth is None:
            return [0.0] * len(completions)
        
        import json as _json
        parsed_gts = [_json.loads(gt) if isinstance(gt, str) else gt for gt in ground_truth]

        # For batched reward functions
        if getattr(reward_fn, 'is_batched', False):
            kwargs["task"] = task
            scores = reward_fn(completions, parsed_gts, **kwargs)
        else:
            scores = [reward_fn(c, gt, task=task) for c, gt in zip(completions, parsed_gts)]

        return _apply_repetition_penalty(scores, completions, task)
    batch_fn.__name__ = getattr(reward_fn, '__name__', 'reward_fn')
    return batch_fn


def get_reward_funcs_for_task(task: str = "unified") -> Tuple[List[Callable], List[float]]:
    """Returns reward functions and weights for the specified task.
    
    The active components and their weights are read from
    configs/tasks/<task>.yaml. A task YAML that omits ``reward_components``
    falls back to the full 6-component set at default weights, which is what
    'unified' relies on.

    Args:
        task: Any task registered in core/tasks.py::TASK_REGISTRY.
    
    Returns:
        Tuple of (list of batch-compatible reward functions, list of weights).
    """
    from core.config import load_task_config
    from core.tasks import validate_task

    validate_task(task)
    task_cfg = load_task_config(task)

    # Get active components from task config, defaulting to all components
    active_components = task_cfg.get(
        "reward_components", [name for name, _, _ in REWARD_COMPONENTS]
    )
    weight_overrides = task_cfg.get("reward_weights", {})
    
    funcs = []
    weights = []
    for name in active_components:
        if name not in ALL_REWARD_COMPONENTS:
            raise ValueError(
                f"Unknown reward component: {name!r}. "
                f"Available: {list(ALL_REWARD_COMPONENTS.keys())}"
            )
        fn, default_weight = ALL_REWARD_COMPONENTS[name]
        funcs.append(_make_task_aware_batch_reward(fn, task))
        weights.append(weight_overrides.get(name, default_weight))
    
    stray_weights = set(weight_overrides) - set(active_components)
    if stray_weights:
        raise ValueError(
            f"configs/tasks/{task}.yaml declares reward_weights for "
            f"{sorted(stray_weights)}, which are not in reward_components "
            f"{list(active_components)}. Unknown weight keys are otherwise ignored "
            "silently, so the intended weighting would never take effect."
        )

    logger.info(
        f"Task '{task}' reward assembly: {len(funcs)} components, "
        f"names={active_components}, weights={weights}"
    )
    return funcs, weights
