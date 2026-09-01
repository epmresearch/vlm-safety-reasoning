"""
Violation identification using F-beta (beta=2, recall-weighted) over predicted vs ground-truth rule sets.
"""

from rewards.reward_utils import (
    _strict_parse_for_task,
    _safe_reward,
    _is_violation_present,
    _is_substantive_violation,
    reward_constant,
)
from core.constants import RULES
from core.logging import get_logger

logger = get_logger(__name__)

@_safe_reward
def compute_reward(completion: str, ground_truth: dict, **kwargs) -> float:
    task = kwargs.get("task", "unified")
    parsed = _strict_parse_for_task(completion, task=task)
    if parsed is None:
        return 0.0

    # Two predicates, deliberately different, because a contentless violation
    # object -- {"reason": "", "bounding_box": []} -- is an ASSERTION with no
    # CONTENT, and those two facts have opposite consequences:
    #
    #   * As an assertion it must still count as a prediction, so that flagging a
    #     safe image is penalised as a false positive. Dropping it from pred_rules
    #     entirely would let the model earn true-negative credit for a false alarm.
    #   * As contentless it must NOT earn true-positive credit, because it names no
    #     location and gives no justification. Presence alone used to score a
    #     perfect F-beta = 1.0 here -- the most heavily weighted component under
    #     violations_only (0.40) -- while contributing nothing to violation
    #     grounding or reasoning, both of which are TP-conditioned and so never
    #     penalised it. That made it the cheapest possible way to farm this reward.
    #
    # So: presence drives precision (the FP denominator), substance drives recall
    # (the TP numerator). A contentless assertion on a real violation therefore
    # scores as a MISS, and on a safe image as a FALSE ALARM -- never as a hit.
    # Set `require_violation_substance: false` to restore presence-only scoring.
    require_substance = bool(reward_constant(task, "require_violation_substance", True))

    pred_rules = set()        # anything the model asserted, contentful or not
    substantive_rules = set() # assertions that actually say where or why
    gt_rules = set()
    for r in RULES:
        v = parsed.get(f"{r}_violation")
        if _is_violation_present(v):
            pred_rules.add(r)
            if not require_substance or _is_substantive_violation(v):
                substantive_rules.add(r)
        if _is_violation_present(ground_truth.get(f"{r}_violation")):
            gt_rules.add(r)

    # Both empty = correctly identified as safe.
    #
    # Tunable from the task YAML via `violation_tn_constant`; defaults to the
    # historical 0.15. The original rationale (balancing expected value against a
    # "91% class imbalance") does not describe what GRPO actually sees: the pool is
    # built 50/50 safe-vs-violation by data/build_grpo_pool.py. At 50/50 and c=0.15
    # the expected values are
    #     always-safe          : 0.5 * c            = 0.075
    #     always-assert-rule_1 : P(rule_1) * 1.0    ~ 0.385   (rule_1 covers ~39% of the pool)
    # so unconditional over-flagging beat honest abstention by ~5x on this
    # component — for a policy that never looks at the image. Honest abstention
    # only becomes competitive at c > ~0.78. See scripts/validate_rewards.py.
    if not pred_rules and not gt_rules:
        return float(reward_constant(task, "violation_tn_constant", 0.15))

    # TP requires substance; FP counts every assertion; FN is every ground-truth
    # rule not covered by a SUBSTANTIVE assertion (so a contentless "match" is a
    # miss, not a hit).
    tp = len(substantive_rules & gt_rules)
    fp = len(pred_rules - gt_rules)
    fn = len(gt_rules - substantive_rules)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta = float(reward_constant(task, "violation_fbeta", 2.0))  # recall-weighted for safety inspection
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

compute_reward.__name__ = "reward_violation_id"
