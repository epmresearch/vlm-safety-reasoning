import pytest

from rewards.reward_format import compute_reward as reward_format
from rewards.reward_grounding import compute_reward as reward_grounding
from rewards.unified_reward import get_reward_funcs_for_task
from tests.conftest import _make_object_only_completion


# ---------------------------------------------------------------------------
# Reward assembly
# ---------------------------------------------------------------------------

def test_get_reward_funcs_for_task_oo():
    funcs, weights = get_reward_funcs_for_task(task="object_only")

    names = [f.__name__ for f in funcs]
    assert names == ["reward_format", "reward_grounding"]
    assert weights == pytest.approx([0.10, 0.90])
    assert sum(weights) == pytest.approx(1.0)

    # The four components whose fields object_only never emits must be absent —
    # each would otherwise return its schema-parse floor and add pure variance.
    for absent in ("reward_caption", "reward_violation_id",
                   "reward_violation_grounding", "reward_reasoning"):
        assert absent not in names


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

def test_format_reward_oo_accepts_complete_output():
    completion = _make_object_only_completion(excavator=[[100, 200, 300, 400]])
    assert reward_format(completion, {}, task="object_only") == 1.0


def test_format_reward_oo_rejects_missing_class_keys():
    """ObjectOnlyOutput requires all 3 keys. Without that, {} would validate and
    the format reward would be free — see data/schemas.py."""
    assert reward_format('```json\n{"excavator": []}\n```', {}, task="object_only") == 0.0
    assert reward_format('```json\n{}\n```', {}, task="object_only") == 0.0


def test_format_reward_oo_rejects_unparseable():
    assert reward_format("I see an excavator.", {}, task="object_only") == 0.0


def test_format_reward_oo_rejects_bad_box_shape():
    bad = '```json\n{"excavator": [[1,2,3]], "rebar": [], "worker_with_white_hard_hat": []}\n```'
    assert reward_format(bad, {}, task="object_only") == 0.0


# ---------------------------------------------------------------------------
# Grounding reward — the sole objective for this task
# ---------------------------------------------------------------------------

# The per-class true-negative constant, now configurable per task from the task
# YAML (see rewards/reward_utils.py::grounding_tn_constant). Under object_only
# this is the sole objective's floor rather than one of six components, and it
# sets each class's break-even detection quality:
#
#     emit class k is positive-EV  <=>  E[IoU_k] > c_k * (1 - p_k) / p_k
#
# At the historical flat 0.15 that break-even was 1.55 for rebar and 1.15 for the
# hard-hat class — both unreachable, making suppression of those classes strictly
# dominant. object_only.yaml now carries frequency-aware values. Read them from
# the helper so retuning does not silently invalidate the arithmetic below; what
# is pinned here is the STRUCTURE (one perfect class + two true negatives).
from rewards.reward_utils import grounding_tn_constant as _tn

TN_EXC = _tn("object_only", "excavator")
TN_REB = _tn("object_only", "rebar")
TN_HAT = _tn("object_only", "worker_with_white_hard_hat")
TN_ALL = (TN_EXC + TN_REB + TN_HAT) / 3


def test_grounding_reward_oo_perfect_match(oo_gt_with_excavator):
    """Predictions are [0,1000], ground truth is [0,1]; the reward rescales the
    prediction only. Getting that backwards silently zeroes every IoU."""
    completion = _make_object_only_completion(excavator=[[100, 200, 300, 400]])
    score = reward_grounding(completion, oo_gt_with_excavator, task="object_only")
    # excavator IoU == 1.0, rebar and worker are correct true negatives.
    assert score == pytest.approx((1.0 + TN_REB + TN_HAT) / 3, abs=1e-3)


def test_grounding_reward_oo_all_true_negatives_equals_tn_constant(oo_gt_no_objects):
    score = reward_grounding(
        _make_object_only_completion(), oo_gt_no_objects, task="object_only"
    )
    assert score == pytest.approx(TN_ALL, abs=1e-9)


def test_grounding_reward_oo_disjoint_boxes_score_low(oo_gt_with_excavator):
    completion = _make_object_only_completion(excavator=[[700, 700, 900, 900]])
    score = reward_grounding(completion, oo_gt_with_excavator, task="object_only")
    perfect = reward_grounding(
        _make_object_only_completion(excavator=[[100, 200, 300, 400]]),
        oo_gt_with_excavator,
        task="object_only",
    )
    assert score < perfect


def test_grounding_reward_oo_unparseable_scores_zero(oo_gt_with_excavator):
    assert reward_grounding("garbage", oo_gt_with_excavator, task="object_only") == 0.0


def test_grounding_reward_oo_correct_empty_prediction_earns_credit(oo_gt_no_objects):
    """Correctly reporting an absent class earns the true-negative constant, not
    zero — otherwise the only safe strategy would be to hallucinate boxes."""
    completion = _make_object_only_completion()
    score = reward_grounding(completion, oo_gt_no_objects, task="object_only")
    assert score > 0.0


def test_grounding_reward_oo_false_positive_scores_below_correct_empty(oo_gt_no_objects):
    hallucinated = _make_object_only_completion(excavator=[[100, 100, 200, 200]])
    empty = _make_object_only_completion()
    assert (
        reward_grounding(hallucinated, oo_gt_no_objects, task="object_only")
        < reward_grounding(empty, oo_gt_no_objects, task="object_only")
    )


def test_oo_reward_funcs_run_batched_end_to_end(oo_gt_with_excavator):
    """The wrappers get_reward_funcs_for_task returns are batch-callable with the
    TRL signature, and receive task= so they resolve the right schema."""
    import json

    funcs, _ = get_reward_funcs_for_task(task="object_only")
    completions = [
        _make_object_only_completion(excavator=[[100, 200, 300, 400]]),
        "not json at all",
    ]
    ground_truth = [json.dumps(oo_gt_with_excavator)] * 2

    for fn in funcs:
        scores = fn(completions=completions, ground_truth=ground_truth)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert scores[1] == 0.0
