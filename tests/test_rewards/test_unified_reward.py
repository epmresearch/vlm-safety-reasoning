"""
Tests for rewards/unified_reward.py — the composite reward and batch wrappers.
"""
import json
import pytest
from rewards.unified_reward import (
    compute_reward,
    compute_reward_with_breakdown,
    get_reward_funcs_and_weights,
    build_grpo_reward_fn,
    REWARD_COMPONENTS,
    REPETITION_PENALTY_FACTOR,
)


def _make_valid_completion(caption="A construction site.", violations=None):
    payload = {
        "caption": caption,
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    if violations:
        payload.update(violations)
    return "```json\n" + json.dumps(payload) + "\n```"


GT_SAFE = {
    "caption": "A construction site.",
    "rule_1_violation": None,
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
    "excavator": [],
    "rebar": [],
    "worker_with_white_hard_hat": [],
}


class TestRewardWeights:
    def test_weights_sum_to_one(self):
        total = sum(w for _, _, w in REWARD_COMPONENTS)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_six_components(self):
        assert len(REWARD_COMPONENTS) == 6

    def test_component_names(self):
        names = [name for name, _, _ in REWARD_COMPONENTS]
        assert "reward_format" in names
        assert "reward_caption" in names
        assert "reward_grounding" in names
        assert "reward_violation_id" in names
        assert "reward_violation_grounding" in names
        assert "reward_reasoning" in names

    def test_get_reward_funcs_and_weights_returns_matching_lengths(self):
        funcs, weights = get_reward_funcs_and_weights()
        assert len(funcs) == len(weights)
        assert len(funcs) == 6

    def test_weights_are_positive(self):
        _, weights = get_reward_funcs_and_weights()
        assert all(w > 0 for w in weights)


class TestCompositeReward:
    def test_invalid_json_returns_zero(self):
        score = compute_reward("bad json", GT_SAFE)
        assert score == pytest.approx(0.0)

    def test_score_between_zero_and_one(self):
        score = compute_reward(_make_valid_completion(), GT_SAFE)
        assert 0.0 <= score <= 1.0

    def test_perfect_safe_prediction_high_score(self):
        """Correct safe prediction (no violations, matching GT) should score high."""
        score = compute_reward(_make_valid_completion("A construction site."), GT_SAFE)
        # Format=1.0, ViolationID=0.10, Grounding=0.10, Caption≈1.0
        assert score > 0.20

    def test_breakdown_total_matches_compute_reward(self):
        completion = _make_valid_completion()
        score = compute_reward(completion, GT_SAFE)
        breakdown = compute_reward_with_breakdown(completion, GT_SAFE)
        assert score == pytest.approx(breakdown["total"], abs=1e-6)

    def test_breakdown_has_all_components(self):
        breakdown = compute_reward_with_breakdown(_make_valid_completion(), GT_SAFE)
        assert "reward_format" in breakdown
        assert "reward_caption" in breakdown
        assert "reward_grounding" in breakdown
        assert "reward_violation_id" in breakdown
        assert "reward_violation_grounding" in breakdown
        assert "reward_reasoning" in breakdown
        assert "total" in breakdown
        assert "repetition_penalty_applied" in breakdown

    def test_repetition_pathology_reduces_score(self):
        """Repeated boxes trigger penalty."""
        repeated_boxes = [[100, 100, 500, 500]] * 10
        payload_rep = {
            "caption": "test",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "excavator": repeated_boxes,
            "rebar": [], "worker_with_white_hard_hat": [],
        }
        completion_rep = "```json\n" + json.dumps(payload_rep) + "\n```"
        completion_normal = _make_valid_completion()

        score_rep = compute_reward(completion_rep, GT_SAFE)
        score_normal = compute_reward(completion_normal, GT_SAFE)
        assert score_rep < score_normal

    def test_repetition_penalty_factor_applied(self):
        """Verify penalty factor is exactly REPETITION_PENALTY_FACTOR."""
        repeated_boxes = [[100, 100, 500, 500]] * 10
        payload_rep = {
            "caption": "test",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "excavator": repeated_boxes,
            "rebar": [], "worker_with_white_hard_hat": [],
        }
        completion = "```json\n" + json.dumps(payload_rep) + "\n```"
        breakdown = compute_reward_with_breakdown(completion, GT_SAFE)
        assert breakdown["repetition_penalty_applied"] == 1.0
        # total = sum_of_components * REPETITION_PENALTY_FACTOR
        sum_components = sum(
            breakdown[name] * weight
            for name, _, weight in REWARD_COMPONENTS
        )
        assert breakdown["total"] == pytest.approx(sum_components * REPETITION_PENALTY_FACTOR, abs=1e-6)


class TestBatchRewardFunctions:
    """Tests for the TRL-compatible batch reward functions."""

    def test_batch_fn_returns_list_of_floats(self):
        funcs, _ = get_reward_funcs_and_weights()
        completions = [_make_valid_completion(), _make_valid_completion()]
        gts = [GT_SAFE, GT_SAFE]
        for fn in funcs:
            result = fn(completions=completions, ground_truth=gts)
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(s, float) for s in result)

    def test_batch_fn_none_ground_truth_returns_zeros(self):
        funcs, _ = get_reward_funcs_and_weights()
        completions = [_make_valid_completion()]
        for fn in funcs:
            result = fn(completions=completions, ground_truth=None)
            assert result == [0.0]

    def test_batch_fn_accepts_prompts_positional(self):
        """Newer TRL passes prompts as first positional arg."""
        funcs, _ = get_reward_funcs_and_weights()
        completions = [_make_valid_completion()]
        gts = [GT_SAFE]
        for fn in funcs:
            # Simulate new TRL signature: fn(prompts, completions, ground_truth=...)
            try:
                result = fn(["dummy_prompt"], completions, ground_truth=gts)
                assert isinstance(result, list)
                assert len(result) == 1
            except TypeError:
                pytest.fail(
                    f"Reward function {fn.__name__} failed with new TRL signature "
                    f"(prompts as positional arg). Fix _make_batch_reward."
                )

    def test_grpo_reward_fn_wrapper(self):
        reward_fn = build_grpo_reward_fn()
        completions = [_make_valid_completion(), "bad json"]
        gts = [GT_SAFE, GT_SAFE]
        result = reward_fn(prompts=None, completions=completions, ground_truth=gts)
        assert len(result) == 2
        assert result[0] > 0.0
        assert result[1] == pytest.approx(0.0)  # bad json → 0