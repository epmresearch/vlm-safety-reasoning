"""Tests for rewards/reward_violation_id.py."""
import json
import pytest
from rewards.reward_violation_id import compute_reward


def _completion(r1=None, r2=None, r3=None, r4=None):
    def _v(v):
        return {"bounding_box": [[100, 100, 500, 500]], "reason": "violation"} if v else None
    payload = {
        "caption": "test",
        "rule_1_violation": _v(r1),
        "rule_2_violation": _v(r2),
        "rule_3_violation": _v(r3),
        "rule_4_violation": _v(r4),
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _gt(r1=None, r2=None, r3=None, r4=None):
    def _v(v):
        return {"bounding_box": [[0.1, 0.1, 0.5, 0.5]], "reason": "violation"} if v else None
    return {
        "caption": "test",
        "rule_1_violation": _v(r1),
        "rule_2_violation": _v(r2),
        "rule_3_violation": _v(r3),
        "rule_4_violation": _v(r4),
        "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
    }


class TestRewardViolationId:
    def test_both_empty_returns_one(self):
        """Correctly identifies safe site — both predict and GT have no violations."""
        from rewards.reward_utils import reward_constant
        expected = reward_constant("unified", "violation_tn_constant", 0.15)
        assert compute_reward(_completion(), _gt()) == pytest.approx(expected)

    def test_perfect_single_violation(self):
        """Predicts rule_1, GT has rule_1 → F1=1.0."""
        assert compute_reward(_completion(r1=True), _gt(r1=True)) == pytest.approx(1.0)

    def test_all_violations_match(self):
        """All 4 rules match → F1=1.0."""
        score = compute_reward(
            _completion(r1=True, r2=True, r3=True, r4=True),
            _gt(r1=True, r2=True, r3=True, r4=True),
        )
        assert score == pytest.approx(1.0)

    def test_false_positive_only(self):
        """Predicts rule_1, GT has nothing → F1=0.0."""
        assert compute_reward(_completion(r1=True), _gt()) == pytest.approx(0.0)

    def test_false_negative_only(self):
        """Predicts nothing, GT has rule_1 → F1=0.0."""
        assert compute_reward(_completion(), _gt(r1=True)) == pytest.approx(0.0)

    def test_partial_match_fbeta2(self):
        """Predicts rule_1 and rule_2, GT has rule_1 and rule_3.
        TP={rule_1}, FP={rule_2}, FN={rule_3}
        Precision = 1/2, Recall = 1/2
        F2 = (1+4)*0.5*0.5/(4*0.5+0.5) = 5*0.25/2.5 = 0.5
        """
        score = compute_reward(
            _completion(r1=True, r2=True),
            _gt(r1=True, r3=True),
        )
        assert score == pytest.approx(0.5, abs=0.01)

    def test_recall_weighted_f2_favors_recall(self):
        """F2 should give higher score to high-recall, low-precision prediction."""
        # High recall (predicts all 4, GT has 2)
        high_recall_score = compute_reward(
            _completion(r1=True, r2=True, r3=True, r4=True),
            _gt(r1=True, r2=True),
        )
        # High precision (predicts only r1, GT has r1 and r2)
        high_precision_score = compute_reward(
            _completion(r1=True),
            _gt(r1=True, r2=True),
        )
        # F2 weights recall more → high recall case should score higher
        assert high_recall_score > high_precision_score

    def test_invalid_json_returns_zero(self):
        assert compute_reward("bad json", _gt(r1=True)) == pytest.approx(0.0)

    def test_score_between_zero_and_one(self):
        score = compute_reward(_completion(r1=True, r3=True), _gt(r1=True, r2=True))
        assert 0.0 <= score <= 1.0