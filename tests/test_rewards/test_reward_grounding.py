"""
Tests for rewards/reward_grounding.py.
GT boxes are in [0,1] scale. Prediction boxes are in [0,1000] scale.
"""
import json
import pytest
from rewards.reward_grounding import compute_reward


def _make_completion(excavator_1000=None, rebar_1000=None, hat_1000=None):
    payload = {
        "caption": "test",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": excavator_1000 or [],
        "rebar": rebar_1000 or [],
        "worker_with_white_hard_hat": hat_1000 or [],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _gt(excavator_01=None, rebar_01=None, hat_01=None):
    return {
        "caption": "test",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": excavator_01 or [],
        "rebar": rebar_01 or [],
        "worker_with_white_hard_hat": hat_01 or [],
    }


class TestRewardGrounding:
    def test_true_negative_all_classes_returns_one(self):
        """No objects predicted, no objects in GT → 0.10 (correct TN for all classes)."""
        completion = _make_completion()
        gt = _gt()
        score = compute_reward(completion, gt)
        assert score == pytest.approx(0.10)

    def test_perfect_true_positive_excavator(self):
        """Predicted box matches GT exactly (after scale conversion)."""
        # GT in [0,1]: [0.1, 0.05, 0.8, 0.7]
        # Pred in [0,1000]: [100, 50, 800, 700]
        completion = _make_completion(excavator_1000=[[100, 50, 800, 700]])
        gt = _gt(excavator_01=[[0.1, 0.05, 0.8, 0.7]])
        score = compute_reward(completion, gt)
        # Perfect IoU = 1.0 for excavator
        # Other 2 classes: TN → 0.10 each
        # Average = (1.0 + 0.10 + 0.10) / 3 = 0.40
        assert score == pytest.approx(0.40, abs=0.02)

    def test_false_positive_single_class(self):
        """Predicts excavator when none in GT → FP → 0.0 for that class."""
        completion = _make_completion(excavator_1000=[[100, 50, 800, 700]])
        gt = _gt()  # no excavator in GT
        score = compute_reward(completion, gt)
        # excavator: FP → 0.0; rebar: TN → 0.10; hat: TN → 0.10
        assert score == pytest.approx(0.20 / 3.0, abs=0.02)

    def test_false_negative_single_class(self):
        """Misses excavator that exists in GT → FN → 0.0 for that class."""
        completion = _make_completion()  # no prediction
        gt = _gt(excavator_01=[[0.1, 0.05, 0.8, 0.7]])
        score = compute_reward(completion, gt)
        # excavator: FN → 0.0; others: TN → 0.10
        assert score == pytest.approx(0.20 / 3.0, abs=0.02)

    def test_low_iou_tp(self):
        """Predicted box overlaps but not perfectly."""
        # GT: [0.1, 0.1, 0.9, 0.9] — large box
        # Pred in 1000 scale: [500, 500, 900, 900] — only covers bottom-right quarter
        completion = _make_completion(excavator_1000=[[500, 500, 900, 900]])
        gt = _gt(excavator_01=[[0.1, 0.1, 0.9, 0.9]])
        score = compute_reward(completion, gt)
        # score for excavator < 1.0, other classes TN = 0.10
        assert score < 1.0

    def test_invalid_json_returns_zero(self):
        assert compute_reward("```json\n{bad\n```", _gt()) == 0.0

    def test_scale_conversion_correctness(self):
        """Boxes at extremes of 1000 scale convert correctly to [0,1]."""
        completion = _make_completion(excavator_1000=[[0, 0, 1000, 1000]])
        gt = _gt(excavator_01=[[0.0, 0.0, 1.0, 1.0]])
        score = compute_reward(completion, gt)
        # Perfect overlap → excavator score = 1.0, TN = 0.10
        assert score == pytest.approx((1.0 + 0.10 + 0.10) / 3, abs=0.02)

    def test_multiple_classes_independently_scored(self):
        """Each class is scored independently."""
        completion = _make_completion(
            excavator_1000=[[100, 50, 800, 700]],
            rebar_1000=[],
        )
        gt = _gt(
            excavator_01=[[0.1, 0.05, 0.8, 0.7]],  # TP
            rebar_01=[[0.2, 0.2, 0.5, 0.5]],        # FN (missed)
        )
        score = compute_reward(completion, gt)
        # excavator: TP perfect → 1.0, rebar: FN → 0.0, hat: TN → 0.10
        assert score == pytest.approx(1.10 / 3.0, abs=0.02)