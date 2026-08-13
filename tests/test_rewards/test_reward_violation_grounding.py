"""Tests for rewards/reward_violation_grounding.py."""
import json
import pytest
from rewards.reward_violation_grounding import compute_reward


def _completion_with_violation(rule, boxes_1000, reason="violation"):
    payload = {
        "caption": "test",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    payload[f"{rule}_violation"] = {"bounding_box": boxes_1000, "reason": reason}
    return "```json\n" + json.dumps(payload) + "\n```"


def _gt_with_violation(rule, boxes_01, reason="violation"):
    gt = {
        "caption": "test",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
    }
    gt[f"{rule}_violation"] = {"bounding_box": boxes_01, "reason": reason}
    return gt


def _no_violation_completion():
    payload = {
        "caption": "test",
        "rule_1_violation": None, "rule_2_violation": None,
        "rule_3_violation": None, "rule_4_violation": None,
        "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
    }
    return "```json\n" + json.dumps(payload) + "\n```"


class TestRewardViolationGrounding:
    def test_no_common_rules_returns_zero(self):
        """No TP rules → 0.0 (can't grade grounding)."""
        completion = _no_violation_completion()
        gt = _gt_with_violation("rule_1", [[0.1, 0.1, 0.5, 0.5]])
        assert compute_reward(completion, gt) == pytest.approx(0.0)

    def test_perfect_iou_tp(self):
        """TP with perfect box overlap → 1.0."""
        completion = _completion_with_violation("rule_1", [[100, 100, 500, 500]])
        gt = _gt_with_violation("rule_1", [[0.1, 0.1, 0.5, 0.5]])
        score = compute_reward(completion, gt)
        assert score == pytest.approx(1.0, abs=0.02)

    def test_no_overlap_tp(self):
        """TP with completely non-overlapping boxes → near 0.0."""
        completion = _completion_with_violation("rule_1", [[0, 0, 100, 100]])
        gt = _gt_with_violation("rule_1", [[0.5, 0.5, 1.0, 1.0]])
        score = compute_reward(completion, gt)
        assert score == pytest.approx(0.0, abs=0.02)

    def test_partial_iou_tp(self):
        completion = _completion_with_violation("rule_1", [[0, 0, 500, 500]])
        gt = _gt_with_violation("rule_1", [[0.0, 0.0, 1.0, 1.0]])
        score = compute_reward(completion, gt)
        assert 0.0 < score < 1.0

    def test_multiple_tp_rules_averaged(self):
        """Score is mean IoU across all TP rules."""
        payload = {
            "caption": "test",
            "rule_1_violation": {"bounding_box": [[100, 100, 500, 500]], "reason": "v"},
            "rule_2_violation": {"bounding_box": [[600, 600, 900, 900]], "reason": "v"},
            "rule_3_violation": None,
            "rule_4_violation": None,
            "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
        }
        completion = "```json\n" + json.dumps(payload) + "\n```"
        gt = {
            "caption": "test",
            "rule_1_violation": {"bounding_box": [[0.1, 0.1, 0.5, 0.5]], "reason": "v"},
            "rule_2_violation": {"bounding_box": [[0.6, 0.6, 0.9, 0.9]], "reason": "v"},
            "rule_3_violation": None, "rule_4_violation": None,
            "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
        }
        score = compute_reward(completion, gt)
        assert score == pytest.approx(1.0, abs=0.02)

    def test_invalid_completion_returns_zero(self):
        gt = _gt_with_violation("rule_1", [[0.1, 0.1, 0.5, 0.5]])
        assert compute_reward("bad json", gt) == pytest.approx(0.0)