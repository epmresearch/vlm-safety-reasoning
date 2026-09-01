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

class TestTrueNegativeBranch:
    """Coverage for the both-empty true-negative branch (reward_violation_grounding.py:35).

    This branch had NO test coverage at all before — its constant could have changed
    silently. The true-negative value is deliberate: it is the same constant used by
    reward_violation_id / reward_reasoning / reward_grounding, and all four must move
    together or the objective is silently re-biased (see CLAUDE.md invariant #6).
    """

    def _safe_payload(self):
        return {
            "caption": "test",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
        }

    def test_correctly_safe_returns_tn_constant(self):
        completion = "```json\n" + json.dumps(self._safe_payload()) + "\n```"
        gt = self._safe_payload()
        from rewards.reward_utils import reward_constant
        expected = reward_constant("unified", "violation_tn_constant", 0.15)
        assert compute_reward(completion, gt) == pytest.approx(expected)

    def test_false_alarm_on_safe_image_returns_zero(self):
        """Predicting a violation where GT is safe is not a true negative."""
        completion = _completion_with_violation("rule_1", [[100, 100, 500, 500]])
        gt = self._safe_payload()
        assert compute_reward(completion, gt) == pytest.approx(0.0)

    def test_missed_violation_returns_zero(self):
        completion = "```json\n" + json.dumps(self._safe_payload()) + "\n```"
        gt = _gt_with_violation("rule_1", [[0.1, 0.1, 0.5, 0.5]])
        assert compute_reward(completion, gt) == pytest.approx(0.0)

    def test_contentless_violation_object_is_not_a_true_negative(self):
        """A keyed-but-empty violation object asserts a violation, so on a safe image it
        is a false alarm — it must NOT collect the true-negative reward. This closes the
        reward-hacking surface where emitting empty violation objects banked TN credit."""
        payload = self._safe_payload()
        payload["rule_1_violation"] = {"bounding_box": [], "reason": ""}
        completion = "```json\n" + json.dumps(payload) + "\n```"
        gt = self._safe_payload()
        assert compute_reward(completion, gt) == pytest.approx(0.0)
