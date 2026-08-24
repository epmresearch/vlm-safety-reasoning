"""Tests for rewards/reward_format.py."""
import json
import pytest
from rewards.reward_format import compute_reward


def _make_valid(caption="test", **overrides):
    base = {
        "caption": caption,
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    base.update(overrides)
    return "```json\n" + json.dumps(base) + "\n```"


GT = {"caption": "test", "rule_1_violation": None, "rule_2_violation": None,
      "rule_3_violation": None, "rule_4_violation": None,
      "excavator": [], "rebar": [], "worker_with_white_hard_hat": []}


class TestRewardFormat:
    def test_valid_json_valid_schema_returns_one(self):
        assert compute_reward(_make_valid(), GT) == 1.0

    def test_invalid_json_returns_zero(self):
        assert compute_reward("```json\n{broken\n```", GT) == 0.0

    def test_empty_string_returns_zero(self):
        assert compute_reward("", GT) == 0.0

    def test_valid_json_missing_caption_returns_zero(self):
        bad = {"rule_1_violation": None, "excavator": []}
        text = "```json\n" + json.dumps(bad) + "\n```"
        assert compute_reward(text, GT) == 0.0

    def test_valid_json_wrong_box_format_returns_zero(self):
        # Box with 3 coords instead of 4
        bad_payload = {**GT, "excavator": [[100, 200, 300]]}
        text = "```json\n" + json.dumps(bad_payload) + "\n```"
        assert compute_reward(text, GT) == 0.0

    def test_valid_with_violation_returns_one(self):
        text = _make_valid(
            rule_1_violation={"bounding_box": [[100, 100, 500, 500]], "reason": "No PPE"}
        )
        assert compute_reward(text, GT) == 1.0

    def test_ground_truth_not_used(self):
        """Format reward doesn't depend on ground truth."""
        valid = _make_valid()
        assert compute_reward(valid, {}) == compute_reward(valid, GT)

    def test_exception_in_reward_returns_zero(self):
        """_safe_reward decorator must catch exceptions."""
        assert compute_reward(None, GT) == 0.0  # type: ignore