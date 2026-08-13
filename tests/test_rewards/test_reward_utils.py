"""
Tests for rewards/reward_utils.py — the shared parsing and utility layer.
These tests run CPU-only and require no GPU.
"""
import json
import pytest
from rewards.reward_utils import (
    _strict_parse,
    _is_violation_present,
    _has_repetition_pathology,
    _ngram_f1,
)


# =============================================================================
# _strict_parse
# =============================================================================

def _valid_payload(**overrides):
    base = {
        "caption": "test",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    base.update(overrides)
    return base


class TestStrictParse:
    def test_valid_fenced_json(self):
        text = "```json\n" + json.dumps(_valid_payload()) + "\n```"
        result = _strict_parse(text)
        assert result is not None
        assert result["caption"] == "test"

    def test_valid_unfenced_json(self):
        text = json.dumps(_valid_payload())
        result = _strict_parse(text)
        assert result is not None

    def test_invalid_json_returns_none(self):
        result = _strict_parse("```json\n{not valid\n```")
        assert result is None

    def test_empty_string_returns_none(self):
        assert _strict_parse("") is None

    def test_schema_invalid_returns_none(self):
        # Missing required 'caption' field
        bad = {"rule_1_violation": None, "excavator": []}
        text = "```json\n" + json.dumps(bad) + "\n```"
        assert _strict_parse(text) is None

    def test_wrong_box_format_returns_none(self):
        # Box has 3 elements instead of 4
        payload = _valid_payload(excavator=[[100, 200, 300]])
        text = "```json\n" + json.dumps(payload) + "\n```"
        assert _strict_parse(text) is None

    def test_boxes_in_1000_scale_valid(self):
        # Qwen outputs boxes in [0,1000]; schema accepts any float
        payload = _valid_payload(excavator=[[100, 50, 800, 700]])
        text = "```json\n" + json.dumps(payload) + "\n```"
        result = _strict_parse(text)
        assert result is not None
        assert result["excavator"] == [[100, 50, 800, 700]]

    def test_violation_with_boxes_valid(self):
        payload = _valid_payload(
            rule_1_violation={"bounding_box": [[100, 100, 500, 500]], "reason": "No PPE"}
        )
        text = "```json\n" + json.dumps(payload) + "\n```"
        result = _strict_parse(text)
        assert result is not None
        assert result["rule_1_violation"]["reason"] == "No PPE"

    def test_repeated_calls_same_result(self):
        """lru_cache must return consistent results."""
        text = "```json\n" + json.dumps(_valid_payload(caption="cached")) + "\n```"
        r1 = _strict_parse(text)
        r2 = _strict_parse(text)
        assert r1 == r2
        assert r1 is r2  # same cached object

    def test_preamble_stripped(self):
        text = "Here is my analysis:\n```json\n" + json.dumps(_valid_payload()) + "\n```"
        result = _strict_parse(text)
        assert result is not None


# =============================================================================
# _is_violation_present
# =============================================================================

class TestIsViolationPresent:
    def test_none_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present(None) is False

    def test_empty_dict_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present({}) is False

    def test_dict_with_boxes_returns_true(self):
        from rewards.reward_utils import _is_violation_present
        v = {"bounding_box": [[100, 100, 500, 500]], "reason": ""}
        assert _is_violation_present(v) is True

    def test_dict_with_reason_returns_true(self):
        from rewards.reward_utils import _is_violation_present
        v = {"bounding_box": [], "reason": "Worker not wearing PPE."}
        assert _is_violation_present(v) is True

    def test_dict_empty_boxes_empty_reason_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        v = {"bounding_box": [], "reason": ""}
        assert _is_violation_present(v) is False

    def test_dict_whitespace_reason_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        v = {"bounding_box": [], "reason": "   "}
        assert _is_violation_present(v) is False

    def test_bool_true_returns_true(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present(True) is True

    def test_bool_false_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present(False) is False

    def test_string_null_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present("null") is False

    def test_string_none_returns_false(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present("none") is False

    def test_string_reason_returns_true(self):
        from rewards.reward_utils import _is_violation_present
        assert _is_violation_present("Worker in blind spot") is True


# =============================================================================
# _has_repetition_pathology
# =============================================================================

class TestRepetitionPathology:
    def _make_parsed(self, excavator_boxes):
        return {
            "caption": "test",
            "rule_1_violation": None,
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
            "excavator": excavator_boxes,
            "rebar": [],
            "worker_with_white_hard_hat": [],
        }

    def test_no_repetition_normal(self):
        parsed = self._make_parsed([[10, 10, 500, 500], [200, 200, 600, 600]])
        assert _has_repetition_pathology(parsed, threshold=3) is False

    def test_repetition_below_threshold(self):
        parsed = self._make_parsed([[10, 10, 500, 500]] * 3)
        assert _has_repetition_pathology(parsed, threshold=3) is False

    def test_repetition_above_threshold(self):
        parsed = self._make_parsed([[10, 10, 500, 500]] * 4)
        assert _has_repetition_pathology(parsed, threshold=3) is True

    def test_repetition_in_violation_field(self):
        parsed = {
            "caption": "test",
            "rule_1_violation": {
                "bounding_box": [[100, 100, 500, 500]] * 5,
                "reason": "No PPE",
            },
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
            "excavator": [],
            "rebar": [],
            "worker_with_white_hard_hat": [],
        }
        assert _has_repetition_pathology(parsed, threshold=3) is True

    def test_empty_parsed_no_pathology(self):
        parsed = {
            "caption": "test",
            "rule_1_violation": None,
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
            "excavator": [],
            "rebar": [],
            "worker_with_white_hard_hat": [],
        }
        assert _has_repetition_pathology(parsed) is False


# =============================================================================
# _ngram_f1
# =============================================================================

class TestNgramF1:
    def test_identical_strings(self):
        assert _ngram_f1("worker not wearing hard hat", "worker not wearing hard hat") == pytest.approx(1.0)

    def test_empty_pred_returns_zero(self):
        assert _ngram_f1("", "some reference") == 0.0

    def test_empty_ref_returns_zero(self):
        assert _ngram_f1("some prediction", "") == 0.0

    def test_no_overlap_returns_zero(self):
        assert _ngram_f1("abc def", "xyz uvw") == 0.0

    def test_partial_overlap(self):
        score = _ngram_f1("worker wearing hard hat", "worker not wearing hard hat")
        assert 0.0 < score < 1.0

    def test_score_between_zero_and_one(self):
        score = _ngram_f1("the worker has no PPE", "worker missing personal protective equipment")
        assert 0.0 <= score <= 1.0