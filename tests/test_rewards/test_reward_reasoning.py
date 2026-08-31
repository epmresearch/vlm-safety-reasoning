import pytest
import math
import json
from rewards.reward_reasoning import compute_reward

def _make_valid_payload(rule_1_reason=""):
    return {
        "caption": "test",
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
        "rule_1_violation": {"bounding_box": [[100,200,300,400]], "reason": rule_1_reason},
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None
    }

def _make_completion(reason=""):
    return "```json\n" + json.dumps(_make_valid_payload(reason)) + "\n```"

def _make_gt(reason=""):
    gt = _make_valid_payload(reason)
    # GT boxes are 01 scale
    gt["rule_1_violation"]["bounding_box"] = [[0.1, 0.2, 0.3, 0.4]]
    return gt

def test_invalid_json():
    completion = "not a json string"
    gt = {}
    assert compute_reward([completion], [gt])[0] == 0.0

def test_no_common_rules():
    # Model predicts rule_1, but GT only has rule_2
    completion = _make_completion("test")
    gt = _make_valid_payload("test")
    gt["rule_1_violation"] = None
    gt["rule_2_violation"] = {"bounding_box": [[0.1,0.2,0.3,0.4]], "reason": "test"}
    assert compute_reward([completion], [gt])[0] == 0.0

def test_missing_reason_in_gt_or_pred():
    # Common rule but missing reason
    completion = _make_completion("")
    gt = _make_gt("actual reason")
    assert compute_reward([completion], [gt])[0] == 0.0

    completion2 = _make_completion("some reason")
    gt2 = _make_gt("")
    assert compute_reward([completion2], [gt2])[0] == 0.0

def test_identical_reasons():
    # Same reason should give ~1.0
    completion = _make_completion("The worker is not wearing a hardhat.")
    gt = _make_gt("The worker is not wearing a hardhat.")
    
    score = compute_reward([completion], [gt])[0]
    assert score > 0.95 # Almost perfect (floating point differences in embeddings)

def test_semantic_similarity_different_words():
    # Similar meaning, different words. Should score high but not 1.0.
    completion = _make_completion("A guy without a helmet is seen.")
    gt = _make_gt("The construction worker is missing a hard hat.")
    
    score = compute_reward([completion], [gt])[0]
    assert 0.2 < score < 0.95

def test_length_penalty():
    # Model rambles on for too long compared to GT
    completion = _make_completion("The worker is not wearing a hardhat. Hardhats are very important for safety on a construction site. Without a hardhat, things can fall on your head. This is a severe violation of the safety protocols established in 1992.")
    gt = _make_gt("Missing hard hat.")
    
    score = compute_reward([completion], [gt])[0]
    # The length factor should severely penalize this
    assert score < 0.5


def _safe_payload():
    return {
        "caption": "test",
        "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
        "rule_1_violation": None, "rule_2_violation": None,
        "rule_3_violation": None, "rule_4_violation": None,
    }


class TestTrueNegativeBranch:
    """Coverage for the both-empty true-negative branch (reward_reasoning.py:47).

    Previously untested — the existing "no common rules" test exercises a rule mismatch
    (an FN), not a genuine both-empty true negative, so this constant could have changed
    silently. Must stay in step with the other three TN sites (CLAUDE.md invariant #6).
    """

    def test_correctly_safe_returns_tn_constant(self):
        completion = "```json\n" + json.dumps(_safe_payload()) + "\n```"
        gt = _safe_payload()
        assert compute_reward([completion], [gt])[0] == pytest.approx(0.15)

    def test_false_alarm_on_safe_image_returns_zero(self):
        completion = _make_completion("Worker without a hard hat.")
        gt = _safe_payload()
        assert compute_reward([completion], [gt])[0] == pytest.approx(0.0)

    def test_contentless_violation_object_is_not_a_true_negative(self):
        """Asserting a violation with no reason is a false alarm on a safe image,
        not a correct safe call — so it must not collect the TN reward."""
        payload = _safe_payload()
        payload["rule_1_violation"] = {"bounding_box": [], "reason": ""}
        completion = "```json\n" + json.dumps(payload) + "\n```"
        gt = _safe_payload()
        assert compute_reward([completion], [gt])[0] == pytest.approx(0.0)
