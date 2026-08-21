import pytest
import json
from rewards.reward_caption import compute_reward

def _make_valid_payload(caption=""):
    return {
        "caption": caption,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None
    }

def _make_completion(caption=""):
    return "```json\n" + json.dumps(_make_valid_payload(caption)) + "\n```"

def _make_gt(caption=""):
    return _make_valid_payload(caption)

def test_invalid_json():
    completion = "not a json string"
    gt = _make_gt("A construction site")
    assert compute_reward([completion], [gt])[0] == 0.0

def test_empty_caption_returns_zero():
    completion = _make_completion("")
    gt = _make_gt("A worker is digging.")
    assert compute_reward([completion], [gt])[0] == 0.0

    completion2 = _make_completion("A worker is digging.")
    gt2 = _make_gt("")
    assert compute_reward([completion2], [gt2])[0] == 0.0

def test_identical_captions():
    # Identical captions should have 1.0 semantic, 1.0 lexical, 1.0 length penalty -> 1.0 total
    caption = "A worker in a yellow vest operates a bulldozer."
    completion = _make_completion(caption)
    gt = _make_gt(caption)
    
    score = compute_reward([completion], [gt])[0]
    assert score > 0.95

def test_semantic_similarity_different_words():
    # High semantic match, partial lexical match
    completion = _make_completion("An individual wearing a safety vest drives construction machinery.")
    gt = _make_gt("A worker in a yellow vest operates a bulldozer.")
    
    score = compute_reward([completion], [gt])[0]
    # Should be decent but not perfect (say 0.3 to 0.9)
    assert 0.3 < score < 0.95

def test_completely_unrelated_caption():
    completion = _make_completion("A cat sleeps on the sofa.")
    gt = _make_gt("A worker in a yellow vest operates a bulldozer.")
    
    score = compute_reward([completion], [gt])[0]
    assert score < 0.3

def test_length_penalty_rambling():
    gt_caption = "A bulldozer moves dirt."
    rambling_caption = gt_caption + " " + "It is very sunny today. " * 20
    
    completion = _make_completion(rambling_caption)
    gt = _make_gt(gt_caption)
    
    score = compute_reward([completion], [gt])[0]
    # Even if semantic similarity is somehow okay because it contains the GT,
    # the length penalty (Gaussian) should crush the score.
    assert score < 0.4
