import json
import pytest
from rewards.reward_format import compute_reward_for_task

def test_compute_reward_for_task_vo():
    # Valid VO output
    valid_vo = {
        "rule_1_violation": {
            "bounding_box": [[100, 100, 200, 200]],
            "reason": "Something"
        },
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None
    }
    
    completion = "```json\n" + json.dumps(valid_vo) + "\n```"
    gt = {} # format reward doesn't use gt
    
    score = compute_reward_for_task(completion, gt, task="violations_only")
    assert score == 1.0

    # Invalid VO output
    bad_vo = {
        "rule_1_violation": "bad"
    }
    completion_bad = "```json\n" + json.dumps(bad_vo) + "\n```"
    
    score_bad = compute_reward_for_task(completion_bad, gt, task="violations_only")
    assert score_bad == 0.0
