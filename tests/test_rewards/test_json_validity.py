import pytest
from rewards.unified_reward import compute_reward

def test_json_validity_reward():
    valid_pred = '''```json
{"caption": "test", "excavator": [], "rule_1_violation": null}
```'''
    invalid_pred = '''```json
{"caption": "test", 
```'''
    
    gt = {"caption": "test", "excavator": [], "rule_1_violation": None}
    
    valid_score = compute_reward(valid_pred, gt)
    invalid_score = compute_reward(invalid_pred, gt)
    
    assert valid_score > invalid_score