import pytest
from rewards.json_validity import compute_reward

def test_json_validity_reward():
    valid_pred = '''```json
{"caption": "test", "excavator": [], "rule_1_violation": null}
```'''
    invalid_pred = '''```json
{"caption": "test", 
```'''
    empty_pred = ""
    valid_unfenced = '{"caption": "test", "excavator": [], "rule_1_violation": null}'
    valid_with_preamble = '''Here is the result:
```json
{"caption": "test", "excavator": [], "rule_1_violation": null}
```'''

    gt = {"caption": "test", "excavator": [], "rule_1_violation": None}
    
    assert compute_reward(valid_pred, gt) == 1.0
    assert compute_reward(invalid_pred, gt) == 0.0
    assert compute_reward(empty_pred, gt) == 0.0
    assert compute_reward(valid_unfenced, gt) == 1.0
    assert compute_reward(valid_with_preamble, gt) == 1.0