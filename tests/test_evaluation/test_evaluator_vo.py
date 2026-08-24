import pytest
from evaluation.output_parser import validate_output_for_task
from evaluation.evaluator import run_full_evaluation

def test_validate_output_for_task():
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
    
    validated = validate_output_for_task(valid_vo, task="violations_only")
    assert validated is not None
    assert validated.rule_1_violation.reason == "Something"
    
    # Invalid VO output (missing rule_1_violation entirely, schema requires it as Optional but key must exist if strict)
    # Actually pydantic might add defaults, but let's test a bad field type
    bad_vo = {
        "rule_1_violation": "this should be a dict or None"
    }
    validated_bad = validate_output_for_task(bad_vo, task="violations_only")
    assert validated_bad is None

def test_run_full_evaluation_vo():
    raw_preds = [
        '```json\n{"rule_1_violation": null, "rule_2_violation": null, "rule_3_violation": null, "rule_4_violation": null}\n```'
    ]
    refs = [
        {
            "rule_1_violation": None,
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None
        }
    ]
    
    # This should run without crashing and skip caption/grounding
    # Since we are not providing images, if it tried to run caption metrics, it would crash
    from PIL import Image
    from unittest.mock import patch
    
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(raw_preds, refs, images=[Image.new('RGB', (10, 10))], task="violations_only")
    
    # Check that caption metrics and grounding metrics are excluded or empty
    assert not any(k.startswith("captioning_") for k in res['metrics'].keys())
    assert not any(k.startswith("grounding_") for k in res['metrics'].keys())
    
    # It should have violation metrics
    assert "violation_identification_f1_rule_1" in res['metrics']
    assert "structural_json_validity_rate" in res['metrics']
