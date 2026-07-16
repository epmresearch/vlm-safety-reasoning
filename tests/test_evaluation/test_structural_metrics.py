import pytest
from evaluation.metrics_structural import compute_structural_metrics

def test_empty_raw_outputs():
    """Test that empty inputs return an empty dictionary."""
    assert compute_structural_metrics([]) == {}

def test_structural_metrics_all_valid():
    """Test structural metrics with perfectly formatted outputs."""
    outputs = [
        "```json\n{\"caption\": \"safe\"}\n```",
        "{\"caption\": \"test\", \"rule_1_violation\": {}}"
    ]
    res = compute_structural_metrics(outputs)
    assert res["json_validity_rate"] == 1.0
    assert res["schema_adherence_rate"] == 1.0
    assert res["total_samples"] == 2
    assert res["valid_json_count"] == 2

def test_structural_metrics_mixed():
    """Test structural metrics with mixed valid and invalid outputs."""
    outputs = [
        "```json\n{\"caption\": \"safe\"}\n```", # Valid JSON
        "This is just a conversational hallucination without JSON.", # Invalid JSON
        "{\"another_key\": true}" # Valid JSON
    ]
    res = compute_structural_metrics(outputs)
    
    assert res["json_validity_rate"] == 2.0 / 3.0
    assert res["valid_json_count"] == 2
    assert res["total_samples"] == 3
