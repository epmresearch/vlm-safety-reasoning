import pytest
from evaluation.metrics_structural import compute_structural_metrics

def test_empty_raw_outputs():
    """Test that empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_structural_metrics([])

def test_structural_metrics_all_valid():
    """Test structural metrics with perfectly formatted outputs."""
    outputs = [
        "```json\n{\"caption\": \"safe\"}\n```",
        "{\"caption\": \"danger\", \"rule_1_violation\": {}}"
    ]
    res = compute_structural_metrics(outputs)
    assert res["structural_json_validity_rate"] == 1.0
    assert res["structural_schema_adherence_rate"] == 1.0
    assert res["structural_total_samples_count"] == 2
    assert res["structural_valid_json_count"] == 2

def test_structural_metrics_mixed():
    """Test structural metrics with mixed valid and invalid outputs."""
    outputs = [
        "```json\n{\"caption\": \"safe\"}\n```", # Valid JSON
        "This is just a conversational hallucination without JSON.", # Invalid JSON
        "{\"another_key\": true}" # Valid JSON
    ]
    res = compute_structural_metrics(outputs)
    
    assert res["structural_json_validity_rate"] == 2.0 / 3.0
    assert res["structural_valid_json_count"] == 2
    assert res["structural_total_samples_count"] == 3
