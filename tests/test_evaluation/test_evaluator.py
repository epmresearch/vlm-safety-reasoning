import pytest
from unittest.mock import patch

from evaluation.evaluator import run_full_evaluation

@patch("evaluation.evaluator.compute_structural_metrics")
@patch("evaluation.evaluator.compute_all_caption_metrics")
@patch("evaluation.evaluator.compute_grounding_metrics")
@patch("evaluation.evaluator.compute_violation_metrics")
@patch("evaluation.evaluator.batch_score_reasoning")
def test_run_full_evaluation(mock_reasoning, mock_violation, mock_grounding, mock_caption, mock_structural):
    """Test that evaluator orchestrates all sub-modules correctly and aggregates dictionaries."""
    
    # Mock return values for all modules
    mock_structural.return_value = {"json_validity_rate": 1.0}
    mock_caption.return_value = {"bertscore_f1": 0.9}
    mock_grounding.return_value = {"grounding_iou": 0.8}
    mock_violation.return_value = {"violation_macro_f1": 0.7}
    mock_reasoning.return_value = {"reasoning_macro_bertscore_f1": 0.6}
    
    # Inputs
    raw_predictions = ["```json\n{\"caption\": \"safe\"}\n```"]
    references = [{"caption": "safe_gt"}]
    
    # Run
    res = run_full_evaluation(raw_predictions, references)
    
    # Assert all sub-modules were called
    mock_structural.assert_called_once()
    mock_caption.assert_called_once()
    mock_grounding.assert_called_once()
    mock_violation.assert_called_once()
    mock_reasoning.assert_called_once()
    
    # Verify exact argument passing to the caption module (ensures the parser worked)
    args, _ = mock_caption.call_args
    assert args[0] == ["safe"] # Extracted caption from raw_predictions string
    assert args[1] == ["safe_gt"] # Ground truth caption
    
    # Verify final aggregation structure
    metrics = res["metrics"]
    assert metrics["json_validity_rate"] == 1.0
    assert metrics["bertscore_f1"] == 0.9
    assert metrics["grounding_iou"] == 0.8
    assert metrics["violation_macro_f1"] == 0.7
    assert metrics["reasoning_macro_bertscore_f1"] == 0.6
    
    # Verify parsed outputs are returned so they can be logged
    assert len(res["parsed_predictions"]) == 1
    assert res["parsed_predictions"][0] == {"caption": "safe"}

def test_run_full_evaluation_empty_inputs():
    """Test that evaluator doesn't crash on empty inputs."""
    res = run_full_evaluation([], [])
    assert "metrics" in res
    assert "parsed_predictions" in res
    assert res["parsed_predictions"] == []
