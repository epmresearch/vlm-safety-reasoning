import pytest
from unittest.mock import patch

from evaluation.metrics_reasoning import batch_score_reasoning

@patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
def test_batch_score_reasoning_splitting(mock_metrics):
    """Test that reasons are correctly split into global and rule-specific buckets."""
    
    mock_metrics.return_value = {"bertscore_f1": 0.9, "meteor": 0.8}
    
    preds = [
        {"rule_1_violation": {"reason": "pred_r1_img1"}}, # Image 1
        {"rule_1_violation": {"reason": "pred_r1_img2"}, "rule_2_violation": {"reason": "pred_r2_img2"}}, # Image 2
    ]
    refs = [
        {"rule_1_violation": {"reason": "ref_r1_img1"}}, # Image 1
        {"rule_1_violation": {"reason": "ref_r1_img2"}, "rule_2_violation": {"reason": "ref_r2_img2"}}, # Image 2
    ]
    
    res = batch_score_reasoning(preds, refs)
    
    # compute_all_caption_metrics should be called exactly 3 times:
    # 1. Global (Macro) containing all 3 matching reasons
    # 2. Rule 1 bucket containing 2 matching reasons
    # 3. Rule 2 bucket containing 1 matching reason
    assert mock_metrics.call_count == 3
    
    # Check that the global metrics are prefixed correctly
    assert res["reasoning_macro_bertscore_f1"] == 0.9
    assert res["reasoning_macro_meteor"] == 0.8
    
    # Check that rule 1 and 2 received the mocked scores
    assert res["reasoning_rule_1_bertscore_f1"] == 0.9
    assert res["reasoning_rule_2_bertscore_f1"] == 0.9
    
    # Check that rule 3 and 4 correctly fell back to 0.0 since they had no data
    assert res["reasoning_rule_3_bertscore_f1"] == 0.0
    assert res["reasoning_rule_3_meteor"] == 0.0
    assert res["reasoning_rule_4_bertscore_f1"] == 0.0

def test_batch_score_reasoning_empty():
    """Test fallback logic when there are completely empty lists or no common rules."""
    # Case 1: Empty lists
    res_empty = batch_score_reasoning([], [])
    assert res_empty["reasoning_macro_bertscore_f1"] == 0.0
    assert res_empty["reasoning_rule_1_bertscore_f1"] == 0.0
    
    # Case 2: No overlapping rules (e.g., 100% False Positives and False Negatives)
    preds = [{"rule_1_violation": {"reason": "a"}}]
    refs = [{"rule_2_violation": {"reason": "b"}}]
    
    res_no_overlap = batch_score_reasoning(preds, refs)
    assert res_no_overlap["reasoning_macro_bertscore_f1"] == 0.0
    assert res_no_overlap["reasoning_rule_1_bertscore_f1"] == 0.0
    assert res_no_overlap["reasoning_rule_2_bertscore_f1"] == 0.0
