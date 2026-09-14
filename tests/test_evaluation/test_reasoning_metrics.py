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
    
    res = batch_score_reasoning(preds, refs, images=["img1", "img2"])
    
    # compute_all_caption_metrics should be called exactly 3 times:
    # 1. Global (Macro) containing all 3 matching reasons
    # 2. Rule 1 bucket containing 2 matching reasons
    # 3. Rule 2 bucket containing 1 matching reason
    assert mock_metrics.call_count == 3
    
    # Check that the global metrics are prefixed correctly for pooled (micro)
    assert res["reasoning_text_similarity_bertscore_f1_micro"] == 0.9
    assert res["reasoning_text_similarity_meteor_micro"] == 0.8
    
    # Macro averages only the rules that were actually MEASURED.
    # Rules 3 and 4 have no true positives here, so there was no reasoning text
    # to score for them -- that is missing data, not a score of zero.
    #   correct : (0.9 + 0.9) / 2 = 0.9   over n_rules = 2
    #   old bug : (0.9 + 0.9 + 0.0 + 0.0) / 4 = 0.45
    # The old form is what reported vo-2b-sft's reasoning as 0.391 when the two
    # rules it could be scored on averaged 0.782.
    assert res["reasoning_text_similarity_bertscore_f1_macro"] == 0.9
    assert res["reasoning_text_similarity_meteor_macro"] == 0.8
    assert res["reasoning_text_similarity_bertscore_f1_macro_n_rules"] == 2
    assert res["reasoning_text_similarity_meteor_macro_n_rules"] == 2
    
    # Check that rule 1 and 2 received the mocked scores
    assert res["reasoning_text_similarity_bertscore_f1_rule_1"] == 0.9
    assert res["reasoning_text_similarity_bertscore_f1_rule_2"] == 0.9
    
    # Check that rule 3 and 4 correctly fell back to 0.0 since they had no data.
    # The PER-RULE fallback stays 0.0 (downstream CSV/chart code expects the key
    # to exist for every rule); only the MACRO stops averaging it in.
    assert res["reasoning_text_similarity_bertscore_f1_rule_3"] == 0.0
    assert res["reasoning_text_similarity_meteor_rule_3"] == 0.0
    assert res["reasoning_text_similarity_bertscore_f1_rule_4"] == 0.0

    # scored_count is now always emitted, including as 0, so the sample size a
    # reasoning score rests on is never a blank cell in the comparison CSV.
    assert res["reasoning_text_similarity_scored_count_rule_3"] == 0
    assert res["reasoning_text_similarity_scored_count_rule_4"] == 0

def test_batch_score_reasoning_empty():
    """Test fallback logic when there are completely empty lists or no common rules."""
    # Case 1: Empty lists - raises ValueError (N4 fix)
    with pytest.raises(ValueError, match="non-empty"):
        batch_score_reasoning([], [], images=[])
    
    # Case 2: No overlapping rules (e.g., 100% False Positives and False Negatives)
    preds = [{"rule_1_violation": {"reason": "a"}}]
    refs = [{"rule_2_violation": {"reason": "b"}}]
    
    res_no_overlap = batch_score_reasoning(preds, refs, images=["img1"])
    # Nothing measurable at all -> macro stays 0.0 (the key must still exist),
    # and n_rules == 0 says explicitly that the 0.0 is "no data", not "scored 0".
    assert res_no_overlap["reasoning_text_similarity_bertscore_f1_macro"] == 0.0
    assert res_no_overlap["reasoning_text_similarity_bertscore_f1_macro_n_rules"] == 0
    assert res_no_overlap["reasoning_text_similarity_bertscore_f1_rule_1"] == 0.0
    assert res_no_overlap["reasoning_text_similarity_bertscore_f1_rule_2"] == 0.0
