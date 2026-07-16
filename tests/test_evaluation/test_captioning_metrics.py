import pytest
from unittest.mock import patch

from evaluation.metrics_captioning import (
    compute_all_caption_metrics,
    compute_clipscore
)

def test_empty_inputs():
    """Test that empty or mismatched inputs return safely."""
    assert compute_all_caption_metrics([], []) == {}
    assert compute_all_caption_metrics(["text"], []) == {} # Mismatched length

@patch("evaluation.metrics_captioning.compute_clipscore")
@patch("evaluation.metrics_captioning.compute_cider")
@patch("evaluation.metrics_captioning.compute_meteor")
@patch("evaluation.metrics_captioning.compute_bertscore")
def test_all_caption_metrics_aggregation(mock_bert, mock_meteor, mock_cider, mock_clip):
    """Test that metrics are aggregated correctly and empty strings are sanitized."""
    mock_bert.return_value = {"bertscore_f1": 0.8}
    mock_meteor.return_value = {"meteor": 0.7}
    mock_cider.return_value = {"cider": 0.6}
    mock_clip.return_value = {"clipscore": 0.5}
    
    preds = ["A good caption", "   ", ""] # Includes whitespace and empty string
    refs = ["A great caption", "Also good", " "]
    
    # Test without images
    res = compute_all_caption_metrics(preds, refs)
    
    # Check that sanitization worked before passing to the underlying functions
    args, _ = mock_bert.call_args
    assert args[0] == ["A good caption", "empty", "empty"]
    assert args[1] == ["A great caption", "Also good", "empty"]
    
    assert res["bertscore_f1"] == 0.8
    assert res["meteor"] == 0.7
    assert res["cider"] == 0.6
    assert "clipscore" not in res # Images were not provided
    
    # Test with images
    res_images = compute_all_caption_metrics(preds, refs, images=["img1", "img2", "img3"])
    assert res_images["clipscore"] == 0.5

def test_clipscore_mismatched_lengths():
    """Test CLIPScore fails safely with bad inputs."""
    assert compute_clipscore(["text1", "text2"], ["image1"]) == {}
    assert compute_clipscore([], []) == {}
