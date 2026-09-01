import pytest
from unittest.mock import patch

from evaluation.metrics_captioning import (
    compute_all_caption_metrics,
    compute_clipscore
)

def test_empty_inputs():
    """Test that empty/invalid inputs raise ValueError (fail-fast, N4 fix)."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_all_caption_metrics([], [], images=[])
    with pytest.raises(ValueError, match="length mismatch"):
        compute_all_caption_metrics(["text1", "text2"], ["ref1"], images=["img1", "img2"]) # Mismatched length

@patch("evaluation.metrics_captioning.compute_clipscore")
@patch("evaluation.metrics_captioning.compute_cider")
@patch("evaluation.metrics_captioning.compute_meteor")
@patch("evaluation.metrics_captioning.compute_bertscore")
def test_all_caption_metrics_aggregation(mock_bert, mock_meteor, mock_cider, mock_clip):
    """Blank predictions and references are EXCLUDED from the graders and reported
    as their own rate, not rewritten to the literal string "empty".

    The old behaviour handed the graders the word "empty", so a completely failed
    generation earned a real nonzero BERTScore and the failure was invisible in the
    caption metrics.
    """
    mock_bert.return_value = {"bertscore_f1": 0.8}
    mock_meteor.return_value = {"meteor": 0.7}
    mock_cider.return_value = {"cider": 0.6}
    mock_clip.return_value = {"clipscore": 0.5}

    preds = ["A good caption", "   ", ""]      # index 1 and 2 are blank
    refs = ["A great caption", "Also good", " "]  # index 2's reference is blank too

    res = compute_all_caption_metrics(preds, refs, images=["img1", "img2", "img3"])

    # Only the one genuinely-scoreable pair reaches the graders, verbatim.
    args, _ = mock_bert.call_args
    assert args[0] == ["A good caption"]
    assert args[1] == ["A great caption"]
    assert "empty" not in args[0] and "empty" not in args[1]

    # Images are filtered in lockstep so text[i] still matches image[i].
    clip_args, _ = mock_clip.call_args
    assert clip_args[0] == ["A good caption"]
    assert clip_args[1] == ["img1"]

    # The failure is reported rather than hidden.
    assert res["blank_prediction_count"] == 2
    assert res["blank_prediction_rate"] == pytest.approx(2 / 3)
    assert res["blank_reference_count"] == 1
    assert res["scored_count"] == 1
    assert res["total_count"] == 3

    assert res["bertscore_f1"] == 0.8
    assert res["meteor"] == 0.7
    assert res["cider"] == 0.6
    assert res["clipscore"] == 0.5


@patch("evaluation.metrics_captioning.compute_clipscore")
@patch("evaluation.metrics_captioning.compute_cider")
@patch("evaluation.metrics_captioning.compute_meteor")
@patch("evaluation.metrics_captioning.compute_bertscore")
def test_all_blank_returns_only_counters(mock_bert, mock_meteor, mock_cider, mock_clip):
    """With nothing scoreable, no grader is called at all."""
    res = compute_all_caption_metrics(["", "  "], ["a", "b"], images=["i1", "i2"])
    assert res["scored_count"] == 0
    assert res["blank_prediction_count"] == 2
    assert "bertscore_f1" not in res
    mock_bert.assert_not_called()
    mock_clip.assert_not_called()


def test_long_clipscore_omits_itself_when_disabled(monkeypatch):
    """A disabled or failed long-CLIPScore must yield NO key, never a 0.0 that
    would read as a terrible score."""
    import evaluation.metrics_captioning as m
    monkeypatch.setenv("VLM_DISABLE_LONG_CLIP", "1")
    assert m.compute_long_clipscore(["a caption"], ["img"]) == {}

def test_clipscore_mismatched_lengths():
    """Test CLIPScore fails safely with bad inputs."""
    assert compute_clipscore(["text1", "text2"], ["image1"]) == {}
    assert compute_clipscore([], []) == {}
