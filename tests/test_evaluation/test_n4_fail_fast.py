"""
N4: Tests that all metric functions raise ValueError on invalid inputs
instead of silently returning empty dicts.
"""
import pytest


class TestViolationMetricsFailFast:
    """compute_violation_metrics must raise on invalid inputs."""

    def test_empty_predictions(self):
        from evaluation.metrics_violations import compute_violation_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_violation_metrics([], [{"caption": "x"}])

    def test_empty_references(self):
        from evaluation.metrics_violations import compute_violation_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_violation_metrics([{}], [])

    def test_length_mismatch(self):
        from evaluation.metrics_violations import compute_violation_metrics
        with pytest.raises(ValueError, match="length mismatch"):
            compute_violation_metrics([{}, {}], [{}])


class TestGroundingMetricsFailFast:
    """compute_grounding_metrics must raise on invalid inputs."""

    def test_empty_predictions(self):
        from evaluation.metrics_grounding import compute_grounding_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_grounding_metrics([], [{"excavator": []}])

    def test_empty_references(self):
        from evaluation.metrics_grounding import compute_grounding_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_grounding_metrics([{}], [])

    def test_length_mismatch(self):
        from evaluation.metrics_grounding import compute_grounding_metrics
        with pytest.raises(ValueError, match="length mismatch"):
            compute_grounding_metrics([{}, {}], [{}])


class TestCaptionMetricsFailFast:
    """compute_all_caption_metrics must raise on invalid inputs."""

    def test_empty_predictions(self):
        from evaluation.metrics_captioning import compute_all_caption_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_all_caption_metrics([], ["reference text"])

    def test_empty_references(self):
        from evaluation.metrics_captioning import compute_all_caption_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_all_caption_metrics(["pred text"], [])

    def test_length_mismatch(self):
        from evaluation.metrics_captioning import compute_all_caption_metrics
        with pytest.raises(ValueError, match="length mismatch"):
            compute_all_caption_metrics(["a", "b"], ["a"])


class TestStructuralMetricsFailFast:
    """compute_structural_metrics must raise on invalid inputs."""

    def test_empty_outputs(self):
        from evaluation.metrics_structural import compute_structural_metrics
        with pytest.raises(ValueError, match="non-empty"):
            compute_structural_metrics([])
