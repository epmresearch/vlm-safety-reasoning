"""
Test that run_full_evaluation raises RuntimeError when Java is not available.
"""
import pytest
from unittest.mock import patch


def test_java_missing_raises_runtime_error():
    """C2: run_full_evaluation must fail-fast if Java is absent."""
    from evaluation.evaluator import run_full_evaluation

    # Mock at the source module where _check_java_available is defined,
    # since evaluator.py imports it locally inside run_full_evaluation.
    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        with pytest.raises(RuntimeError, match="Java is required"):
            run_full_evaluation(
                raw_predictions=["test"],
                references=[{"caption": "test"}],
            )


def test_java_present_does_not_raise():
    """Sanity check: when Java IS available, we should NOT get RuntimeError.
    
    The pipeline will proceed past the Java check. It may fail for other
    reasons (e.g., model parsing), but NOT RuntimeError about Java.
    """
    from evaluation.evaluator import run_full_evaluation

    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        try:
            run_full_evaluation(
                raw_predictions=["test"],
                references=[{"caption": "test"}],
            )
        except RuntimeError as e:
            if "Java" in str(e):
                pytest.fail(f"Unexpected Java RuntimeError when Java is available: {e}")
        except Exception:
            pass  # Any non-Java error is fine for this test
