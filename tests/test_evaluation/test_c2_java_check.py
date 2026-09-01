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
                images=["dummy"]
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
                images=["dummy"]
            )
        except RuntimeError as e:
            if "Java" in str(e):
                pytest.fail(f"Unexpected Java RuntimeError when Java is available: {e}")
        except Exception:
            pass  # Any non-Java error is fine for this test

def test_java_check_is_gated_on_the_task_producing_text():
    """The fail-fast is required only for tasks that score text: captioning
    metrics directly, and reasoning metrics because they score violation reasons
    through the captioning suite. object_only produces neither, so it must
    evaluate without a JVM — the check used to run unconditionally, before any
    task gating.

    See evaluation/evaluator.py and core/tasks.py for the capability model.
    """
    from evaluation.evaluator import run_full_evaluation

    oo_pred = ('```json\n{"excavator": [], "rebar": [], '
               '"worker_with_white_hard_hat": []}\n```')
    oo_ref = {"excavator": [], "rebar": [], "worker_with_white_hard_hat": []}

    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        res = run_full_evaluation(
            raw_predictions=[oo_pred], references=[oo_ref], images=None,
            task="object_only",
        )
    assert any(k.startswith("grounding_") for k in res["metrics"])

    # ...while a caption-producing task still hard-fails.
    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        with pytest.raises(RuntimeError, match="Java is required"):
            run_full_evaluation(
                raw_predictions=["some prose"], references=[{"caption": "some prose"}],
                images=["dummy"], task="caption_only",
            )
