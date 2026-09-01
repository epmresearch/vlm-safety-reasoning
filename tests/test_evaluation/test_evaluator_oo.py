from unittest.mock import patch

import pytest
from PIL import Image

from evaluation.evaluator import run_full_evaluation
from evaluation.output_parser import parse_output_for_task, validate_output_for_task

OO_PRED = ('```json\n{"excavator": [[100, 200, 300, 400]], "rebar": [], '
           '"worker_with_white_hard_hat": []}\n```')
OO_REF = {
    "image_id": "img_1",
    "excavator": [[0.1, 0.2, 0.3, 0.4]],
    "rebar": [],
    "worker_with_white_hard_hat": [],
}


def test_validate_output_for_task_oo():
    parsed = parse_output_for_task(OO_PRED, task="object_only")
    validated = validate_output_for_task(parsed, task="object_only")
    assert validated is not None
    assert validated.excavator == [[100.0, 200.0, 300.0, 400.0]]
    assert validated.rebar == []


def test_validate_output_for_task_oo_rejects_missing_keys():
    validated = validate_output_for_task({"excavator": []}, task="object_only")
    assert validated is None


def test_validate_output_for_task_oo_rejects_wrong_type():
    validated = validate_output_for_task(
        {"excavator": "not a list", "rebar": [], "worker_with_white_hard_hat": []},
        task="object_only",
    )
    assert validated is None


def test_run_full_evaluation_oo_runs_only_grounding_and_structural():
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(
            [OO_PRED], [OO_REF], images=[Image.new("RGB", (10, 10))], task="object_only"
        )
    m = res["metrics"]

    assert any(k.startswith("grounding_") for k in m)
    assert any(k.startswith("structural_") for k in m)

    # object_only emits no caption, no violations and no reasons — scoring those
    # families would compare its output against ground truth it never predicts.
    assert not any(k.startswith("captioning_") for k in m)
    assert not any(k.startswith("violation_") for k in m)
    assert not any(k.startswith("reasoning_") for k in m)


def test_run_full_evaluation_oo_needs_no_java():
    """object_only scores no text at all, so it must not hard-fail without a JVM.
    The Java fail-fast used to run unconditionally, before any task gating."""
    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        res = run_full_evaluation(
            [OO_PRED], [OO_REF], images=[Image.new("RGB", (10, 10))], task="object_only"
        )
    assert any(k.startswith("grounding_") for k in res["metrics"])


def test_run_full_evaluation_oo_needs_no_images():
    """Nor does it need pixels: CLIPScore belongs to the text families."""
    res = run_full_evaluation([OO_PRED], [OO_REF], images=None, task="object_only")
    assert any(k.startswith("grounding_") for k in res["metrics"])


def test_run_full_evaluation_oo_records_parse_failures():
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(
            ["total garbage"], [OO_REF], images=[Image.new("RGB", (10, 10))],
            task="object_only",
        )
    assert len(res["failures"]) == 1
    assert res["failures"][0]["error_type"] == "json_parse_error"
    assert res["metrics"]["structural_json_validity_rate"] == 0.0


def test_run_full_evaluation_violation_capable_tasks_still_need_java():
    """The gate must not have loosened for the existing pipelines: reasoning
    metrics score violation reasons through the captioning suite, which needs a
    JVM."""
    vo_pred = ('```json\n{"rule_1_violation": null, "rule_2_violation": null, '
               '"rule_3_violation": null, "rule_4_violation": null}\n```')
    vo_ref = {f"rule_{i}_violation": None for i in range(1, 5)}
    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        with pytest.raises(RuntimeError, match="Java is required"):
            run_full_evaluation(
                [vo_pred], [vo_ref], images=[Image.new("RGB", (10, 10))],
                task="violations_only",
            )
