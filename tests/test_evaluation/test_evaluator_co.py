from unittest.mock import patch

import pytest
from PIL import Image

from evaluation.evaluator import run_full_evaluation
from evaluation.metrics_structural import compute_structural_metrics
from evaluation.output_parser import (
    is_clean_prose,
    parse_output_for_task,
    serialize_output_for_task,
    validate_output_for_task,
)

PROSE = ("Two workers in white hard hats stand beside a yellow excavator on a "
         "muddy site; bundled rebar is stacked at the left edge.")
CO_REF = {
    "image_id": "img_1",
    "caption": "Two workers wearing white hard hats next to an excavator, with rebar nearby.",
}


# ---------------------------------------------------------------------------
# Plain-text parsing contract
# ---------------------------------------------------------------------------

def test_parse_output_for_task_co_wraps_bare_prose():
    assert parse_output_for_task(PROSE, task="caption_only") == {"caption": PROSE}


def test_parse_output_for_task_co_unwraps_a_stray_fence():
    assert parse_output_for_task("```\n%s\n```" % PROSE, task="caption_only") == {"caption": PROSE}


def test_parse_output_for_task_co_unwraps_a_stray_json_object():
    got = parse_output_for_task('{"caption": "%s"}' % PROSE, task="caption_only")
    assert got == {"caption": PROSE}


def test_parse_output_for_task_co_joins_a_caption_list():
    got = parse_output_for_task('{"caption": ["First sentence.", "Second."]}',
                                task="caption_only")
    assert got == {"caption": "First sentence. Second."}


def test_parse_output_for_task_co_unwraps_a_json_string_literal():
    assert parse_output_for_task('"%s"' % PROSE, task="caption_only") == {"caption": PROSE}


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_parse_output_for_task_co_rejects_blank(empty):
    assert parse_output_for_task(empty, task="caption_only") is None


def test_parse_output_for_task_co_rejects_json_without_a_caption():
    assert parse_output_for_task('{"excavator": []}', task="caption_only") is None


def test_parse_output_for_task_json_tasks_are_unchanged():
    """The plain-text branch must not touch the three fenced-JSON tasks."""
    from evaluation.output_parser import parse_model_output

    raw = '```json\n{"rule_1_violation": null}\n```'
    for task in ("unified", "violations_only", "object_only"):
        assert parse_output_for_task(raw, task=task) == parse_model_output(raw)


def test_serialize_output_for_task_round_trips_co():
    payload = parse_output_for_task(PROSE, task="caption_only")
    assert serialize_output_for_task(payload, "caption_only") == PROSE


def test_is_clean_prose():
    assert is_clean_prose(PROSE)
    assert not is_clean_prose("```json\n{}\n```")
    assert not is_clean_prose('{"caption": "x"}')
    assert not is_clean_prose('"caption": x')


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_validate_output_for_task_co():
    validated = validate_output_for_task({"caption": PROSE}, task="caption_only")
    assert validated is not None
    assert validated.caption == PROSE


@pytest.mark.parametrize("bad", [{"caption": ""}, {"caption": "   "}, {}, {"caption": 5}])
def test_validate_output_for_task_co_rejects_bad_captions(bad):
    assert validate_output_for_task(bad, task="caption_only") is None


# ---------------------------------------------------------------------------
# Structural metrics use the task's wire format
# ---------------------------------------------------------------------------

def test_structural_metrics_co_counts_prose_as_valid():
    m = compute_structural_metrics([PROSE, "", "```json\n{\"caption\":\"x\"}\n```"],
                                   task="caption_only")
    # prose -> ok, blank -> not ok, fenced JSON with a caption -> recoverable
    assert m["structural_valid_json_count"] == 2
    assert m["structural_valid_schema_count"] == 2
    assert m["structural_total_samples_count"] == 3


# ---------------------------------------------------------------------------
# Evaluator gating
# ---------------------------------------------------------------------------

def test_run_full_evaluation_co_runs_only_captioning_and_structural():
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True), \
         patch("evaluation.metrics_captioning.compute_all_caption_metrics") as mock_caps:
        mock_caps.return_value = {"captioning_bertscore_f1": 0.9}
        # Patch at the evaluator's import site too.
        with patch("evaluation.evaluator.compute_all_caption_metrics",
                   return_value={"captioning_bertscore_f1": 0.9}):
            res = run_full_evaluation(
                [PROSE], [CO_REF], images=[Image.new("RGB", (10, 10))],
                task="caption_only",
            )
    m = res["metrics"]

    assert "captioning_bertscore_f1" in m
    assert any(k.startswith("structural_") for k in m)

    # caption_only emits no boxes and no violations.
    assert not any(k.startswith("grounding_") for k in m)
    assert not any(k.startswith("violation_") for k in m)
    assert not any(k.startswith("reasoning_") for k in m)


def test_run_full_evaluation_co_requires_java():
    """caption_only scores text (METEOR / CIDEr-D), so the JVM fail-fast must
    still fire for it."""
    with patch("evaluation.metrics_captioning._check_java_available", return_value=False):
        with pytest.raises(RuntimeError, match="Java is required"):
            run_full_evaluation(
                [PROSE], [CO_REF], images=[Image.new("RGB", (10, 10))],
                task="caption_only",
            )


def test_run_full_evaluation_co_requires_images():
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        with pytest.raises(ValueError, match="requires `images`"):
            run_full_evaluation([PROSE], [CO_REF], images=None, task="caption_only")


def test_run_full_evaluation_co_blank_prediction_is_a_parse_failure():
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True), \
         patch("evaluation.evaluator.compute_all_caption_metrics", return_value={}):
        res = run_full_evaluation(
            [""], [CO_REF], images=[Image.new("RGB", (10, 10))], task="caption_only"
        )
    assert len(res["failures"]) == 1
    assert res["failures"][0]["error_type"] == "json_parse_error"
