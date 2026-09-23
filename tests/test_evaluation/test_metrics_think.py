"""Tests for the <think>-block diagnostics.

These keys are the ONLY visibility into the block after training: no reward reads it,
no other metric scores it, and structural repair treats it as preamble. A run could
emit a truncated or self-contradicting block on all 3,004 images and every other number
would look normal.
"""
import json

import pytest
from unittest.mock import patch

from core.think_format import THINK_CLOSE, THINK_OPEN
from evaluation.metrics_think import compute_think_metrics, task_expects_think_block

CAPTION = "A worker stands beside an excavator on a muddy site."


def _block(lines):
    return f"{THINK_OPEN}\n" + "\n".join(lines) + f"\n{THINK_CLOSE}\n"


def _payload(flags):
    d = {
        f"rule_{i}_violation": (
            {"bounding_box": [[100, 100, 200, 200]], "reason": "x"} if flags[i - 1] else None
        )
        for i in range(1, 5)
    }
    return "```json\n" + json.dumps(d, separators=(",", ":")) + "\n```"


def _pred(flags):
    return {
        f"rule_{i}_violation": ({"reason": "x"} if flags[i - 1] else None)
        for i in range(1, 5)
    }


GOOD_LINES = [CAPTION, "rule_1: no hard hat -> yes (1)", "rule_2: no", "rule_3: no", "rule_4: no"]


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,expected", [
    ("violations_think", True),
    ("violations_only", False),
    ("unified", False),
    ("object_only", False),
    ("caption_only", False),
])
def test_gating_is_derived_from_the_prompt(task, expected):
    """Derived, not hardcoded against a task name: core/tasks.py forbids task-name
    literals, and "were we asking for a block?" is the actual question."""
    assert task_expects_think_block(task) is expected


def test_gating_survives_an_unresolvable_prompt():
    with patch("data.prompt_templates.get_prompt_for_task", side_effect=ValueError("boom")):
        assert task_expects_think_block("violations_think") is False


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

def test_a_perfect_run():
    m = compute_think_metrics([_block(GOOD_LINES) + _payload([1, 0, 0, 0])], [_pred([1, 0, 0, 0])])
    assert m["think_block_present_rate"] == 1.0
    assert m["think_block_closed_rate"] == 1.0
    assert m["think_block_wellformed_rate"] == 1.0
    assert m["think_verdict_json_agreement_rate"] == 1.0
    assert m["think_verdict_json_comparable_count"] == 4
    assert m["think_top_problem"] == ""


def test_the_interesting_failure_block_disagrees_with_the_json():
    """The model wrote 'rule_2: no' and then reported a rule_2 violation -- reasoning
    that does not describe its own answer."""
    m = compute_think_metrics(
        [_block(GOOD_LINES) + _payload([1, 1, 0, 0])], [_pred([1, 1, 0, 0])]
    )
    assert m["think_block_wellformed_rate"] == 1.0, "the block itself is fine"
    assert m["think_verdict_json_agreement_rate"] == pytest.approx(3 / 4)
    assert m["think_verdict_json_agreement_rate_rule_2"] == 0.0
    assert m["think_verdict_json_agreement_rate_rule_1"] == 1.0


def test_a_truncated_block_is_counted_as_present_but_not_closed():
    """An unclosed block is what a completion that hit max_completion_length looks
    like, and truncation is otherwise indistinguishable from a bad model: every reward
    component returns exactly 0.0 for both."""
    m = compute_think_metrics([f"{THINK_OPEN}\n{CAPTION}\nrule_1: no hard ha"], [None])
    assert m["think_block_present_rate"] == 1.0
    assert m["think_block_closed_rate"] == 0.0
    assert m["think_verdict_json_comparable_count"] == 0, "no JSON to agree with"


def test_no_block_at_all():
    m = compute_think_metrics([_payload([0, 0, 0, 0])], [_pred([0, 0, 0, 0])])
    assert m["think_block_present_count"] == 0
    assert m["think_block_present_rate"] == 0.0
    # Empty denominators report 0.0 next to a 0 count, so a reader can tell them from a
    # real zero.
    assert m["think_block_closed_rate"] == 0.0
    assert m["think_verdict_json_comparable_count"] == 0


def test_agreement_is_counted_per_rule_slot_not_per_record():
    """A block that gets three rules right and one wrong must not score the same as one
    that gets all four wrong."""
    three_right = _block(GOOD_LINES) + _payload([1, 1, 0, 0])      # rule_2 differs
    all_wrong = _block([CAPTION, "rule_1: no", "rule_2: no", "rule_3: no", "rule_4: no"]) \
        + _payload([1, 1, 1, 1])
    a = compute_think_metrics([three_right], [_pred([1, 1, 0, 0])])
    b = compute_think_metrics([all_wrong], [_pred([1, 1, 1, 1])])
    assert a["think_verdict_json_agreement_rate"] == pytest.approx(0.75)
    assert b["think_verdict_json_agreement_rate"] == 0.0


def test_unparseable_json_contributes_no_agreement_slots():
    m = compute_think_metrics([_block(GOOD_LINES) + "garbage"], [None])
    assert m["think_block_present_rate"] == 1.0
    assert m["think_verdict_json_comparable_count"] == 0


def test_word_stats_and_problem_reporting():
    malformed = _block([CAPTION, "rule_1: no", "rule_2: no"]) + _payload([0, 0, 0, 0])
    m = compute_think_metrics([malformed], [_pred([0, 0, 0, 0])])
    assert m["think_block_wellformed_rate"] == 0.0
    assert m["think_block_problem_count_mean"] > 0
    assert m["think_top_problem"]
    assert m["think_block_words_max"] >= m["think_block_words_p50"] > 0


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="align"):
        compute_think_metrics(["a", "b"], [None])


def test_no_crash_on_hostile_input():
    m = compute_think_metrics([None, 42, "", THINK_OPEN], [None, None, None, None])
    assert m["think_total_samples_count"] == 4


# ---------------------------------------------------------------------------
# Evaluator wiring
# ---------------------------------------------------------------------------

def _safe_case():
    raw = ['```json\n{"rule_1_violation": null, "rule_2_violation": null, '
           '"rule_3_violation": null, "rule_4_violation": null}\n```']
    refs = [{f"rule_{i}_violation": None for i in range(1, 5)}]
    from PIL import Image
    return raw, refs, [Image.new("RGB", (10, 10))]


def test_evaluator_emits_no_think_keys_for_violations_only():
    """v2 and v4 metrics.json must stay free of these keys -- absent, never zero."""
    from evaluation.evaluator import run_full_evaluation
    raw, refs, imgs = _safe_case()
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(raw, refs, images=imgs, task="violations_only",
                                  model_texts=[_block(GOOD_LINES) + raw[0]])
    assert not any(k.startswith("think_") for k in res["metrics"])


def test_evaluator_emits_think_keys_for_violations_think():
    from evaluation.evaluator import run_full_evaluation
    raw, refs, imgs = _safe_case()
    model_text = _block([CAPTION, "rule_1: no", "rule_2: no", "rule_3: no", "rule_4: no"]) + raw[0]
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(raw, refs, images=imgs, task="violations_think",
                                  model_texts=[model_text])
    assert res["metrics"]["think_block_present_rate"] == 1.0
    assert res["metrics"]["think_verdict_json_agreement_rate"] == 1.0


def test_evaluator_skips_think_metrics_when_model_texts_is_missing():
    """Absent keys, not keys computed from the REPAIRED text. structural_repair replaces
    raw_output with re-serialized JSON for every record it fixed, so a fallback would
    report a block-presence rate that is partly measuring the repair stage -- this repo
    has already paid for one metric that silently described post-repair output."""
    from evaluation.evaluator import run_full_evaluation
    raw, refs, imgs = _safe_case()
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(raw, refs, images=imgs, task="violations_think")
    assert not any(k.startswith("think_") for k in res["metrics"])


def test_run_evaluation_prefers_original_raw_output():
    """The rule that recovers the model's real text from a repaired file: repair sets
    original_raw_output ONLY for records it changed (status fixed_valid), and leaves
    raw_output alone otherwise."""
    records = [
        {"raw_output": "REPAIRED", "original_raw_output": "MODEL"},   # repair rewrote it
        {"raw_output": "MODEL2"},                                     # untouched
        {},                                                           # defensive
    ]
    texts = [r.get("original_raw_output", r.get("raw_output", "")) for r in records]
    assert texts == ["MODEL", "MODEL2", ""]
