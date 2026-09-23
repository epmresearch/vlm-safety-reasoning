"""Tests for the violations_think SFT target.

The arm's entire claim is that it differs from violations_only in ONE way: a <think>
block in front of a byte-identical JSON payload. If the payload ever drifts, the
v3-vs-v4 comparison silently becomes a two-variable experiment, and the result is
uninterpretable rather than wrong -- which is worse, because nothing looks broken.
"""
import json

import pytest

from core.think_format import (
    THINK_CLOSE,
    THINK_FIELD,
    THINK_OPEN,
    build_think_body,
    violations_from_row,
)
from data.preprocessor import (
    _GT_BUILDERS,
    _TARGET_BUILDERS,
    build_gt_dict,
    build_target_json,
    raw_sample_to_conversation_for_task,
)
from data.schemas import get_output_schema

TASK = "violations_think"
CAPTION = "A worker stands beside an excavator in a muddy trench."
REASON = "The worker on the left is not wearing a hard hat."


def _row(**overrides):
    row = {
        "image_id": "0001234",
        "image_caption": CAPTION,
        "rule_1_violation": {"bounding_box": [[0.5, 0.5, 0.6, 0.6]], "reason": REASON},
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
    }
    row.update(overrides)
    row.setdefault(THINK_FIELD,
                   build_think_body(row["image_caption"], violations_from_row(row)))
    return row


# ---------------------------------------------------------------------------
# The byte-identity guarantee
# ---------------------------------------------------------------------------

def test_target_ends_with_the_violations_only_target_verbatim():
    """THE load-bearing assertion for this arm."""
    row = _row()
    vo = build_target_json(row, task="violations_only")
    vt = build_target_json(row, task=TASK)
    assert vt.endswith(vo)
    assert vt[: -len(vo)] == f"{THINK_OPEN}\n{row[THINK_FIELD]}\n{THINK_CLOSE}\n"


def test_the_json_payload_is_exactly_what_v2_trains_on():
    """A literal, so a change to the vo payload format trips here rather than being
    discovered as an unexplained v3-vs-v2 difference after 12 jobs."""
    row = _row()
    expected_json = (
        '```json\n'
        '{"rule_1_violation":{"bounding_box":[[500,500,600,600]],'
        f'"reason":"{REASON}"}},'
        '"rule_2_violation":null,"rule_3_violation":null,"rule_4_violation":null}\n'
        '```'
    )
    assert build_target_json(row, task="violations_only") == expected_json
    assert build_target_json(row, task=TASK).endswith(expected_json)


def test_the_reason_keeps_its_period_in_the_json_and_loses_it_in_the_block():
    """reward_reasoning and the LLM judge score against the exact GT string, so the
    JSON copy must be untouched; the block strips it only so '-> yes' reads cleanly."""
    vt = build_target_json(_row(), task=TASK)
    block, payload = vt.split(THINK_CLOSE)
    assert f"{REASON.rstrip('.')} -> yes (1)" in block
    assert f'"reason":"{REASON}"' in payload


def test_no_coordinates_appear_in_the_block():
    """Coordinates ARE the answer; emitting them twice doubles the exposure to the
    malformed-box-array failure mode for no derivation benefit."""
    vt = build_target_json(_row(), task=TASK)
    block = vt.split(THINK_CLOSE)[0]
    for token in ("500", "600", "[[", "bounding_box"):
        assert token not in block


def test_payload_still_validates_against_the_shared_schema():
    vt = build_target_json(_row(), task=TASK)
    payload = vt.split("```json\n")[1].split("\n```")[0]
    get_output_schema(TASK)(**json.loads(payload))


def test_schema_and_gt_builder_are_the_SAME_objects_as_violations_only():
    """Not copies. Identical ground truth means identical reward and metric behaviour,
    which is what makes the target text provably the only variable."""
    assert get_output_schema(TASK) is get_output_schema("violations_only")
    assert _GT_BUILDERS[TASK] is _GT_BUILDERS["violations_only"]


def test_ground_truth_is_identical():
    row = _row()
    assert build_gt_dict(row, task=TASK) == build_gt_dict(row, task="violations_only")


def test_registered_in_both_dispatch_tables():
    assert TASK in _TARGET_BUILDERS
    assert TASK in _GT_BUILDERS


# ---------------------------------------------------------------------------
# The builder refuses bad data
# ---------------------------------------------------------------------------

def test_refuses_a_block_that_contradicts_its_own_label():
    row = _row()
    row["rule_4_violation"] = {"bounding_box": [[0, 0, 1, 1]], "reason": "In the swing radius."}
    with pytest.raises(ValueError, match="rule_4"):
        build_target_json(row, task=TASK)


def test_refuses_a_missing_thinking_column():
    """violations_only's own dataset has no such column, so pointing this task at the
    wrong directory must fail loudly at dataset-build time, not train on nothing."""
    row = _row()
    row.pop(THINK_FIELD)
    with pytest.raises(ValueError, match="missing"):
        build_target_json(row, task=TASK)


def test_the_error_names_the_image_and_points_at_the_validator():
    row = _row()
    row[THINK_FIELD] = "garbage"
    with pytest.raises(ValueError) as exc:
        build_target_json(row, task=TASK)
    assert "0001234" in str(exc.value)
    assert "validate_think_dataset" in str(exc.value)


def test_violations_only_is_unaffected_by_a_junk_thinking_column():
    """The two arms share a dataset directory, so violations_only (the v4 arm) must
    ignore the column entirely rather than validate it."""
    row = _row()
    row[THINK_FIELD] = "garbage that would fail every check"
    build_target_json(row, task="violations_only")     # must not raise


# ---------------------------------------------------------------------------
# Conversation assembly
# ---------------------------------------------------------------------------

def test_conversation_carries_the_block_in_the_assistant_turn():
    conv = raw_sample_to_conversation_for_task(_row(), pil_image="IMG", task=TASK)
    assistant = conv["messages"][-1]
    assert assistant["role"] == "assistant"
    text = assistant["content"][0]["text"]
    assert text.startswith(THINK_OPEN)
    assert text.endswith("```")


def test_the_prompt_asks_for_a_block():
    from data.prompt_templates import get_prompt_for_task
    prompt = get_prompt_for_task(TASK)
    assert THINK_OPEN in prompt and "```json" in prompt
    # violations_only must NOT have grown one.
    assert THINK_OPEN not in get_prompt_for_task("violations_only")
