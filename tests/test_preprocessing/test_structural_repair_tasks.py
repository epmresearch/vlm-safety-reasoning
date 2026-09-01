"""Task-awareness of the structural repair pipeline.

Two classes of bug are pinned here:

  1. Repairs a task NEEDS must actually run. object_only box repair used to be
     gated behind `task == "unified"`, so a recoverable object_only output landed
     in still_broken.
  2. Repairs a task must NOT get. The violation group used to run unconditionally
     and *assign* rather than test, injecting four phantom `rule_N_violation: null`
     keys into every repaired record of every task.

Plus the plain-text path for caption_only, and byte-level non-regression for the
two pre-existing pipelines.
"""
import json

import pytest

from preprocessing.structural_repair import (
    ChangeTracker,
    _canonical_keys_for_task,
    fix_prediction_structure,
    repair_and_validate,
)

VIOLATION_KEYS = [f"rule_{i}_violation" for i in range(1, 5)]
OBJECT_KEYS = ["excavator", "rebar", "worker_with_white_hard_hat"]


def _fenced(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


def _change_types(result):
    return [c["type"] for c in result["changes"]]


# ---------------------------------------------------------------------------
# Canonical-key ownership is derived from the schema
# ---------------------------------------------------------------------------

def test_canonical_keys_match_each_schema():
    assert _canonical_keys_for_task("object_only") == set(OBJECT_KEYS)
    assert _canonical_keys_for_task("caption_only") == {"caption"}
    assert _canonical_keys_for_task("violations_only") == set(VIOLATION_KEYS)
    assert _canonical_keys_for_task("unified") == set(
        ["caption"] + OBJECT_KEYS + VIOLATION_KEYS
    )


# ---------------------------------------------------------------------------
# object_only
# ---------------------------------------------------------------------------

def test_oo_clean_output_is_valid_raw():
    raw = _fenced({"excavator": [[100, 100, 200, 200]], "rebar": [],
                   "worker_with_white_hard_hat": []})
    res = repair_and_validate(raw, task="object_only")
    assert res["status"] == "valid_raw"
    assert res["changes"] == []


@pytest.mark.parametrize("payload,label", [
    ({"excavator": [100, 100, 200, 200], "rebar": [], "worker_with_white_hard_hat": []},
     "flat box list"),
    ({"excavator": [[[100, 100], [200, 200]]], "rebar": [], "worker_with_white_hard_hat": []},
     "corner pairs"),
])
def test_oo_repairable_boxes_are_recovered_not_marked_broken(payload, label):
    """This is the regression the `task == "unified"` gate caused: object box
    normalization never ran for object_only, so these landed in still_broken."""
    res = repair_and_validate(_fenced(payload), task="object_only")
    assert res["status"] in ("valid_raw", "fixed_valid"), f"{label}: {res['error']}"
    assert res["fixed_parsed"]["excavator"] == [[100.0, 100.0, 200.0, 200.0]]


def test_oo_key_alias_is_renamed_and_the_detection_survives():
    """A real detection emitted under the alias `excavators` must not be scored as
    "no objects detected". Requires ObjectOnlyOutput's keys to be mandatory so the
    strict gate rejects the aliased form and the repair pass gets a chance."""
    res = repair_and_validate(_fenced({"excavators": [[100, 100, 200, 200]]}),
                              task="object_only")
    assert res["status"] == "fixed_valid"
    assert "key_renamed" in _change_types(res)
    assert res["fixed_parsed"]["excavator"] == [[100.0, 100.0, 200.0, 200.0]]


def test_oo_repaired_output_has_no_phantom_violation_keys():
    res = repair_and_validate(_fenced({"excavator": [100, 100, 200, 200]}),
                              task="object_only")
    assert res["status"] == "fixed_valid"
    assert set(res["fixed_parsed"]) == set(OBJECT_KEYS)
    for k in VIOLATION_KEYS:
        assert k not in res["fixed_parsed"]
    assert "caption" not in res["fixed_parsed"]


def test_oo_caption_alias_is_not_renamed_into_the_output():
    """`description` is a caption alias, but object_only's schema does not own
    `caption`, so the key must be left verbatim as a harmless extra."""
    fixed = fix_prediction_structure(
        {"description": "a site", "excavator": [], "rebar": [],
         "worker_with_white_hard_hat": []},
        tracker=ChangeTracker(), task="object_only",
    )
    assert "caption" not in fixed
    assert fixed["description"] == "a site"


def test_oo_unparseable_stays_invalid_json():
    assert repair_and_validate("no json here", task="object_only")["status"] == "invalid_json"


# ---------------------------------------------------------------------------
# caption_only (plain text)
# ---------------------------------------------------------------------------

PROSE = "Two workers beside an excavator on a muddy site."


def test_co_clean_prose_is_valid_raw():
    res = repair_and_validate(PROSE, task="caption_only")
    assert res["status"] == "valid_raw"
    assert res["fixed_parsed"] == {"caption": PROSE}
    assert res["changes"] == []


def test_co_fenced_prose_is_recovered():
    res = repair_and_validate("```\n%s\n```" % PROSE, task="caption_only")
    assert res["status"] == "fixed_valid"
    assert res["fixed_parsed"] == {"caption": PROSE}
    assert "caption_fence_stripped" in _change_types(res)


def test_co_json_wrapped_caption_is_recovered():
    res = repair_and_validate('{"caption": "%s"}' % PROSE, task="caption_only")
    assert res["status"] == "fixed_valid"
    assert res["fixed_parsed"] == {"caption": PROSE}
    assert "caption_json_unwrapped" in _change_types(res)


def test_co_caption_list_is_joined():
    res = repair_and_validate('{"caption": ["One.", "Two."]}', task="caption_only")
    assert res["status"] == "fixed_valid"
    assert res["fixed_parsed"] == {"caption": "One. Two."}


def test_co_whitespace_padding_is_not_a_repair():
    """Surrounding whitespace is stripped but does not count as a fix — the same
    way the JSON path treats whitespace around a fenced object."""
    res = repair_and_validate("   %s   " % PROSE, task="caption_only")
    assert res["status"] == "valid_raw"
    assert res["fixed_parsed"] == {"caption": PROSE}
    assert res["changes"] == []


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_co_blank_output_is_invalid_json(empty):
    res = repair_and_validate(empty, task="caption_only")
    assert res["status"] == "invalid_json"
    assert res["fixed_parsed"] is None


def test_co_json_without_a_caption_is_invalid_schema():
    res = repair_and_validate('{"excavator": []}', task="caption_only")
    assert res["status"] == "invalid_schema"


def test_co_repaired_output_carries_only_a_caption():
    res = repair_and_validate('{"caption": "%s"}' % PROSE, task="caption_only")
    assert set(res["fixed_parsed"]) == {"caption"}


# ---------------------------------------------------------------------------
# Non-regression for the two pre-existing pipelines
# ---------------------------------------------------------------------------

def test_unified_still_repairs_caption_objects_and_violations():
    raw = _fenced({
        "caption": ["First.", "Second."],
        "excavator": [100, 100, 200, 200],
        "rule_1_violation": True,
    })
    res = repair_and_validate(raw, task="unified")
    assert res["status"] == "fixed_valid"
    fixed = res["fixed_parsed"]
    assert fixed["caption"] == "First. Second."
    assert fixed["excavator"] == [[100.0, 100.0, 200.0, 200.0]]
    # A bare `true` is an assertion of violation, normalized to a contentless
    # violation object — NOT to null, which would invert the model's answer.
    assert fixed["rule_1_violation"] == {"reason": "", "bounding_box": []}
    types = _change_types(res)
    assert "caption_list_joined" in types
    assert "violation_bool_converted" in types


def test_vo_still_gets_violation_normalization_and_no_object_keys():
    raw = _fenced({"rule_1_violation": True, "rule_2_violation": {},
                   "rule_3_violation": "worker on an unprotected edge",
                   "rule_4_violation": None})
    res = repair_and_validate(raw, task="violations_only")
    assert res["status"] == "fixed_valid"
    fixed = res["fixed_parsed"]
    assert fixed["rule_1_violation"] == {"reason": "", "bounding_box": []}
    # A bare {} carries no keys and no assertion -> null.
    assert fixed["rule_2_violation"] is None
    assert fixed["rule_3_violation"] == {
        "reason": "worker on an unprotected edge", "bounding_box": []
    }
    for k in OBJECT_KEYS:
        assert k not in fixed
    assert "caption" not in fixed


def test_vo_object_and_caption_aliases_are_still_suppressed():
    fixed = fix_prediction_structure(
        {"description": "a site", "excavators": [[1, 2, 3, 4]], "rule_1_violation": None},
        tracker=ChangeTracker(), task="violations_only",
    )
    assert "caption" not in fixed
    assert "excavator" not in fixed
    assert fixed["description"] == "a site"


# ---------------------------------------------------------------------------
# Batch driver writes the repaired record in the task's wire format
# ---------------------------------------------------------------------------

def test_process_jsonl_writes_prose_for_co_and_json_for_oo(tmp_path):
    from preprocessing.structural_repair import process_jsonl

    cases = [
        ("caption_only", '{"caption": "%s"}' % PROSE, PROSE),
        ("object_only", _fenced({"excavator": [100, 100, 200, 200]}), None),
    ]
    for task, raw, expected in cases:
        d = tmp_path / task
        d.mkdir()
        inp = d / "predictions.jsonl"
        inp.write_text(json.dumps({"image_id": "a", "raw_output": raw}) + "\n",
                       encoding="utf-8")
        out = d / "repaired.jsonl"
        process_jsonl(
            input_path=str(inp), output_path=str(out),
            report_path=str(d / "report.json"), broken_path=str(d / "broken.json"),
            manifest_path=str(d / "manifest.json"), task=task,
        )
        rec = json.loads(out.read_text(encoding="utf-8").strip())
        assert rec["repair_status"] == "fixed_valid"
        assert rec["original_raw_output"] == raw
        if expected is not None:
            # Plain-text task: written back as bare prose, so evaluation re-parses
            # it with the same contract the model was trained on.
            assert rec["raw_output"] == expected
            assert "```" not in rec["raw_output"]
        else:
            payload = json.loads(rec["raw_output"])
            assert set(payload) == set(OBJECT_KEYS)
