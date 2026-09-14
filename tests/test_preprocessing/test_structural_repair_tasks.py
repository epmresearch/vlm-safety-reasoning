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
    normalize_violation_value,
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


# ---------------------------------------------------------------------------
# A LIST of violations is merged, not dropped
#
# normalize_violation_value used to return None for any list value, i.e. "this
# rule was NOT violated". That does not drop a repair, it INVERTS the model's
# answer. On the real vo-baseline-2b run it fired on 1260 of 3004 records (42%)
# and destroyed 182 true positives, dragging recall from 0.887 to 0.469 -- which
# then read as a model difference against the 4b/8b baselines rather than as a
# repair artifact. It fired on ~0% of the SFT/GRPO runs, so it also made one
# column of the comparison table silently non-comparable with the rest.
#
# Two list shapes occur in real output and both must survive.
# ---------------------------------------------------------------------------

def test_violation_list_of_objects_is_merged_not_dropped():
    """vo-baseline-2b's shape: one violation object per instance."""
    out = normalize_violation_value(
        [
            {"bounding_box": [345, 27, 380, 234], "reason": "A worker on foot is missing a hard hat."},
            {"bounding_box": [555, 50, 590, 270], "reason": "A worker on foot is missing a hard hat."},
        ],
        rule_key="rule_1",
    )
    assert out is not None, "a list of violation objects must not become 'no violation'"
    assert len(out["bounding_box"]) == 2
    # Identical reasons are de-duplicated: N instances of the SAME finding is the
    # dominant real shape, and repeating it N times only depresses the
    # length-calibrated reasoning score without adding information.
    assert out["reason"] == "A worker on foot is missing a hard hat."


def test_violation_list_joins_genuinely_different_reasons():
    out = normalize_violation_value(
        [
            {"bounding_box": [10, 10, 20, 20], "reason": "The worker on the left has no hard hat."},
            {"bounding_box": [30, 30, 40, 40], "reason": "The worker on the right has no vest."},
        ],
        rule_key="rule_1",
    )
    assert out["reason"] == (
        "The worker on the left has no hard hat. The worker on the right has no vest."
    )
    assert len(out["bounding_box"]) == 2


def test_violation_list_of_bare_boxes_is_merged():
    """vo-baseline-8b's shape: the wrapping object omitted, boxes given alone.

    No reason is invented -- an empty reason beside a real box is the same shape
    this module already manufactures for a bare `true`, and it still counts as
    substantive to the rewards because it IS localized.
    """
    out = normalize_violation_value([[0, 0, 1000, 999]], rule_key="rule_3")
    assert out == {"bounding_box": [[0.0, 0.0, 1000.0, 999.0]], "reason": ""}

    out = normalize_violation_value([[825, 352, 862, 408], [852, 588, 878, 640]], rule_key="rule_2")
    assert len(out["bounding_box"]) == 2


def test_violation_flat_box_inside_a_list_is_reshaped():
    out = normalize_violation_value([0, 0, 1000, 999], rule_key="rule_3")
    assert out == {"bounding_box": [[0.0, 0.0, 1000.0, 999.0]], "reason": ""}


def test_violation_list_mixing_an_object_and_a_bare_box_keeps_both():
    out = normalize_violation_value(
        [{"bounding_box": [10, 10, 20, 20], "reason": "A finding."}, [50, 50, 60, 60]],
        rule_key="rule_1",
    )
    assert len(out["bounding_box"]) == 2
    assert out["reason"] == "A finding."


@pytest.mark.parametrize("payload", [[], [{}, {}], ["garbage", "more garbage"]])
def test_violation_list_with_nothing_usable_is_still_null(payload):
    """The only case that may still collapse to 'no violation': nothing to keep."""
    assert normalize_violation_value(payload, rule_key="rule_1") is None


def test_violation_list_merge_is_logged_as_a_merge_not_a_drop():
    tracker = ChangeTracker()
    normalize_violation_value(
        [{"bounding_box": [1, 1, 2, 2], "reason": "x"}], rule_key="rule_1", tracker=tracker
    )
    types = [c["type"] for c in tracker.changes]
    assert "violation_list_merged" in types
    assert "violation_list_dropped" not in types


def test_vo_end_to_end_list_shape_survives_repair():
    """Through the real repair entry point, not just the normalizer."""
    raw = _fenced({
        "rule_1_violation": [
            {"bounding_box": [345, 27, 380, 234], "reason": "No hard hat."},
            {"bounding_box": [555, 50, 590, 270], "reason": "No hard hat."},
        ],
        "rule_2_violation": None,
        "rule_3_violation": [[0, 0, 1000, 999]],
        "rule_4_violation": None,
    })
    res = repair_and_validate(raw, task="violations_only")
    assert res["status"] == "fixed_valid"
    fixed = res["fixed_parsed"]
    assert fixed["rule_1_violation"] is not None, "the list must not become 'no violation'"
    assert len(fixed["rule_1_violation"]["bounding_box"]) == 2
    assert fixed["rule_1_violation"]["reason"] == "No hard hat."
    assert fixed["rule_3_violation"] == {"bounding_box": [[0.0, 0.0, 1000.0, 999.0]], "reason": ""}
    assert fixed["rule_2_violation"] is None and fixed["rule_4_violation"] is None
