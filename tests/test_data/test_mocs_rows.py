"""
Tests for data/mocs_rows.py -- the review file -> ConstructionSite-row conversion.

Mirrors tests/test_data/test_preprocessor_vt.py in shape. Every one of these pins a
decision documented in the module's docstring, because each of them silently changes
what the model is taught:

  * boxes must stay in [0,1]                -- rescaling zeroes every IoU
  * `provenance` must not reach the row     -- it changes the Arrow struct type
  * an empty assertion is nulled, not kept  -- it is a miss AND a false alarm
  * a blank caption drops the row           -- there is no honest repair
  * rule_4 multi-box -> one union box       -- REVIEW_RUBRIC.md's GT convention
  * the schema/scale header is CHECKED      -- the one failure nothing downstream sees
"""
import json

import pytest

from core.constants import GROUNDING_CLASSES, METADATA_FIELDS, RULES
from core.think_format import build_think_body, think_row_problems, violations_from_row
from data.mocs_rows import (
    PROVENANCE_MOCS,
    REVIEW_BOX_SCALE,
    REVIEW_SCHEMA,
    MocsConversionReport,
    build_mocs_rows,
    load_review_file,
    review_row_to_dataset_row,
    union_box,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _violation(boxes, reason, **prov):
    base = {"proposed_by_model": True, "reason_edited": False,
            "model_boxes_kept": len(boxes), "model_boxes_deleted": 0, "reviewer_boxes": 0}
    base.update(prov)
    return {"bounding_box": [list(b) for b in boxes], "reason": reason, "provenance": base}


def _entry(image_id="mocs_0000001", caption="A worker stands beside an excavator.", **rules):
    row = {
        "new_image_id": image_id,
        "mocs_image_id": 1,
        "file_name": image_id.replace("mocs_", "") + ".jpg",
        "source": "val",
        "run": "mocs_annotation",
        "image_path_original": f"/home/u/datasets/filtered/instances_val/{image_id[5:]}.jpg",
        "review_image": f"{image_id}.jpg",
        "width": 1200,
        "height": 900,
        "image_caption": caption,
        "caption_usable": bool(caption),
        "caption_ok": "y",
        "caption_edited": False,
        "caption_model": caption,
        "mocs_categories": ["Worker", "Excavator"],
        "flagged_rules": [],
        "hard": False,
        "notes": "",
        "reviewed_at": "2026-10-05T00:00:00.000Z",
    }
    for rule in RULES:
        row[f"{rule}_violation"] = rules.get(rule)
    row["flagged_rules"] = [r for r in RULES if row[f"{r}_violation"] is not None]
    return row


def _review(entries, schema=REVIEW_SCHEMA, box_scale=REVIEW_BOX_SCALE):
    return {"schema": schema, "box_scale": box_scale, "reviewer": "T",
            "saved_at": "2026-10-07T00:00:00.000Z", "dataset_rows": entries}


# ---------------------------------------------------------------------------
# the header check
# ---------------------------------------------------------------------------

def test_load_review_file_accepts_the_expected_header(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_review([_entry()])), encoding="utf-8")
    assert load_review_file(p)["schema"] == REVIEW_SCHEMA


def test_load_review_file_rejects_a_different_box_scale(tmp_path):
    """The one failure mode nothing downstream can see: boxes in the wrong scale."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_review([_entry()], box_scale="xyxy_0_1000")), encoding="utf-8")
    with pytest.raises(ValueError, match="box_scale"):
        load_review_file(p)


def test_load_review_file_rejects_a_different_schema(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_review([_entry()], schema="mocs_review/3")), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_review_file(p)


# ---------------------------------------------------------------------------
# the row shape
# ---------------------------------------------------------------------------

def test_row_carries_every_constructionsite_column_and_nothing_extra():
    entry = _entry(rule_1=_violation([[0.1, 0.1, 0.4, 0.8]], "No hard hat."))
    row, ref = review_row_to_dataset_row(entry, MocsConversionReport())
    expected = ({"image_id", "image_caption", "resolution", "provenance"}
                | {f"{r}_violation" for r in RULES}
                | set(GROUNDING_CLASSES) | set(METADATA_FIELDS))
    assert set(row) == expected
    assert row["provenance"] == PROVENANCE_MOCS
    assert ref.image_id == "mocs_0000001"


def test_violation_struct_is_exactly_bounding_box_and_reason():
    """An extra key changes the Arrow struct type and breaks concatenate_datasets."""
    entry = _entry(rule_2=_violation([[0.1, 0.1, 0.4, 0.8]], "No harness."))
    row, _ = review_row_to_dataset_row(entry, MocsConversionReport())
    assert set(row["rule_2_violation"]) == {"bounding_box", "reason"}
    assert "provenance" not in row["rule_2_violation"]


def test_boxes_are_carried_through_unscaled():
    box = [0.1234, 0.2345, 0.3456, 0.4567]
    entry = _entry(rule_3=_violation([box], "The trench edge is unguarded."))
    row, _ = review_row_to_dataset_row(entry, MocsConversionReport())
    assert row["rule_3_violation"]["bounding_box"] == [box]
    assert all(0.0 <= c <= 1.0 for c in row["rule_3_violation"]["bounding_box"][0])


def test_object_and_metadata_columns_are_empty_not_invented():
    """MOCS annotates none of these. Empty means 'unannotated', and provenance says so."""
    entry = _entry()
    row, _ = review_row_to_dataset_row(entry, MocsConversionReport())
    assert all(row[c] == [] for c in GROUNDING_CLASSES)
    assert all(row[m] == "" for m in METADATA_FIELDS)


def test_resolution_is_pixel_area_not_a_pair():
    entry = _entry()
    row, _ = review_row_to_dataset_row(entry, MocsConversionReport())
    assert row["resolution"] == 1200 * 900


# ---------------------------------------------------------------------------
# the three repairs
# ---------------------------------------------------------------------------

def test_empty_assertion_drops_the_whole_row():
    """mocs_0023379's real shape: rule_1 asserted with nothing, rule_2 genuinely
    violated. The verified rule_2 goes with it -- that cost is the decision, not an
    oversight. One rule: a row that cannot carry a clean block is dropped."""
    entry = _entry(
        rule_1={"bounding_box": [], "reason": "",
                "provenance": {"proposed_by_model": False, "reason_edited": False,
                               "model_boxes_kept": 0, "model_boxes_deleted": 0,
                               "reviewer_boxes": 0}},
        rule_2=_violation([[0.2, 0.1, 0.5, 0.6], [0.6, 0.1, 0.8, 0.6]],
                          "The workers on top of the structure are not wearing safety harnesses."),
    )
    report = MocsConversionReport()
    assert review_row_to_dataset_row(entry, report) is None
    assert report.dropped == [{"image_id": "mocs_0000001",
                               "cause": "asserted with no reason",
                               "reason": "rule_1: asserted with no reason (and no box)",
                               "rule": "rule_1"}]


def test_a_dropped_row_leaves_no_findings_behind():
    """rule_1 is converted (and its boxes recorded) before rule_4 takes the row down.
    The manifest must not then describe a row that is not in the dataset."""
    entry = _entry(
        rule_1=_violation([[0.1, 0.1, 0.4, 0.8], [9.0, 9.0, 9.5, 9.5]], "No hard hat."),
        rule_4={"bounding_box": [[0.1, 0.2, 0.3, 0.6], [0.5, 0.1, 0.7, 0.8]],
                "reason": "", "provenance": {}},
    )
    report = MocsConversionReport()
    assert review_row_to_dataset_row(entry, report) is None
    assert report.rule4_boxes_unioned == []
    assert report.boxes_rejected == []
    assert report.reason_only_violations == []


def test_blank_caption_drops_the_row():
    report = MocsConversionReport()
    entry = _entry(caption="   ", rule_4=_violation([[0.1, 0.1, 0.9, 0.9]], "Too close."))
    assert review_row_to_dataset_row(entry, report) is None
    assert report.dropped[0]["cause"] == "blank_caption"
    assert "image_caption is blank" in report.dropped[0]["reason"]


def test_rule4_multiple_boxes_are_unioned_by_default():
    entry = _entry(rule_4=_violation([[0.1, 0.2, 0.3, 0.6], [0.5, 0.1, 0.7, 0.8]],
                                     "Both workers are within the operation radius."))
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report)
    assert row["rule_4_violation"]["bounding_box"] == [[0.1, 0.1, 0.7, 0.8]]
    assert len(report.rule4_boxes_unioned) == 1


def test_rule4_box_policy_keep_leaves_the_reviewer_alone():
    boxes = [[0.1, 0.2, 0.3, 0.6], [0.5, 0.1, 0.7, 0.8]]
    entry = _entry(rule_4=_violation(boxes, "Both workers are within the operation radius."))
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report, rule4_box_policy="keep")
    assert row["rule_4_violation"]["bounding_box"] == boxes
    assert report.rule4_boxes_unioned == []


def test_single_box_rule4_is_untouched_by_the_union_policy():
    entry = _entry(rule_4=_violation([[0.1, 0.1, 0.9, 0.9]], "One worker, one box."))
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report)
    assert row["rule_4_violation"]["bounding_box"] == [[0.1, 0.1, 0.9, 0.9]]
    assert report.rule4_boxes_unioned == []


def test_other_rules_keep_their_multiple_boxes():
    """rule_1/2's documented convention IS one box per person -- do not union those."""
    boxes = [[0.1, 0.2, 0.3, 0.6], [0.5, 0.1, 0.7, 0.8]]
    entry = _entry(rule_1=_violation(boxes, "Two workers without hard hats."))
    row, _ = review_row_to_dataset_row(entry, MocsConversionReport())
    assert row["rule_1_violation"]["bounding_box"] == boxes


def test_boxed_violation_with_no_reason_drops_the_row_too():
    """The third shape: boxes but no sentence. Same one rule -- build_think_body
    cannot write a line for it either, and nulling it would discard verified geometry
    while leaving the image in the dataset asserting the opposite."""
    entry = _entry(
        rule_1={"bounding_box": [[0.1, 0.1, 0.4, 0.8]], "reason": "",
                "provenance": {"proposed_by_model": True, "reason_edited": False,
                               "model_boxes_kept": 1, "model_boxes_deleted": 0,
                               "reviewer_boxes": 0}},
        rule_2=_violation([[0.2, 0.1, 0.5, 0.6]], "The worker has no harness."),
    )
    report = MocsConversionReport()
    assert review_row_to_dataset_row(entry, report) is None
    assert report.dropped[0]["cause"] == "asserted with no reason"
    assert "1 box(es) discarded" in report.dropped[0]["reason"]


def test_a_non_dict_violation_value_drops_the_row_rather_than_inverting_it():
    """Reading a bare `true` as None would FLIP the label to 'not violated' -- the
    same failure class as the structural_repair list-drop bug that cost one run 182
    true positives. Dropping the row cannot invert anything."""
    entry = _entry(rule_1=True)
    report = MocsConversionReport()
    assert review_row_to_dataset_row(entry, report) is None
    assert report.dropped[0]["cause"] == "violation value is not an object"
    assert report.dropped[0]["rule"] == "rule_1"


def test_rows_in_minus_rows_out_always_equals_the_dropped_row_count():
    rows, _, report = build_mocs_rows(_review([
        _entry("mocs_1"),
        _entry("mocs_2", caption=""),
        _entry("mocs_3", rule_1=True),
        _entry("mocs_4", rule_2={"bounding_box": [], "reason": "", "provenance": {}}),
    ]))
    assert len(rows) == 1
    assert report.rows_in - report.rows_out == len(report.dropped) == 3
    assert report.dropped_by_cause == {
        "blank_caption": 1,
        "violation value is not an object": 1,
        "asserted with no reason": 1,
    }


def test_malformed_box_elements_are_counted_against_what_was_submitted():
    """normalize_boxes silently discards junk elements; comparing downstream of it
    hides exactly what boxes_rejected exists to surface."""
    entry = _entry(rule_1={"bounding_box": [[0.1, 0.1, 0.4, 0.8], "junk", {"nope": 1}],
                           "reason": "No hard hat.", "provenance": {}})
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report)
    assert len(row["rule_1_violation"]["bounding_box"]) == 1
    assert report.boxes_rejected == [
        {"image_id": "mocs_0000001", "rule": "rule_1", "kept": 1, "submitted": 3}]


def test_reason_only_violation_survives_and_is_counted():
    entry = _entry(rule_4=_violation([], "The worker in the trench is within the operation radius."))
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report)
    assert row["rule_4_violation"]["bounding_box"] == []
    assert row["rule_4_violation"]["reason"]
    assert len(report.reason_only_violations) == 1


def test_out_of_range_boxes_are_rejected_and_recorded():
    """clean_boxes enforces [0,1]; doing it here makes the rejection visible."""
    entry = _entry(rule_1=_violation([[0.1, 0.1, 0.4, 0.8], [10.0, 10.0, 20.0, 20.0]],
                                     "Somebody has no hard hat."))
    report = MocsConversionReport()
    row, _ = review_row_to_dataset_row(entry, report)
    assert len(row["rule_1_violation"]["bounding_box"]) == 1
    assert report.boxes_rejected == [
        {"image_id": "mocs_0000001", "rule": "rule_1", "kept": 1, "submitted": 2}]


def test_union_box_is_the_enclosing_rectangle():
    assert union_box([[0.2, 0.3, 0.4, 0.5], [0.1, 0.4, 0.9, 0.6]]) == [0.1, 0.3, 0.9, 0.6]


# ---------------------------------------------------------------------------
# the corpus-level pass
# ---------------------------------------------------------------------------

def test_build_mocs_rows_reports_rules_boxes_sources_and_safe_rows():
    rows, refs, report = build_mocs_rows(_review([
        _entry("mocs_0000001", rule_1=_violation([[0.1, 0.1, 0.4, 0.8]], "No hard hat.")),
        _entry("mocs_0000002"),                                     # confirmed safe
        _entry("mocs_0000003", rule_2=_violation([[0.1, 0.1, 0.4, 0.8],
                                                  [0.5, 0.1, 0.9, 0.8]], "No harness.")),
    ]))
    assert len(rows) == len(refs) == 3
    assert report.by_rule == {"rule_1": 1, "rule_2": 1, "rule_3": 0, "rule_4": 0}
    assert report.boxes_by_rule["rule_2"] == 2
    assert report.by_source == {"val": 3}
    assert report.zero_violation_rows == ["mocs_0000002"]


def test_build_mocs_rows_drops_a_duplicate_image_id():
    rows, _, report = build_mocs_rows(_review([_entry("mocs_1"), _entry("mocs_1")]))
    assert len(rows) == 1
    assert report.dropped_by_cause == {"duplicate_image_id": 1}


def test_build_mocs_rows_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="rule4_box_policy"):
        build_mocs_rows(_review([_entry()]), rule4_box_policy="shrink")


def test_provenance_is_tallied_before_being_stripped():
    rows, _, report = build_mocs_rows(_review([
        _entry("mocs_1", rule_3=_violation([[0.1, 0.1, 0.9, 0.9]], "Unguarded trench.",
                                           reason_edited=True, reviewer_boxes=1)),
    ]))
    assert "provenance" not in rows[0]["rule_3_violation"]
    assert report.provenance_tallies["rule_3"]["reason_edited"] == 1
    assert report.provenance_tallies["rule_3"]["reviewer_boxes"] == 1


# ---------------------------------------------------------------------------
# the property that actually matters: every converted row can carry a think block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rules", [
    {},
    {"rule_1": _violation([[0.1, 0.1, 0.4, 0.8]], "No hard hat.")},
    {"rule_4": _violation([], "Within the operation radius.")},            # reason-only
    {"rule_2": _violation([[0.1, 0.1, 0.4, 0.8]], "No harness"),           # no trailing period
     "rule_3": _violation([[0.0, 0.0, 1.0, 1.0]], "The trench edge is unguarded.")},
])
def test_every_converted_row_builds_a_valid_think_block(rules):
    row, _ = review_row_to_dataset_row(_entry(**rules), MocsConversionReport())
    body = build_think_body(row["image_caption"], violations_from_row(row))
    assert think_row_problems({**row, "thinking": body}) == []
    # Five lines, caption first, rule_1 -> rule_4, no coordinates.
    lines = body.split("\n")
    assert len(lines) == 5
    assert lines[0] == row["image_caption"]
    assert [l.split(":")[0] for l in lines[1:]] == RULES
