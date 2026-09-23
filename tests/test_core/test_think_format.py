"""Tests for core/think_format.py -- the <think>-block wire format.

The block is unrewarded, unparsed and unrepaired: nothing downstream of the SFT target
builder can tell a good block from a contradictory one. These tests are therefore the
only thing standing between a mis-baked `thinking` column and a model trained to invert
evidence, so they cover the negative cases at least as hard as the positive ones.
"""
import pytest

from core.constants import RULES
from core.think_format import (
    THINK_CLOSE,
    THINK_FIELD,
    THINK_OPEN,
    build_think_body,
    canonical_reason,
    extract_think_block,
    is_violation_asserted,
    parse_think_body,
    render_think_block,
    think_row_problems,
    verdicts_from_violations,
    violations_from_row,
)

CAPTION = "A worker is sitting on a scaffold attached to the formwork."
REASON_1 = "The worker with a green jacket sitting on a scaffolding is not wearing a hard hat."


def _row(**overrides):
    """A canonical row: rule_1 violated with one box, the rest safe."""
    row = {
        "image_id": "0000704",
        "image_caption": CAPTION,
        "rule_1_violation": {"bounding_box": [[0.33, 0.11, 0.56, 0.42]], "reason": REASON_1},
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
    }
    row.update(overrides)
    if THINK_FIELD not in row:
        row[THINK_FIELD] = build_think_body(row["image_caption"], violations_from_row(row))
    return row


# ---------------------------------------------------------------------------
# The canonical form
# ---------------------------------------------------------------------------

def test_build_think_body_shape():
    body = build_think_body(CAPTION, violations_from_row(_row()))
    lines = body.split("\n")
    assert len(lines) == 5, "caption + one line per rule"
    assert lines[0] == CAPTION
    # The trailing period is stripped INSIDE the block only, so "-> yes" reads cleanly.
    assert lines[1] == f"rule_1: {REASON_1.rstrip('.')} -> yes (1)"
    assert lines[2:] == ["rule_2: no", "rule_3: no", "rule_4: no"]


def test_all_four_rule_lines_present_on_a_safe_image():
    """Omitting them would leak the answer through the block's mere presence."""
    safe = {f"{r}_violation": None for r in RULES}
    safe["image_caption"] = CAPTION
    body = build_think_body(CAPTION, violations_from_row(safe))
    assert body.split("\n")[1:] == [f"{r}: no" for r in RULES]


def test_rule_order_is_never_sorted_by_violation_status():
    """Sorting violated rules first would leak the answer through position."""
    row = {"image_caption": CAPTION,
           "rule_1_violation": None,
           "rule_2_violation": None,
           "rule_3_violation": None,
           "rule_4_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": "In the swing radius."}}
    lines = build_think_body(CAPTION, violations_from_row(row)).split("\n")
    assert [l.split(":")[0] for l in lines[1:]] == RULES


def test_box_count_suffix_tracks_the_label():
    row = _row(rule_1_violation={"bounding_box": [[0, 0, 1, 1]] * 3, "reason": REASON_1},
               **{THINK_FIELD: None})
    body = build_think_body(CAPTION, violations_from_row(row))
    assert body.split("\n")[1].endswith("-> yes (3)")


def test_no_box_count_when_there_are_no_boxes():
    """MOCS reason-only rows: schemas.py preserves a violation with either a box OR a
    reason, so this shape is real and must not render as '-> yes (0)'."""
    row = {"image_caption": CAPTION,
           "rule_2_violation": {"bounding_box": [], "reason": "No harness on the deck."}}
    body = build_think_body(CAPTION, violations_from_row(row))
    assert "rule_2: No harness on the deck -> yes" in body
    assert "(0)" not in body


def test_build_refuses_a_reasonless_violation():
    """No reason means no line without inventing text, and such a row already scores 0
    on reward_reasoning. Rejected at build time so data prep fixes or drops it."""
    row = {"image_caption": CAPTION,
           "rule_1_violation": {"bounding_box": [[0, 0, 1, 1]], "reason": ""}}
    with pytest.raises(ValueError, match="no reason"):
        build_think_body(CAPTION, violations_from_row(row))


def test_build_refuses_a_blank_or_multiline_caption():
    with pytest.raises(ValueError, match="blank"):
        build_think_body("   ", violations_from_row({}))
    with pytest.raises(ValueError, match="one line"):
        build_think_body("two\nlines", violations_from_row({}))


def test_render_wraps_with_tags_and_a_trailing_newline():
    out = render_think_block("body")
    assert out == f"{THINK_OPEN}\nbody\n{THINK_CLOSE}\n"


def test_canonical_reason_strips_only_the_trailing_period():
    assert canonical_reason("  A worker.  ") == "A worker"
    assert canonical_reason("A worker") == "A worker"
    assert canonical_reason("Mr. Smith is unsafe.") == "Mr. Smith is unsafe"
    assert canonical_reason(None) == ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_round_trip():
    row = _row()
    parsed = parse_think_body(row[THINK_FIELD])
    assert parsed.wellformed, parsed.problems
    assert parsed.caption == CAPTION
    assert parsed.verdicts == {"rule_1": True, "rule_2": False, "rule_3": False, "rule_4": False}
    assert parsed.counts["rule_1"] == 1
    assert parsed.reasons["rule_1"] == REASON_1.rstrip(".")


@pytest.mark.parametrize("body,problem_substr", [
    ("prose with no verdicts at all", "no rule_1 line found"),
    (f"{CAPTION}\nrule_1: no\nrule_2: no\nrule_3: no", "rule_4: line missing"),
    (f"{CAPTION}\nrule_2: no\nrule_1: no\nrule_3: no\nrule_4: no", "rule lines out of order"),
    (f"   \nrule_1: no\nrule_2: no\nrule_3: no\nrule_4: no",
     "no description before the rule lines"),
    (f"rule_1: no\nrule_2: no\nrule_3: no\nrule_4: no",
     "no description before the rule lines"),
    (f"{CAPTION}\nrule_1: no\nrule_2: no\nrule_3: no\nrule_4: no\nrule_4: no",
     "extra content after rule_4"),
    (f"{CAPTION}\nrule_1: no\n\nrule_2: no\nrule_3: no\nrule_4: no",
     "rule lines out of order"),
    (f"{CAPTION}\nrule_1: maybe\nrule_2: no\nrule_3: no\nrule_4: no", "neither 'no' nor"),
    (f"{CAPTION}\nrule_1:  -> yes (1)\nrule_2: no\nrule_3: no\nrule_4: no", "empty reason"),
    ("", "empty or not a string"),
])
def test_parse_reports_problems_without_raising(body, problem_substr):
    parsed = parse_think_body(body)
    assert not parsed.wellformed
    assert any(problem_substr in p for p in parsed.problems), parsed.problems


def test_parse_never_raises_on_arbitrary_input():
    """It reads raw model output at inference; a crash there loses a 3,004-image run."""
    for junk in (None, 42, [], {}, "```json\n{}\n```", "\x00\x01", "rule_1" * 5000):
        parse_think_body(junk)


def test_parse_is_tolerant_of_cosmetic_variation():
    """Same parser reads model output, where casing and spacing will vary."""
    body = f"{CAPTION}\nrule_1: x ->  YES ( 2 )\nrule_2: NO\nrule_3: no\nrule_4: no"
    parsed = parse_think_body(body)
    assert parsed.verdicts["rule_1"] is True
    assert parsed.counts["rule_1"] == 2
    assert parsed.verdicts["rule_2"] is False


def test_a_multiline_description_is_accepted():
    """A newline in the description is HARMLESS and must not be reported.

    Nothing scores the description, the JSON still parses, and no reported metric
    moves. An earlier version pinned the block at exactly five lines and matched the
    rule lines at indices 1-4, so one cosmetic newline produced five cascading
    failures and made the block's own health readout unusable. The prompt no longer
    asks for a single line either -- the two have to agree.
    """
    body = ("A busy construction site with several workers\n"
            "near a large excavator.\n"
            "rule_1: no hard hat -> yes (1)\nrule_2: no\nrule_3: no\nrule_4: no")
    parsed = parse_think_body(body)
    assert parsed.wellformed, parsed.problems
    assert parsed.caption == (
        "A busy construction site with several workers near a large excavator.")
    assert parsed.verdicts == {"rule_1": True, "rule_2": False,
                               "rule_3": False, "rule_4": False}
    assert parsed.counts["rule_1"] == 1


def test_rule_order_is_still_enforced_after_relaxing_the_line_count():
    """Order stays load-bearing: sorting by violation status would leak the answer
    through position, so the rule lines must still be consecutive from rule_1."""
    body = f"{CAPTION}\nrule_1: no\nrule_3: no\nrule_2: no\nrule_4: no"
    assert any("out of order" in p for p in parse_think_body(body).problems)


def test_the_canonical_body_is_still_exactly_five_lines():
    """The PARSER is looser, but the dataset must stay deterministic."""
    body = build_think_body(CAPTION, violations_from_row(_row()))
    assert len(body.split("\n")) == 5


def test_crlf_is_normalized():
    """A column baked on Windows must not fail every line check."""
    row = _row()
    assert parse_think_body(row[THINK_FIELD].replace("\n", "\r\n")).wellformed


def test_a_reason_containing_the_arrow_is_not_mistaken_for_the_verdict():
    body = f"{CAPTION}\nrule_1: he moved -> yes then no -> yes (1)\nrule_2: no\nrule_3: no\nrule_4: no"
    parsed = parse_think_body(body)
    assert parsed.verdicts["rule_1"] is True
    assert parsed.counts["rule_1"] == 1
    assert parsed.reasons["rule_1"] == "he moved -> yes then no"


# ---------------------------------------------------------------------------
# extract_think_block -- reading raw completions
# ---------------------------------------------------------------------------

def test_extract_closed_block():
    body, closed = extract_think_block(f"{THINK_OPEN}\nabc\n{THINK_CLOSE}\n```json\n{{}}\n```")
    assert (body, closed) == ("abc", True)


def test_extract_unclosed_block_stops_at_the_fence():
    """An unclosed block is what a truncated completion looks like -- reported, not
    dropped, because 'present but never closed' is a distinct diagnosis."""
    body, closed = extract_think_block(f"{THINK_OPEN}\nabc\n```json\n{{}}\n```")
    assert body == "abc"
    assert closed is False


def test_extract_unclosed_block_with_nothing_after_it():
    body, closed = extract_think_block(f"{THINK_OPEN}\nabc")
    assert (body, closed) == ("abc", False)


def test_extract_returns_none_when_absent():
    assert extract_think_block("```json\n{}\n```") == (None, False)
    assert extract_think_block(None) == (None, False)
    assert extract_think_block(42) == (None, False)


def test_extract_honours_only_the_first_open_tag():
    body, _ = extract_think_block(f"{THINK_OPEN}\nfirst\n{THINK_CLOSE}{THINK_OPEN}\nsecond\n{THINK_CLOSE}")
    assert body == "first"


# ---------------------------------------------------------------------------
# is_violation_asserted must MIRROR rewards/reward_utils._is_violation_present
# ---------------------------------------------------------------------------

def test_presence_predicate_mirrors_the_reward_utils_one():
    """core/ cannot import rewards/reward_utils (it pulls in torch at module scope, and
    the preprocessor and the dataset validator both run on a CPU login node), so the
    predicate is duplicated. This is what stops the copy from drifting: if it fails,
    core/think_format.py is wrong, not this test.
    """
    from rewards.reward_utils import _is_violation_present

    shapes = [
        None, True, False, {}, {"reason": ""}, {"bounding_box": []},
        {"reason": "x", "bounding_box": [[0, 0, 1, 1]]},
        "", "   ", "false", "None", "null", "n/a", "NULL", "a real reason",
        0, 1, [], [1],
    ]
    for v in shapes:
        assert is_violation_asserted(v) == _is_violation_present(v), f"disagree on {v!r}"


def test_violations_from_row_reads_the_suffixed_keys():
    """Conflating rule_N with rule_N_violation reads every violation as absent, which
    would report every genuinely violated row as a block/label contradiction."""
    row = _row()
    assert violations_from_row(row)["rule_1"] is row["rule_1_violation"]
    assert verdicts_from_violations(violations_from_row(row))["rule_1"] is True
    # The bare-key form must NOT accidentally work.
    assert verdicts_from_violations(row)["rule_1"] is False


# ---------------------------------------------------------------------------
# think_row_problems -- the safeguard
# ---------------------------------------------------------------------------

def test_clean_row_has_no_problems():
    assert think_row_problems(_row()) == []


def test_the_catastrophic_case_a_verdict_contradicting_its_own_label():
    """The whole reason this validation exists: nothing downstream can see it."""
    row = _row()
    row["rule_3_violation"] = {"bounding_box": [[0, 0, 1, 1]], "reason": "Unguarded trench."}
    problems = think_row_problems(row)
    assert any("rule_3" in p and "label says violated" in p for p in problems), problems


def test_block_asserting_a_rule_the_label_calls_safe():
    row = _row()
    row[THINK_FIELD] = row[THINK_FIELD].replace("rule_2: no", "rule_2: something -> yes (1)")
    assert any("rule_2" in p and "label says not violated" in p
               for p in think_row_problems(row))


@pytest.mark.parametrize("mutate,substr", [
    (lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD].replace("(1)", "(4)")), "box count"),
    (lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD].replace("green", "red")), "reason does not match"),
    (lambda r: r.__setitem__(THINK_FIELD, "Wrong caption.\n" + "\n".join(r[THINK_FIELD].split("\n")[1:])), "caption line does not match"),
    (lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD] + ' {"a":1}'), "brace"),
    (lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD] + "\n```"), "code fence"),
    (lambda r: r.pop(THINK_FIELD), "is missing"),
    (lambda r: r.__setitem__(THINK_FIELD, "  "), "is blank"),
    (lambda r: r.__setitem__(THINK_FIELD, 7), "expected str"),
    (lambda r: r.__setitem__("image_caption", ""), "image_caption is blank"),
])
def test_row_problems_catches(mutate, substr):
    row = _row()
    mutate(row)
    problems = think_row_problems(row)
    assert any(substr in p for p in problems), (substr, problems)


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD].replace("hard hat ->", "hard hat. ->")),
    lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD].replace("\n", "\r\n")),
    lambda r: r.__setitem__(THINK_FIELD, r[THINK_FIELD].replace("-> yes (1)", "->  yes ( 1 )")),
])
def test_row_problems_tolerates_cosmetic_differences(mutate):
    """Blocking a 12k-row dataset over a trailing period data prep left in place would
    be a worse outcome than accepting it; nothing about the MEANING differs."""
    row = _row()
    mutate(row)
    assert think_row_problems(row) == []
