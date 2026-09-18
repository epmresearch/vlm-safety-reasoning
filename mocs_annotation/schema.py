"""
Contract for one auto-annotation proposal, and its conversion to dataset scale.

Deliberately NOT registered in data/schemas.py::SCHEMA_REGISTRY. That registry is
keyed by *task* and every metric/reward/repair gate in the repo derives from it;
adding a non-task entry there would make `get_output_schema` answer for something
that is not a pipeline task. This is a local contract for an offline tool.

It does REUSE data/schemas.py::RuleViolation, so a proposal's violation objects are
validated by exactly the same model the training pipeline uses. A proposal that
would not survive `ViolationsOnlyOutput` cannot survive here either.

SCALE, the thing that is easiest to get wrong in this repo (CLAUDE.md, "Rewards and
the output contract"):

    the teacher is asked for boxes in [0, 1000]   (Qwen convention, what the model emits)
    ConstructionSite ground truth is in [0, 1]    (dataset native)

`proposal_to_record` converts 1000 -> [0, 1] via data/box_utils.py::scale_1000_to_01
and then filters with `clean_boxes`, whose validity range is [0, 1]. Skipping the
conversion silently collapses every box to a point and zeroes every IoU downstream.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from pydantic import BaseModel, field_validator

from core.constants import RULES
from data.box_utils import clean_boxes, normalize_boxes, scale_1000_to_01
from data.schemas import RuleViolation
from evaluation.output_parser import strip_fences

# Status values written into proposals.jsonl. Anything other than "ok" carries an
# `error` string and no usable payload -- a failed call is never written out as an
# empty-but-successful record, which is the bug CLAUDE.md documents for the old
# inference retry path (a hard failure that looked like a model emitting nothing).
STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SCHEMA_ERROR = "schema_error"
STATUS_GENERATION_ERROR = "generation_error"


class AnnotationProposal(BaseModel):
    """What the teacher VLM is asked to return for one image."""

    caption: str
    rule_1_violation: Optional[RuleViolation] = None
    rule_2_violation: Optional[RuleViolation] = None
    rule_3_violation: Optional[RuleViolation] = None
    rule_4_violation: Optional[RuleViolation] = None

    @field_validator("caption")
    @classmethod
    def _caption_must_not_be_blank(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("caption must be a non-empty, non-whitespace string")
        return v


def parse_proposal(raw_text: str) -> Tuple[Optional[AnnotationProposal], Optional[str], Optional[str]]:
    """Parses a raw completion into an AnnotationProposal.

    Returns (proposal, status, error). On success status is STATUS_OK and error None.

    Uses the repo's own `strip_fences`, which regex-searches for a ```json ... ```
    block ANYWHERE in the string. That means prose before the fence is tolerated for
    free -- useful because a 32B model often narrates before answering, and it is
    also exactly why a future scratchpad-then-JSON training target needs no parser
    change.
    """
    if raw_text is None or not str(raw_text).strip():
        return None, STATUS_PARSE_ERROR, "empty completion"

    text = strip_fences(str(raw_text))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return None, STATUS_PARSE_ERROR, f"json decode failed: {e}"

    if not isinstance(payload, dict):
        return None, STATUS_PARSE_ERROR, f"expected a JSON object, got {type(payload).__name__}"

    payload = _normalize_violation_shapes(payload)

    try:
        return AnnotationProposal(**payload), STATUS_OK, None
    except Exception as e:  # pydantic ValidationError and anything it wraps
        return None, STATUS_SCHEMA_ERROR, str(e)[:500]


def _normalize_violation_shapes(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerces the four violation values into the None | {reason, bounding_box} shape
    before Pydantic sees them, by reusing the repair stage's own normalizer.

    WHY REUSE preprocessing/structural_repair.py::normalize_violation_value. A model
    told to emit `null` or an object still sometimes emits `true`, a bare reason
    string, or a list of one object per instance -- all three occur in this repo's
    real prediction files. That function already handles every one of those shapes,
    has been exercised on 27 real runs, and (critically) its list branch UNIONS boxes
    and joins reasons rather than returning None: returning None there does not drop
    a repair, it INVERTS the answer to "not violated", which cost one real run 182
    true positives. Reimplementing this by hand is how you get that bug back.

    Scoped to the four rule keys only. `fix_prediction_structure` is deliberately NOT
    used: it normalizes TOP-LEVEL keys against a task's canonical key set, and this
    payload carries `caption`, which is not in violations_only's key set.
    """
    from preprocessing.structural_repair import normalize_violation_value

    out = dict(payload)
    for r in RULES:
        key = f"{r}_violation"
        if key in out:
            out[key] = normalize_violation_value(out[key], rule_key=key, tracker=None)
    return out


def _violation_to_dataset_scale(v: Optional[RuleViolation]) -> Optional[Dict[str, Any]]:
    """One violation object, boxes converted [0,1000] -> [0,1] and cleaned.

    Returns None when the value carries neither a usable box nor a reason, matching
    `preprocessing/structural_repair.py`'s own "a list with no usable box and no
    usable reason collapses to null" rule. A violation that keeps only its reason is
    preserved: `_is_substantive_violation` counts a reason-only assertion as
    substantive, and the engineer can draw the box during review.
    """
    if v is None:
        return None

    raw_boxes = normalize_boxes(v.bounding_box or [])
    boxes = clean_boxes([scale_1000_to_01(b) for b in raw_boxes])
    reason = (v.reason or "").strip()

    if not boxes and not reason:
        return None
    return {"bounding_box": [list(b) for b in boxes], "reason": reason}


def proposal_to_record(
    proposal: AnnotationProposal,
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalises a validated proposal into the annotation record we persist.

    This is NOT yet a HuggingFace dataset row -- it carries no decoded image and no
    metadata columns. Assembling the actual training rows (and matching the column
    set of datasets/processed exactly: `image`, `image_id`, `image_caption`, the four
    `rule_N_violation` fields, `excavator` / `rebar` /
    `worker_with_white_hard_hat`, the four metadata columns, and `resolution`) is the
    job of the separate combine step, run after human verification.

    Boxes here are already in dataset [0, 1] scale, which is what
    data/preprocessor.py::build_target_json expects to receive and then scales up to
    [0, 1000] itself for the SFT target.
    """
    rules: Dict[str, Any] = {}
    for r in RULES:
        rules[f"{r}_violation"] = _violation_to_dataset_scale(
            getattr(proposal, f"{r}_violation", None)
        )

    flagged = [r for r in RULES if rules[f"{r}_violation"] is not None]

    return {
        "new_image_id": selection["new_image_id"],
        "mocs_image_id": selection["mocs_image_id"],
        "file_name": selection["file_name"],
        "width": selection["width"],
        "height": selection["height"],
        "selection_bucket": selection["selection_bucket"],
        # Which MOCS file the image came from ("val" / "test"). Absent on selections
        # made before multi-source support, hence .get() -- the filename number range
        # still distinguishes them (val 19406-23406, test 23407-41672).
        "source": selection.get("source", ""),
        "status": STATUS_OK,
        "image_caption": proposal.caption.strip(),
        **rules,
        "flagged_rules": flagged,
        # Kept so the reviewer and any later audit can see what the model was told.
        "mocs_categories": selection.get("mocs_categories", []),
        "worker_machine_pairs": selection.get("worker_machine_pairs", []),
    }


def failure_record(
    selection: Dict[str, Any],
    status: str,
    error: str,
    raw_output: str = "",
) -> Dict[str, Any]:
    """A record for an image the teacher could not usefully answer.

    Written to proposals.jsonl too, with status != "ok", so the file always has
    exactly one line per attempted image. That is what makes the resume logic in
    annotate.py a simple "skip ids already present" without ever double-writing an
    image -- the failure mode CLAUDE.md describes for the old auto-resume path.
    """
    return {
        "new_image_id": selection["new_image_id"],
        "mocs_image_id": selection["mocs_image_id"],
        "file_name": selection["file_name"],
        "width": selection["width"],
        "height": selection["height"],
        "selection_bucket": selection["selection_bucket"],
        "source": selection.get("source", ""),
        "status": status,
        "error": error,
        "raw_output": (raw_output or "")[:4000],
    }


def has_any_rule(record: Dict[str, Any], rules: Sequence[str] = ("rule_2", "rule_3", "rule_4")) -> bool:
    """True if the record asserts at least one of `rules`. Used to build the
    review queue: the harvest only cares about the rare rules, even though the
    teacher is asked about all four (see the note in annotate.py on why rule_1
    must still be annotated rather than nulled)."""
    return any(record.get(f"{r}_violation") is not None for r in rules)
