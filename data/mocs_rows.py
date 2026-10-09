"""
Turns the human-verified MOCS review file into ConstructionSite-shaped dataset rows.

This is the PURE half of the v3 dataset build: review JSON in, plain dicts out. No
HuggingFace, no PIL, no albumentations, no filesystem beyond reading the one JSON --
so the whole conversion, including every drop/repair decision, is unit-testable on a
laptop with no data root. ``data/build_v3_datasets.py`` is the half that needs ARC.

WHAT THE REVIEW FILE IS. ``mocs_annotation/review_ui.py`` writes one
``dataset_rows`` entry per image a human marked "Use it" -- 266 of 985 decided, the
other 719 being rejected or unsure. Each entry carries the reviewer's FINAL caption
and their final per-rule verdict, reason and boxes, plus a ``provenance`` sub-dict
recording how much of it came from the teacher model versus the human.

THREE THINGS THIS MODULE FIXES, AND WHY EACH IS NOT A JUDGEMENT CALL

  1. ``provenance`` is stripped out of every violation object.
     ``datasets/processed``'s violation struct is exactly ``{bounding_box, reason}``
     (data/schemas.py::RuleViolation). An extra key changes the Arrow struct type, so
     ``concatenate_datasets`` with the ConstructionSite rows would fail -- or worse,
     coerce. The provenance is still reported in the build manifest, just not on the
     row.

  2. ONE RULE FOR A BAD ROW: if it cannot produce a clean, fully-grounded think
     block, the whole ROW is dropped. Nothing is ever repaired in place.

     Two shapes trigger it in the current file, and both lose a verified image:

       * ``mocs_0022103`` -- ``image_caption`` is blank. The reviewer set
         ``caption_ok: "n"``, i.e. explicitly rejected the teacher's caption too, so
         there is no honest fallback, and writing one by hand would put authored text
         on the first line of a ``violations_think`` block.
       * ``mocs_0023379`` -- ``rule_1`` marked violated with no reason and no box
         (its own provenance says the model never proposed it and the reviewer drew
         nothing). This row also carries a genuinely verified ``rule_2`` with two
         boxes, and **that is lost with it** -- one of the 153 rule_2 images the
         harvest exists to buy.

     The alternative considered and rejected was nulling just the offending rule and
     keeping the rest of the image. It is cheaper in rows, and more expensive in
     everything else: nulling writes an assertion the reviewer MADE back out as "not
     violated", a label no human produced; where boxes are present it also discards
     verified geometry while leaving the image in the dataset asserting the opposite;
     and it splits the policy in two, so "which rows are in, and why" stops being one
     sentence anybody can check. One rule, applied uniformly, with every drop listed
     by id and cause in the manifest.

     Why it cannot simply be left alone: ``core/think_format.py::build_think_body``
     refuses to emit a block line for a reason-less assertion, so the row would fail
     the SFT job outright; and even for ``violations_only`` the shape costs twice --
     ``_is_substantive_violation`` scores it as a MISS on a real violation while
     ``_is_violation_present`` still counts it as a false alarm.

  3. ``rule_4`` boxes are unioned to one, by default.

     The reward never counts boxes. ``rewards/reward_violation_grounding.py`` scores
     with ``compute_mask_union_iou``, which rasterises the UNION of the predicted
     boxes against the union of the GT boxes -- so the question is not "one box or
     three", it is whether the covered REGION matches GT's.

     Measured on ConstructionSite (``dataset_report.py`` output, built on ARC):
     rule_1 1.66 and rule_2 1.74 boxes per violated image, but rule_3 1.16 and rule_4
     **1.01** -- and in the TEST split, the key everything is scored against, rule_3
     and rule_4 are **exactly 1.00** (rule_4: 24 boxes on 24 images). CS train is 41
     single-box rule_4 images and one double.

     94 of the 113 verified MOCS rule_4 rows already carry a single box, so leaving
     the other 19 alone produces a MIXED convention (136 single vs 19 multi across the
     combined set) with no consistent habit to learn. Unioning makes it 154 single / 1
     double, matching both CS and the test key. The gap is not cosmetic: mask-union
     IoU between the 18 rows' boxes as drawn and their enclosing box is mean 0.571,
     median 0.626, min 0.098.

     It is NOT a claim that one box is better annotation -- a GT group box is drawn by
     a human and may be generous, while ``union_box`` is the minimal enclosing
     rectangle, so this approximates GT's convention rather than reproducing it. And
     the training prompt says the opposite ("List more than one box if more than one
     instance violates the same rule"), a contradiction that predates this file and
     is live in v2. Because it rewrites human-drawn geometry either way, it is a named
     policy (``rule4_box_policy="keep"`` reverts it) and every affected id is listed
     in the report. rule_1 and rule_2 are untouched.

WHAT THIS MODULE DELIBERATELY DOES NOT DO

  * **It does not rescale boxes.** The review file declares ``box_scale:
    "xyxy_0_1"`` and :func:`load_review_file` refuses any other value. The boxes are
    already in ConstructionSite's native [0,1]; running ``scale_1000_to_01`` over
    them again collapses every box to a point and silently zeroes every IoU
    downstream (CLAUDE.md, "Rewards and the output contract"). The schema check is
    the guard that makes a future review file in a different scale fail loudly
    instead of training on points.

  * **It does not invent object annotations.** ``excavator``, ``rebar`` and
    ``worker_with_white_hard_hat`` are written as ``[]`` because MOCS carries none of
    them -- its 13 categories are vehicles and an unqualified ``Worker``, with no
    PPE, no materials and no boxes that mean what ConstructionSite means. For a MOCS
    row ``[]`` reads as *unannotated*, not *absent*. That is harmless for
    ``violations_only``/``violations_think``, which never read those columns, and it
    is exactly why the ``provenance`` column exists and why MOCS rows must never be
    written into ``datasets/augmented`` (CLAUDE.md; PLAN_V3_THINK.md trap 9).
    ``"Excavator" in mocs_categories`` is NOT used: the column holds boxes, not a
    boolean, and the category list is empty on all 67 ``source=="test"`` rows because
    MOCS's test split ships no annotations at all -- so deriving it would be both
    geometry-free and systematically wrong for a quarter of the rows.

  * **It does not fill the four metadata columns.** ``illumination``,
    ``camera_distance``, ``view`` and ``quality_of_info`` are written as ``""``.
    Nothing trains on them (core/constants.py:61-64); their only consumer is
    ``evaluation/error_analyzer.py``, which strata evaluation results by them. ``""``
    reads as "unlabelled", which is true. Any guessed value would be a fabricated
    stratum label.

  * **It does not normalise reason wording.** The rubric notes the dataset writes
    "operation radius" while several verified rule_4 reasons say "operating radius".
    Changing verified human text to chase ``reward_reasoning`` string similarity is a
    data decision, not a build step; the count is reported so it can be made
    deliberately.

  * **It does not build the ``thinking`` column.** ConstructionSite rows need one
    too, and a single code path for both is the only way the two provenances cannot
    drift. ``data/build_v3_datasets.py`` does it, for every row, through
    ``core/think_format.py::build_think_body``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.constants import GROUNDING_CLASSES, METADATA_FIELDS, RULES
from data.box_utils import clean_boxes, normalize_boxes

# The review file self-describes. Both are asserted rather than assumed: the box
# scale especially, because getting it wrong is silent (CLAUDE.md) and a future
# review-UI version could plausibly emit [0,1000] to match the model's own
# convention.
REVIEW_SCHEMA = "mocs_review/2"
REVIEW_BOX_SCALE = "xyxy_0_1"

# The `provenance` COLUMN's two values. Reported, never trained on -- but it is what
# makes "this row has no object annotations" recoverable from the dataset itself
# rather than from a manifest somebody has to remember to read.
PROVENANCE_CS = "constructionsite"
PROVENANCE_MOCS = "mocs"

# Keys of the per-violation provenance sub-dict that review_ui.py writes and that
# must not reach the dataset (see the module docstring, point 1).
_REVIEW_ONLY_VIOLATION_KEYS = ("provenance",)

RULE4 = "rule_4"


@dataclass(frozen=True)
class MocsImageRef:
    """Where one MOCS row's pixels can be found, in the three places they exist.

    Resolution happens in ``build_v3_datasets.py`` because it touches the filesystem;
    carrying all three candidates here keeps that step a pure lookup.
    """

    image_id: str
    file_name: str          # "0020121.jpg" -- the MOCS basename
    source: str             # "val" | "test" -- the MOCS split, NOT a ConstructionSite one
    path_original: str      # absolute ARC path recorded at annotation time
    review_image: str       # basename of the staged render, "<image_id>.jpg"
    width: int
    height: int


@dataclass
class MocsConversionReport:
    """Everything the conversion changed or refused, for the build manifest.

    Deliberately exhaustive: every departure from "the human said so" is listed by
    image id, because a reader six months from now cannot re-derive these from the
    output dataset.
    """

    rows_in: int = 0
    rows_out: int = 0
    # `dropped` is whole ROWS only, so `rows_in - rows_out == len(dropped)` always
    # holds and anyone checking the manifest's arithmetic gets a straight answer.
    # Every entry carries a `cause`, and `dropped_by_cause` tallies them.
    dropped: List[Dict[str, str]] = field(default_factory=list)
    rule4_boxes_unioned: List[Dict[str, Any]] = field(default_factory=list)
    boxes_rejected: List[Dict[str, Any]] = field(default_factory=list)
    reason_only_violations: List[Dict[str, str]] = field(default_factory=list)
    zero_violation_rows: List[str] = field(default_factory=list)
    # Reported, not changed -- see the module docstring.
    operating_radius_reasons: List[Dict[str, str]] = field(default_factory=list)
    by_rule: Dict[str, int] = field(default_factory=dict)
    boxes_by_rule: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    provenance_tallies: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # The three lists above that describe PER-RULE findings are filled while a row is
    # still being converted, i.e. before it is known whether a later rule will take the
    # row down. They are therefore written into a scratch report and merged only once
    # the row is committed -- otherwise a dropped row would leave findings behind
    # describing a row that is not in the dataset.
    _ROW_SCOPED = ("rule4_boxes_unioned", "boxes_rejected", "reason_only_violations",
                   "operating_radius_reasons")

    def drop_row(self, image_id: str, cause: str, detail: str, rule: str = "") -> None:
        entry = {"image_id": image_id, "cause": cause, "reason": detail}
        if rule:
            entry["rule"] = rule
        self.dropped.append(entry)

    def merge_row(self, scratch: "MocsConversionReport") -> None:
        """Commits a surviving row's per-rule findings."""
        for name in self._ROW_SCOPED:
            getattr(self, name).extend(getattr(scratch, name))

    @property
    def dropped_by_cause(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self.dropped:
            out[d["cause"]] = out.get(d["cause"], 0) + 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_count": len(self.dropped),
            "dropped_by_cause": self.dropped_by_cause,
            "dropped": self.dropped,
            "rule4_boxes_unioned_count": len(self.rule4_boxes_unioned),
            "rule4_boxes_unioned": self.rule4_boxes_unioned,
            "boxes_rejected": self.boxes_rejected,
            "reason_only_violations": self.reason_only_violations,
            "zero_violation_rows": self.zero_violation_rows,
            "operating_radius_reasons_count": len(self.operating_radius_reasons),
            "operating_radius_reasons": self.operating_radius_reasons,
            "by_rule": self.by_rule,
            "boxes_by_rule": self.boxes_by_rule,
            "by_source": self.by_source,
            # Named for what it is: this counts every review entry's reviewer effort,
            # INCLUDING entries later dropped. It describes the review file, not the
            # dataset -- use `by_rule` for the dataset.
            "provenance_tallies_over_review_file": self.provenance_tallies,
        }


def load_review_file(path: str | Path) -> Dict[str, Any]:
    """Reads and SCHEMA-CHECKS the review file.

    Raises:
        ValueError: if ``schema`` or ``box_scale`` is not what this module was
            written against. The box-scale check is the important one -- a review
            file in [0,1000] would convert cleanly, train, and produce boxes a
            thousand times too small with no error anywhere.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != REVIEW_SCHEMA:
        raise ValueError(
            f"{path}: schema is {schema!r}, expected {REVIEW_SCHEMA!r}. "
            "Re-check mocs_annotation/review_ui.py before trusting this file."
        )
    scale = payload.get("box_scale")
    if scale != REVIEW_BOX_SCALE:
        raise ValueError(
            f"{path}: box_scale is {scale!r}, expected {REVIEW_BOX_SCALE!r}. "
            "Boxes are written into the dataset UNSCALED; a different scale here "
            "would silently zero every IoU downstream."
        )
    if not isinstance(payload.get("dataset_rows"), list):
        raise ValueError(f"{path}: no 'dataset_rows' list found.")
    return payload


def union_box(boxes: Sequence[Sequence[float]]) -> List[float]:
    """The smallest axis-aligned box containing all of ``boxes``."""
    xs = [c for b in boxes for c in (b[0], b[2])]
    ys = [c for b in boxes for c in (b[1], b[3])]
    return [min(xs), min(ys), max(xs), max(ys)]


class UnusableViolation(Exception):
    """A violation value that cannot become a clean dataset value. Drops its ROW.

    Carries ``rule`` and ``cause`` so the drop can be reported by category rather than
    as an opaque string.
    """

    def __init__(self, rule: str, cause: str, detail: str = ""):
        super().__init__(f"{rule}: {cause}{(' (' + detail + ')') if detail else ''}")
        self.rule = rule
        self.cause = cause
        self.detail = detail


def _clean_violation(
    value: Any,
    rule: str,
    image_id: str,
    report: MocsConversionReport,
    rule4_box_policy: str,
) -> Optional[Dict[str, Any]]:
    """One review violation -> the dataset's ``{bounding_box, reason}`` struct, or None.

    Raises:
        UnusableViolation: when the value cannot be represented cleanly. The caller
            drops the whole ROW -- see :func:`review_row_to_dataset_row` and the
            module docstring's point 2 for why that, rather than nulling the one rule.

    Boxes go through the repo's own ``clean_boxes(normalize_boxes(...))``, which
    enforces the [0,1] range and drops degenerate ones -- the same filter
    ``build_target_json`` would apply later anyway, run here so a rejection is
    *recorded* instead of happening invisibly at training time.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        # `review_ui.py` always writes dicts, so this is defensive -- but what it is
        # defensive against is an INVERSION. Reading a bare `true` as None does not
        # drop a label, it FLIPS it to "not violated", which is the same failure class
        # as the structural_repair list-drop bug CLAUDE.md documents (182 destroyed
        # true positives on one run). Dropping the row cannot invert anything.
        raise UnusableViolation(
            rule, "violation value is not an object", type(value).__name__)

    submitted = value.get("bounding_box")
    n_submitted = len(submitted) if isinstance(submitted, list) else (0 if submitted is None else 1)
    boxes = clean_boxes(normalize_boxes(submitted))
    if len(boxes) != n_submitted:
        # Counted against what the reviewer actually submitted, not against
        # normalize_boxes' output -- normalize_boxes silently discards anything that is
        # not a list/tuple/recognised dict, so comparing downstream of it hides exactly
        # the malformed elements this report exists to surface.
        report.boxes_rejected.append(
            {"image_id": image_id, "rule": rule,
             "kept": len(boxes), "submitted": n_submitted}
        )

    reason = str(value.get("reason") or "").strip()

    if not reason:
        # No reason, with or without boxes. Both shapes drop the row.
        #
        # `build_think_body` refuses to emit a block line for a reason-less assertion,
        # so the row could not carry a think block either way; the only question was
        # whether to null the rule and keep the rest of the image. It is not nulled,
        # for two reasons. Nulling writes an assertion the reviewer MADE back out as
        # "not violated" -- a label this pipeline never saw a human produce -- and with
        # boxes present it also throws away verified geometry while leaving the image
        # in the dataset asserting the opposite. And it splits the policy in two:
        # "drop the sample" is one rule a reader can hold, check and defend, where
        # "drop some, silently repair others" is two.
        raise UnusableViolation(
            rule,
            "asserted with no reason",
            f"{len(boxes)} box(es) discarded with it" if boxes else "and no box",
        )

    # The union runs AFTER the reason check, so a violation that is about to take its
    # row down with it never shows up in rule4_boxes_unioned -- that list must describe
    # the dataset, not the attempts.
    if rule == RULE4 and rule4_box_policy == "union" and len(boxes) > 1:
        before = [list(b) for b in boxes]
        boxes = [union_box(boxes)]
        report.rule4_boxes_unioned.append(
            {"image_id": image_id, "before": before, "after": [list(boxes[0])]}
        )

    if not boxes:
        # Legal and preserved: core/think_format.py emits "rule_N: <reason> -> yes"
        # with the count omitted, and _is_substantive_violation counts a reason-only
        # assertion as substantive. It does score 0 on reward_violation_grounding,
        # which is why it is counted here rather than waved through.
        report.reason_only_violations.append(
            {"image_id": image_id, "rule": rule, "text": reason[:120]}
        )

    if rule == RULE4 and "operating radius" in reason.lower():
        report.operating_radius_reasons.append(
            {"image_id": image_id, "rule": rule, "text": reason[:160]}
        )

    return {"bounding_box": [list(b) for b in boxes], "reason": reason}


def review_row_to_dataset_row(
    review_row: Dict[str, Any],
    report: MocsConversionReport,
    rule4_box_policy: str = "union",
) -> Optional[Tuple[Dict[str, Any], MocsImageRef]]:
    """One ``dataset_rows`` entry -> (dataset row without its image, image reference).

    Returns ``None`` -- and records why, by category -- for a row that cannot become a
    training row. **One rule: if the row cannot produce a clean, fully-grounded think
    block, the whole row goes.** The two shapes that trigger it today:

      * a blank ``image_caption``. There is no honest repair: ``caption_model`` still
        holds the teacher's caption, but the reviewer set ``caption_ok: "n"`` on that
        very row, i.e. explicitly rejected it, and writing one by hand would put
        authored text on the first line of a ``violations_think`` block -- the one
        property that makes the arm defensible ("every token is ground truth",
        core/think_format.py).
      * any rule asserted without a usable reason (see :class:`UnusableViolation`).
    """
    image_id = str(review_row.get("new_image_id") or "").strip()
    if not image_id:
        report.drop_row(
            "<missing>", "no_image_id", "entry carries no new_image_id")
        return None

    caption = str(review_row.get("image_caption") or "").strip()
    if not caption:
        report.drop_row(
            image_id, "blank_caption",
            "image_caption is blank; no honest repair exists "
            "(caption_ok=='n' means the reviewer rejected the model's one too)")
        return None

    row: Dict[str, Any] = {
        "image_id": image_id,
        "image_caption": caption,
    }
    # Per-rule findings go to a scratch report first: rule_1 may be converted (and its
    # boxes unioned, and that recorded) before rule_2 turns out to be unusable and
    # takes the row down. Committing as we went would leave the manifest describing
    # rows that are not in the dataset.
    scratch = MocsConversionReport()
    try:
        for rule in RULES:
            row[f"{rule}_violation"] = _clean_violation(
                review_row.get(f"{rule}_violation"), rule, image_id, scratch,
                rule4_box_policy
            )
    except UnusableViolation as exc:
        report.drop_row(image_id, exc.cause, str(exc), rule=exc.rule)
        return None

    # MOCS carries none of these; see the module docstring.
    for cls in GROUNDING_CLASSES:
        row[cls] = []
    for meta in METADATA_FIELDS:
        row[meta] = ""

    width = int(review_row.get("width") or 0)
    height = int(review_row.get("height") or 0)
    # `resolution` is pixel AREA (notebooks/05_dataset_optimization.ipynb writes
    # img.width * img.height), not a (w, h) pair. It feeds ResolutionBucketSampler,
    # which is off today (configs/sft.yaml: use_resolution_bucketing: false) but is
    # still read and length-checked by experiments/run_sft.py.
    row["resolution"] = width * height
    row["provenance"] = PROVENANCE_MOCS

    ref = MocsImageRef(
        image_id=image_id,
        file_name=str(review_row.get("file_name") or ""),
        source=str(review_row.get("source") or ""),
        path_original=str(review_row.get("image_path_original") or ""),
        review_image=str(review_row.get("review_image") or f"{image_id}.jpg"),
        width=width,
        height=height,
    )
    if not ref.file_name and not ref.path_original:
        report.drop_row(image_id, "no_image_path", "no way to locate the image")
        return None

    report.merge_row(scratch)          # the row is committed; so are its findings
    return row, ref


def build_mocs_rows(
    review: Dict[str, Any],
    rule4_box_policy: str = "union",
) -> Tuple[List[Dict[str, Any]], List[MocsImageRef], MocsConversionReport]:
    """Every accepted review row, converted. Returns (rows, image refs, report).

    Rows and refs are index-aligned. Order follows the review file so the build is
    reproducible without sorting.
    """
    if rule4_box_policy not in ("union", "keep"):
        raise ValueError(f"rule4_box_policy must be 'union' or 'keep', got {rule4_box_policy!r}")

    report = MocsConversionReport()
    entries = review.get("dataset_rows") or []
    report.rows_in = len(entries)

    rows: List[Dict[str, Any]] = []
    refs: List[MocsImageRef] = []
    seen: Dict[str, int] = {}
    prov_tally = {r: {k: 0 for k in
                      ("n", "proposed_by_model", "reason_edited",
                       "model_boxes_kept", "model_boxes_deleted", "reviewer_boxes")}
                  for r in RULES}

    for entry in entries:
        # Provenance is tallied from the RAW entry, before _clean_violation strips it.
        for rule in RULES:
            v = entry.get(f"{rule}_violation")
            if not isinstance(v, dict):
                continue
            p = v.get("provenance") or {}
            t = prov_tally[rule]
            t["n"] += 1
            t["proposed_by_model"] += int(bool(p.get("proposed_by_model")))
            t["reason_edited"] += int(bool(p.get("reason_edited")))
            t["model_boxes_kept"] += int(p.get("model_boxes_kept") or 0)
            t["model_boxes_deleted"] += int(p.get("model_boxes_deleted") or 0)
            t["reviewer_boxes"] += int(p.get("reviewer_boxes") or 0)

        converted = review_row_to_dataset_row(entry, report, rule4_box_policy)
        if converted is None:
            continue
        row, ref = converted

        # `new_image_id` is unique in the current file and `mocs_image_id` is NOT
        # (MOCS restarts COCO ids per split, so val and test collide). Asserting it
        # rather than trusting it: a duplicate image_id would make two rows
        # indistinguishable to every manifest, sampler and leakage check downstream.
        if row["image_id"] in seen:
            report.drop_row(row["image_id"], "duplicate_image_id",
                            "a row with this new_image_id was already taken")
            continue
        seen[row["image_id"]] = 1

        rows.append(row)
        refs.append(ref)

    report.rows_out = len(rows)
    report.by_rule = {
        r: sum(1 for row in rows if row[f"{r}_violation"] is not None) for r in RULES
    }
    report.boxes_by_rule = {
        r: sum(len(row[f"{r}_violation"]["bounding_box"])
               for row in rows if row[f"{r}_violation"] is not None)
        for r in RULES
    }
    report.by_source = {}
    for ref in refs:
        report.by_source[ref.source] = report.by_source.get(ref.source, 0) + 1
    report.zero_violation_rows = [
        row["image_id"] for row in rows
        if all(row[f"{r}_violation"] is None for r in RULES)
    ]
    report.provenance_tallies = prov_tally
    return rows, refs, report
