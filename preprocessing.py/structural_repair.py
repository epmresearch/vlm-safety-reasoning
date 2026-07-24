"""
structural_repair.py

Standalone preprocessing script that repairs STRUCTURAL/FORMATTING problems
in raw VLM JSON outputs for the Unified Construction Safety Inspection
schema — WITHOUT ever changing actual semantic content — and TRACKS every
single fix it applies, per-record and in aggregate.

This script ONLY:
  - Extracts a JSON object from noisy text (fences, preamble, postamble,
    stray text, arrays-wrapping-an-object)
  - Repairs JSON *syntax* issues (trailing commas, smart quotes, Python
    literals True/False/None, comments, truncated/unterminated JSON where
    safely recoverable)
  - Normalizes KEY NAMES to the canonical schema (aliases, typos, casing,
    escaped underscores, flattened key patterns, alternate list structures)
  - Normalizes BOUNDING BOX structure (flat -> nested, dict-wrapped,
    string coordinates -> floats, extra nesting, multi-box flattening,
    and TWO-POINT corner-pair reconstruction, e.g.
    [[10,40],[890,450]] -> [[10,40,890,450]])
  - Normalizes VIOLATION structure (bool/string/dict variants, flattened
    per-rule keys, a single "violations" list instead of 4 separate keys)
  - Coerces obviously-numeric strings to floats (type only)

This script NEVER:
  - Invents a missing caption, reason, or bounding box coordinate
  - Changes any numeric value
  - Merges/averages/guesses content across records
  - Converts COCO xywh -> xyxy (removed on request — routed to
    still_broken.json instead, since it involves arithmetic on the numbers
    rather than pure renaming)
  - Joins multiple "reason" strings together (removed on request — a
    violation with a list-of-reasons is dropped instead of concatenated)
  - "Fixes" content that is genuinely missing/ambiguous — it drops it and
    reports it as still-broken instead of fabricating a substitute

TRACKING:
  Every transform function logs exactly what it did (change type, field,
  human-readable detail) to a per-record ChangeTracker. Two additional
  DIAGNOSTIC-ONLY warnings (which never modify data) are also tracked:
    - likely_truncated_output   : raw output looked cut off mid-generation
    - duplicate_box_detected    : same exact box repeated >N times in one
                                   field (sign of a generation loop, even
                                   if the JSON/schema end up perfectly valid)

Usage (standalone CLI):
    python structural_repair.py --input /path/to/predictions.jsonl

Outputs (next to the input file by default):
    predictions_repaired.jsonl   -- ORIGINAL schema preserved. raw_output is
                                     replaced with the repaired JSON string
                                     ONLY for records that needed fixing;
                                     original_raw_output and repair_status
                                     are added (additive, non-breaking —
                                     your eval pipeline only ever reads
                                     r["raw_output"] / r["sample"] by key).
    repair_report.json            -- aggregate status counts/percentages +
                                     a full summary of every change/warning
                                     type seen, with total occurrences and
                                     number of distinct records affected.
    change_manifest.json          -- ONE entry per record (every record,
                                     including fully clean ones) listing
                                     every change and warning applied to it
                                     — this is what lets you cross-check
                                     unprocessed vs. processed per image_id.
    still_broken.json             -- records that failed even after fixing,
                                     with raw output, error, and whatever
                                     fixes were attempted before it still
                                     failed, for manual review.
"""

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, conlist

# =============================================================================
# SCHEMA (mirrors data/schemas.py::UnifiedOutput — keep in sync if that changes)
# =============================================================================

BBox = conlist(float, min_length=4, max_length=4)


class RuleViolation(BaseModel):
    bounding_box: Optional[List[BBox]] = None
    reason: Optional[str] = None


class UnifiedOutput(BaseModel):
    caption: str
    rule_1_violation: Optional[RuleViolation] = None
    rule_2_violation: Optional[RuleViolation] = None
    rule_3_violation: Optional[RuleViolation] = None
    rule_4_violation: Optional[RuleViolation] = None
    excavator: List[BBox] = Field(default_factory=list)
    rebar: List[BBox] = Field(default_factory=list)
    worker_with_white_hard_hat: List[BBox] = Field(default_factory=list)


# =============================================================================
# CHANGE TRACKING
# =============================================================================

class ChangeTracker:
    """Collects every structural transform actually applied to ONE record,
    plus non-modifying diagnostic warnings (truncation, duplicate boxes).

    - `.log(...)`  -> an ACTUAL fix was applied (data was reshaped/renamed).
    - `.warn(...)` -> a DIAGNOSTIC-ONLY observation; nothing was changed.
    """

    def __init__(self):
        self.changes: List[Dict[str, str]] = []
        self.warnings: List[Dict[str, str]] = []

    def log(self, change_type: str, field: str = "", detail: str = ""):
        self.changes.append({"type": change_type, "field": field, "detail": detail})

    def warn(self, warning_type: str, field: str = "", detail: str = ""):
        self.warnings.append({"type": warning_type, "field": field, "detail": detail})


# =============================================================================
# SECTION 1 — JSON EXTRACTION FROM NOISY TEXT
# =============================================================================

_FENCE_PATTERNS = [
    re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```\s*(.*?)```", re.DOTALL),
    re.compile(r"<json>(.*?)</json>", re.DOTALL | re.IGNORECASE),
]


def _extract_outermost_braces(text: str) -> Optional[str]:
    """Finds the first balanced {...} block, respecting quoted strings so
    braces inside caption/reason text don't confuse the counter."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced -> likely truncated, handled separately


def extract_json_candidates(raw_str: str, tracker: Optional[ChangeTracker] = None) -> List[str]:
    """Returns an ordered, de-duplicated list of candidate JSON substrings to
    attempt parsing, from most-likely-correct to most-permissive fallback."""
    candidates = []

    for pattern in _FENCE_PATTERNS:
        for m in pattern.finditer(raw_str):
            candidates.append(m.group(1).strip())

    brace_candidate = _extract_outermost_braces(raw_str)
    if brace_candidate:
        candidates.append(brace_candidate)
    elif "{" in raw_str and tracker:
        # Balanced braces never found even though generation clearly started
        # a JSON object -> strong signal of max_new_tokens truncation.
        tracker.warn(
            "likely_truncated_output",
            detail="No balanced closing brace found in raw output; "
                   "generation may have been cut off by max_new_tokens.",
        )

    start = raw_str.find("{")
    if start != -1:
        candidates.append(raw_str[start:].strip())

    candidates.append(raw_str.strip())

    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# =============================================================================
# SECTION 2 — JSON SYNTAX REPAIR (string-aware: never touches text inside
# actual JSON string values, only the surrounding syntax)
# =============================================================================

def _split_string_segments(text: str):
    """Splits text into (is_inside_json_string, substring) segments so
    syntax fixes can be applied ONLY outside string literals."""
    segments = []
    cur_start = 0
    in_string = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                segments.append((True, text[cur_start:i + 1]))
                cur_start = i + 1
        else:
            if ch == '"':
                if cur_start < i:
                    segments.append((False, text[cur_start:i]))
                in_string = True
                cur_start = i
        i += 1
    if cur_start < n:
        segments.append((in_string, text[cur_start:]))
    return segments


_SYNTAX_FIXES = [
    ("json_syntax_block_comment_removed", re.compile(r"/\*.*?\*/", re.DOTALL), ""),
    ("json_syntax_line_comment_removed", re.compile(r"(?<!:)//[^\n]*"), ""),
    ("json_syntax_python_true_converted", re.compile(r"\bTrue\b"), "true"),
    ("json_syntax_python_false_converted", re.compile(r"\bFalse\b"), "false"),
    ("json_syntax_python_none_converted", re.compile(r"\bNone\b"), "null"),
    ("json_syntax_nan_converted", re.compile(r"\bNaN\b"), "null"),
    ("json_syntax_infinity_converted", re.compile(r"-?Infinity\b"), "null"),
    ("json_syntax_trailing_comma_removed", re.compile(r",\s*([}\]])"), r"\1"),
]


def sanitize_json_syntax(text: str, tracker: Optional[ChangeTracker] = None) -> str:
    """Fixes common JSON SYNTAX issues, applied only OUTSIDE string literals
    so caption/reason text is never rewritten. Logs each specific fix type
    that actually fired (with an occurrence count), not just a generic
    'sanitized' event."""
    segments = _split_string_segments(text)
    out = []
    for is_str, seg in segments:
        if is_str:
            out.append(seg)
            continue
        for change_type, pattern, repl in _SYNTAX_FIXES:
            seg, n = pattern.subn(repl, seg)
            if n > 0 and tracker:
                tracker.log(change_type, detail=f"{n} occurrence(s) fixed.")
        out.append(seg)
    result = "".join(out)

    before = result
    result = result.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    if result != before and tracker:
        tracker.log("json_syntax_smart_quotes_normalized")

    return result


def attempt_truncation_repair(snippet: str) -> Optional[str]:
    """Best-effort repair for a truncated JSON object caused by hitting
    max_new_tokens mid-generation: closes any unterminated string and any
    un-closed brackets/braces.

    If truncation happened mid-field, that incomplete trailing field is
    DROPPED (not guessed) — dropping an incomplete field is a structural
    fix; inventing its value would not be.
    """
    s = snippet
    stack = []
    in_string = False
    escape = False
    last_safe_cut = 0

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_safe_cut = i + 1
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe_cut = i + 1
        elif ch == ",":
            last_safe_cut = i  # cut BEFORE comma -> drops trailing incomplete field

    if in_string:
        s = s[:last_safe_cut]
        # recompute stack against the truncated string
        stack = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    s = re.sub(r"[,:]\s*$", "", s.rstrip())

    closers = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        s += closers[opener]

    return s if s.strip() else None


def _unwrap_if_list(result: Any) -> Any:
    """Some models wrap the object in an array: [{...}]. Unwrap the first
    dict element rather than failing outright."""
    if isinstance(result, list) and len(result) >= 1 and isinstance(result[0], dict):
        return result[0]
    return result


def robust_parse(raw_str: str, tracker: Optional[ChangeTracker] = None) -> Optional[Dict[str, Any]]:
    """Attempts, in increasing order of invasiveness, to turn a raw model
    string into a JSON dict. Every step fixes SYNTAX/STRUCTURE only.
    Returns the first successfully-parsed dict, or None if nothing works.

    Tracking note: sanitization is trialed SILENTLY first (no tracker) so
    that abandoned attempts on candidates that ultimately fail don't pollute
    the log; once we know which candidate/method actually succeeds, that
    exact path is re-run once WITH the tracker so logs reflect only the
    real, winning repair path.
    """
    if not raw_str or not raw_str.strip():
        return None

    candidates = extract_json_candidates(raw_str, tracker=tracker)

    for candidate in candidates:
        # Attempt 1: parse as-is (no fixing needed at all)
        try:
            result = _unwrap_if_list(json.loads(candidate))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Attempt 2: sanitize syntax (silent trial), then parse
        sanitized = sanitize_json_syntax(candidate, tracker=None)
        try:
            result = _unwrap_if_list(json.loads(sanitized))
            if isinstance(result, dict):
                if tracker:
                    sanitize_json_syntax(candidate, tracker=tracker)  # log the real fixes
                return result
        except json.JSONDecodeError:
            pass

        # Attempt 3: truncation repair, then parse
        repaired = attempt_truncation_repair(sanitized)
        if repaired:
            try:
                result = _unwrap_if_list(json.loads(repaired))
                if isinstance(result, dict):
                    if tracker:
                        sanitize_json_syntax(candidate, tracker=tracker)
                        tracker.log(
                            "json_truncation_repaired",
                            detail="Unterminated string/brackets closed; "
                                   "trailing incomplete field(s) dropped.",
                        )
                    return result
            except json.JSONDecodeError:
                pass

        # Attempt 4: Python-literal fallback
        try:
            result = _unwrap_if_list(ast.literal_eval(candidate))
            if isinstance(result, dict):
                if tracker:
                    tracker.log(
                        "json_python_literal_fallback_used",
                        detail="Parsed via Python-literal eval instead of strict JSON.",
                    )
                return result
        except (ValueError, SyntaxError, MemoryError, TypeError):
            pass

    return None


# =============================================================================
# SECTION 3 — KEY NAME NORMALIZATION
# =============================================================================

def _normalize_key_str(k: str) -> str:
    k = str(k).replace("\\_", "_").replace("\\", "")
    return re.sub(r"[\s\-]+", "_", k.strip().lower())


_KEY_ALIASES = {
    "caption": ["caption", "description", "image_caption", "img_caption",
                "scene_caption", "site_caption", "summary"],
    "excavator": ["excavator", "excavators", "excavator_boxes", "excavator_bboxes",
                  "excavator_bbox"],
    "rebar": ["rebar", "rebars", "rebar_boxes", "rebar_bboxes", "rebar_bbox"],
    "worker_with_white_hard_hat": [
        "worker_with_white_hard_hat", "worker_with_white_hardhat",
        "workers_with_white_hard_hat", "white_hard_hat_worker",
        "white_hard_hat_workers", "worker_white_hard_hat",
        "worker_with_hard_hat", "hard_hat_worker", "white_hardhat_worker",
    ],
}
for _i in range(1, 5):
    _KEY_ALIASES[f"rule_{_i}_violation"] = [
        f"rule_{_i}_violation", f"rule{_i}_violation", f"rule_{_i}violation",
        f"rule-{_i}-violation", f"violation_rule_{_i}", f"rule_{_i}",
    ]


def _build_alias_lookup() -> Dict[str, str]:
    lookup = {}
    for canonical, aliases in _KEY_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_key_str(alias)] = canonical
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()


def normalize_top_level_keys(d: dict, tracker: Optional[ChangeTracker] = None) -> dict:
    """Renames known key aliases/typos/casing variants to the canonical
    schema key names. Unknown keys are left untouched (pydantic will ignore
    them as extras — nothing is lost, nothing is invented)."""
    fixed = {}
    for k, v in d.items():
        nk = _normalize_key_str(k)
        canonical = _ALIAS_LOOKUP.get(nk)
        if canonical and canonical != k:
            if tracker:
                tracker.log("key_renamed", field=canonical, detail=f"'{k}' -> '{canonical}'")
            fixed[canonical] = v
        else:
            fixed[k] = v
    return fixed


def reconstruct_flattened_violations(d: dict, tracker: Optional[ChangeTracker] = None) -> dict:
    """Some models flatten a violation into separate top-level keys, e.g.
    "rule_1_reason" + "rule_1_bounding_box" instead of a nested
    rule_1_violation object. Reconstructs the nested object from whatever
    pieces are present — never invents a missing reason or box."""
    fixed = dict(d)
    for i in range(1, 5):
        vkey = f"rule_{i}_violation"
        if fixed.get(vkey):  # already has a truthy value, don't overwrite
            continue

        reason_val, box_val = None, None
        reason_src, box_src = None, None
        for rk in (f"rule_{i}_reason", f"rule{i}_reason", f"rule_{i}_explanation"):
            if rk in fixed:
                reason_val = fixed.pop(rk)
                reason_src = rk
                break
        for bk in (f"rule_{i}_bounding_box", f"rule_{i}_bbox",
                   f"rule_{i}_box", f"rule{i}_bbox"):
            if bk in fixed:
                box_val = fixed.pop(bk)
                box_src = bk
                break

        if reason_val is not None or box_val is not None:
            if tracker:
                tracker.log(
                    "violation_flattened_keys_merged", field=vkey,
                    detail=f"Merged separate '{reason_src or '-'}' / '{box_src or '-'}' "
                           f"keys into one nested object.",
                )
            fixed[vkey] = {"reason": reason_val, "bounding_box": box_val}
    return fixed


def reconstruct_violations_list(d: dict, tracker: Optional[ChangeTracker] = None) -> dict:
    """Some models output a single 'violations' list of
    {rule/rule_id, reason, bounding_box} objects instead of four separate
    rule_X_violation keys. Reconstructs the four canonical keys from it."""
    if "violations" not in d or not isinstance(d["violations"], list):
        return d

    fixed = dict(d)
    violations_list = fixed.pop("violations")

    for item in violations_list:
        if not isinstance(item, dict):
            continue
        item_norm = {_normalize_key_str(k): v for k, v in item.items()}

        rule_id = None
        for rule_key in ("rule", "rule_id", "rule_number", "id"):
            if rule_key in item_norm:
                digits = re.sub(r"\D", "", str(item_norm[rule_key]))
                if digits:
                    rule_id = int(digits)
                break
        if rule_id not in (1, 2, 3, 4):
            continue

        vkey = f"rule_{rule_id}_violation"
        if fixed.get(vkey):
            continue  # don't overwrite an existing entry for this rule

        reason = (item_norm.get("reason") or item_norm.get("reasoning")
                  or item_norm.get("explanation") or "")
        box = (item_norm.get("bounding_box") or item_norm.get("bbox")
               or item_norm.get("box") or [])
        if tracker:
            tracker.log(
                "violation_list_entry_expanded", field=vkey,
                detail="Reconstructed from an entry in a top-level 'violations' list.",
            )
        fixed[vkey] = {"reason": reason, "bounding_box": box}

    return fixed


# =============================================================================
# SECTION 4 — BOUNDING BOX STRUCTURE NORMALIZATION
# =============================================================================

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_comma_string(s: str) -> Optional[List[float]]:
    parts = [p.strip() for p in re.split(r"[,\s]+", s.strip()) if p.strip()]
    nums = [_to_float(p) for p in parts]
    if len(nums) == 4 and None not in nums:
        return nums
    return None


def _looks_like_point(x: Any) -> bool:
    """True if x is a 2-element [num, num] — a single corner point rather
    than a full box."""
    return (
        isinstance(x, (list, tuple)) and len(x) == 2
        and all(isinstance(c, (int, float)) or (isinstance(c, str) and _to_float(c) is not None)
                for c in x)
    )


def _reconstruct_boxes_from_point_pairs(
    raw_boxes: List[Any], field: str = "", tracker: Optional[ChangeTracker] = None
) -> Optional[List[List[float]]]:
    """Some models output each box as two separate 2-element corner points
    instead of one flat 4-element box, e.g.:
        [[10, 40], [890, 450]]                           -> ONE box
        [[10, 40], [890, 450], [900, 100], [1200, 300]]   -> TWO boxes
    Combines consecutive point pairs into flat [xmin,ymin,xmax,ymax] boxes.
    Never invents a coordinate — only regroups numbers already present.
    An unpaired trailing point (odd count) can't be safely matched to a
    partner and is dropped rather than guessed at.
    """
    if not raw_boxes:
        return None

    points = list(raw_boxes)
    if len(points) % 2 != 0:
        points = points[:-1]
        if tracker:
            tracker.log(
                "box_dropped_unpaired_point", field=field,
                detail="Odd number of 2-element corner points; last unpaired point dropped.",
            )

    boxes = []
    for i in range(0, len(points), 2):
        p1 = [_to_float(c) for c in points[i]]
        p2 = [_to_float(c) for c in points[i + 1]]
        if None in p1 or None in p2:
            continue
        boxes.append(p1 + p2)

    if boxes and tracker:
        tracker.log(
            "box_point_pairs_combined", field=field,
            detail=f"{len(boxes)} box(es) reconstructed from paired 2-element corner points.",
        )

    return boxes if boxes else None


def _dict_to_box(
    d: dict, field: str = "", tracker: Optional[ChangeTracker] = None
) -> Optional[List[float]]:
    """Reconstructs [xmin,ymin,xmax,ymax] from various dict key conventions:
    xmin/ymin/xmax/ymax, x_min/y_min/..., left/top/right/bottom, x1/y1/x2/y2,
    or a nested bounding_box/bbox/box wrapper.

    NOTE: COCO-style x/y/width/height -> xyxy conversion was intentionally
    REMOVED (per instruction) since it requires arithmetic on the numbers
    rather than pure renaming/restructuring; such dicts now fall through
    to still_broken.json instead of being silently converted.
    """
    d_norm = {_normalize_key_str(k): v for k, v in d.items()}

    def _find(keys):
        for k in keys:
            if k in d_norm:
                return _to_float(d_norm[k])
        return None

    xmin = _find(["xmin", "x_min", "left", "x1"])
    ymin = _find(["ymin", "y_min", "top", "y1"])
    xmax = _find(["xmax", "x_max", "right", "x2"])
    ymax = _find(["ymax", "y_max", "bottom", "y2"])
    if None not in (xmin, ymin, xmax, ymax):
        if tracker:
            tracker.log(
                "box_dict_corners_converted", field=field,
                detail="Named corner keys (xmin/ymin/xmax/ymax-style) converted to a flat box.",
            )
        return [xmin, ymin, xmax, ymax]

    for wrapper_key in ("bounding_box", "bbox", "box", "boxes", "coordinates", "coords"):
        if wrapper_key in d_norm and isinstance(d_norm[wrapper_key], (list, tuple)):
            inner = list(d_norm[wrapper_key])
            if len(inner) == 4:
                nums = [_to_float(c) for c in inner]
                if None not in nums:
                    if tracker:
                        tracker.log(
                            "box_dict_wrapper_unwrapped", field=field,
                            detail=f"Unwrapped dict key '{wrapper_key}' to a flat box.",
                        )
                    return nums
    return None


def normalize_boxes(
    raw_boxes: Any, field: str = "", tracker: Optional[ChangeTracker] = None
) -> List[List[float]]:
    """Coerces a model's bounding-box output for ONE class/rule into
    List[[xmin, ymin, xmax, ymax], ...] of floats. Handles: flat single box,
    flat multi-box (concatenated), nested-correctly, dict wrappers, TWO-POINT
    corner pairs, comma/space-separated strings, extra redundant nesting,
    and mixed lists. Never changes a numeric value — only restructures/
    retypes. Anything unreconstructable is dropped (not guessed) and logged.
    """
    if raw_boxes is None:
        return []
    if isinstance(raw_boxes, str):
        nums = _parse_comma_string(raw_boxes)
        if nums and tracker:
            tracker.log(
                "box_string_coordinates_parsed", field=field,
                detail="Comma/space-separated string parsed into a box.",
            )
        return [nums] if nums else []
    if isinstance(raw_boxes, dict):
        box = _dict_to_box(raw_boxes, field=field, tracker=tracker)
        return [box] if box else []
    if not isinstance(raw_boxes, (list, tuple)) or len(raw_boxes) == 0:
        return []

    # Unwrap redundant extra nesting: [[[a,b,c,d]]] -> [[a,b,c,d]]
    unwrapped_any = False
    while (len(raw_boxes) == 1 and isinstance(raw_boxes[0], (list, tuple))
           and len(raw_boxes[0]) > 0 and isinstance(raw_boxes[0][0], (list, tuple))):
        raw_boxes = raw_boxes[0]
        unwrapped_any = True
    if unwrapped_any and tracker:
        tracker.log("box_redundant_nesting_removed", field=field,
                     detail="Extra list nesting level(s) removed.")

    # NEW: two-point corner-pair pattern, e.g. [[10,40],[890,450]]
    if raw_boxes and all(_looks_like_point(b) for b in raw_boxes):
        combined = _reconstruct_boxes_from_point_pairs(raw_boxes, field=field, tracker=tracker)
        if combined is not None:
            return combined

    # Flat list of numbers/numeric-strings -> one box, or several boxes
    # concatenated together (clean multiple of 4)
    if all(isinstance(c, (int, float)) or (isinstance(c, str) and _to_float(c) is not None)
           for c in raw_boxes):
        nums = [_to_float(c) for c in raw_boxes]
        if len(nums) > 0 and len(nums) % 4 == 0:
            boxes = [nums[i:i + 4] for i in range(0, len(nums), 4)]
            if tracker:
                tracker.log(
                    "box_flat_list_reshaped", field=field,
                    detail=f"Flat list of {len(nums)} number(s) reshaped into {len(boxes)} box(es).",
                )
            return boxes
        if tracker:
            tracker.log(
                "box_dropped_unreconstructable", field=field,
                detail=f"Flat list of {len(nums)} number(s) is not a multiple of 4; dropped.",
            )
        return []

    valid_boxes = []
    for b in raw_boxes:
        if isinstance(b, (list, tuple)):
            if len(b) == 1 and isinstance(b[0], str):
                nums = _parse_comma_string(b[0])
                if nums:
                    valid_boxes.append(nums)
                    if tracker:
                        tracker.log(
                            "box_string_coordinates_parsed", field=field,
                            detail="Single-element list containing a comma-separated string parsed.",
                        )
                elif tracker:
                    tracker.log(
                        "box_dropped_unreconstructable", field=field,
                        detail="Single-element string box could not be parsed; dropped.",
                    )
                continue
            if len(b) == 4:
                nums = [_to_float(c) for c in b]
                if None not in nums:
                    valid_boxes.append(nums)
                elif tracker:
                    tracker.log(
                        "box_dropped_unreconstructable", field=field,
                        detail="4-element box had non-numeric coordinate(s); dropped.",
                    )
            else:
                if tracker:
                    tracker.log(
                        "box_dropped_unreconstructable", field=field,
                        detail=f"List element of length {len(b)} doesn't match any known box pattern; dropped.",
                    )
        elif isinstance(b, dict):
            box = _dict_to_box(b, field=field, tracker=tracker)
            if box:
                valid_boxes.append(box)
            elif tracker:
                tracker.log(
                    "box_dropped_unreconstructable", field=field,
                    detail="Dict element had no recognizable box keys; dropped.",
                )
        elif isinstance(b, str):
            nums = _parse_comma_string(b)
            if nums:
                valid_boxes.append(nums)
                if tracker:
                    tracker.log(
                        "box_string_coordinates_parsed", field=field,
                        detail="String element parsed as comma-separated box.",
                    )
            elif tracker:
                tracker.log(
                    "box_dropped_unreconstructable", field=field,
                    detail=f"String element '{b}' could not be parsed as a box; dropped.",
                )
        else:
            if tracker:
                tracker.log(
                    "box_dropped_unreconstructable", field=field,
                    detail=f"Element of type {type(b).__name__} is not a recognizable box; dropped.",
                )
    return valid_boxes


# =============================================================================
# SECTION 5 — VIOLATION VALUE NORMALIZATION
# =============================================================================

def normalize_violation_value(
    v: Any, rule_key: str = "", tracker: Optional[ChangeTracker] = None
) -> Optional[Dict[str, Any]]:
    """Normalizes a single rule_X_violation value into either None or
    {"reason": str, "bounding_box": [[...], ...]}. Never invents a reason
    string or box coordinates that weren't already present somewhere in v.
    NOTE: a list of multiple reason strings is DROPPED, not joined (removed
    on request — joining would alter phrasing/meaning)."""
    if v is None:
        return None

    if isinstance(v, bool):
        if tracker:
            tracker.log(
                "violation_bool_converted", field=rule_key,
                detail=f"Bare boolean '{v}' converted to "
                       f"{'an empty-but-present violation object' if v else 'null'}.",
            )
        return {"reason": "", "bounding_box": []} if v else None

    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in ("none", "null", "no violation", "n/a", "na", "false"):
            if tracker:
                tracker.log(
                    "violation_string_normalized_to_null", field=rule_key,
                    detail=f"String value '{v}' interpreted as no violation.",
                )
            return None
        if tracker:
            tracker.log(
                "violation_string_wrapped", field=rule_key,
                detail="Bare reason string wrapped into a violation object (no bounding box available).",
            )
        return {"reason": s, "bounding_box": []}

    if isinstance(v, list):
        if tracker:
            tracker.log(
                "violation_list_dropped", field=rule_key,
                detail="A list was given where an object/null was expected; too ambiguous to reconstruct, dropped.",
            )
        return None

    if isinstance(v, dict):
        key_map = {}
        for k in v.keys():
            nk = _normalize_key_str(k)
            if nk in ("reason", "reasoning", "explanation", "description", "cause"):
                key_map[k] = "reason"
            elif nk in ("bounding_box", "bounding_boxes", "bbox", "bboxes",
                        "box", "boxes", "coordinates", "coords"):
                key_map[k] = "bounding_box"
            else:
                key_map[k] = k

        fixed = {}
        for k, val in v.items():
            new_key = key_map.get(k, k)
            if new_key != k and tracker:
                tracker.log("violation_key_renamed", field=rule_key, detail=f"'{k}' -> '{new_key}'")
            fixed[new_key] = val

        if not fixed:
            if tracker:
                tracker.log(
                    "violation_empty_dict_treated_as_null", field=rule_key,
                    detail="Empty violation object interpreted as no violation.",
                )
            return None

        reason = fixed.get("reason", "")
        if reason is None:
            reason = ""
        if isinstance(reason, list):
            if tracker:
                tracker.log(
                    "violation_reason_list_dropped", field=rule_key,
                    detail="Reason given as a list of multiple strings; joining would alter "
                           "phrasing, so this violation is dropped instead of merged.",
                )
            return None

        boxes = normalize_boxes(
            fixed.get("bounding_box", []), field=f"{rule_key}.bounding_box", tracker=tracker
        )
        return {"reason": str(reason), "bounding_box": boxes}

    if tracker:
        tracker.log(
            "violation_dropped_unrecognized_type", field=rule_key,
            detail=f"Value of type {type(v).__name__} is not a recognizable violation format; dropped.",
        )
    return None


# =============================================================================
# SECTION 6 — ORCHESTRATION: full structural fix for one parsed prediction
# =============================================================================

def fix_prediction_structure(
    parsed: dict, tracker: Optional[ChangeTracker] = None
) -> Dict[str, Any]:
    """Applies ALL structural/key/type fixes to one parsed prediction dict.
    Never touches numeric values or text content — only reshapes/retypes/
    renames so the dict can satisfy the UnifiedOutput schema."""
    fixed = dict(parsed)

    fixed = normalize_top_level_keys(fixed, tracker=tracker)
    fixed = reconstruct_violations_list(fixed, tracker=tracker)
    fixed = reconstruct_flattened_violations(fixed, tracker=tracker)

    # Caption: only type-coerce (list of sentences -> joined string),
    # never invent missing content.
    if "caption" in fixed and isinstance(fixed["caption"], list):
        if tracker:
            tracker.log(
                "caption_list_joined", field="caption",
                detail=f"{len(fixed['caption'])} sentence(s) joined into a single caption string.",
            )
        fixed["caption"] = " ".join(str(x) for x in fixed["caption"] if x)

    for cls in ("excavator", "rebar", "worker_with_white_hard_hat"):
        fixed[cls] = normalize_boxes(fixed.get(cls, []), field=cls, tracker=tracker)

    for i in range(1, 5):
        key = f"rule_{i}_violation"
        fixed[key] = normalize_violation_value(fixed.get(key), rule_key=key, tracker=tracker)

    return fixed


def _check_duplicate_boxes(
    fixed: dict, tracker: ChangeTracker, threshold: int = 3
) -> None:
    """DIAGNOSTIC ONLY — never modifies anything. Flags when the exact same
    box appears more than `threshold` times within one field of one record:
    a strong signal of a model generation loop (e.g. truncation-induced
    repetition), even though the JSON/schema itself may be perfectly valid.
    """
    def _scan(boxes: List[List[float]], field: str) -> None:
        if not boxes:
            return
        counts = Counter(tuple(b) for b in boxes)
        for box, count in counts.items():
            if count > threshold:
                tracker.warn(
                    "duplicate_box_detected", field=field,
                    detail=f"Box {list(box)} repeated {count} times "
                           f"(possible generation loop/truncation artifact).",
                )

    for cls in ("excavator", "rebar", "worker_with_white_hard_hat"):
        _scan(fixed.get(cls) or [], cls)

    for i in range(1, 5):
        v = fixed.get(f"rule_{i}_violation")
        if isinstance(v, dict):
            _scan(v.get("bounding_box") or [], f"rule_{i}_violation.bounding_box")


def repair_and_validate(raw_str: str, duplicate_box_threshold: int = 3) -> Dict[str, Any]:
    """Full pipeline for one record's raw model output.

    Returns:
        {
          "status": "valid_raw" | "fixed_valid" | "invalid_json" | "invalid_schema",
          "original_parsed": dict or None,
          "fixed_parsed": dict or None,
          "validated": UnifiedOutput or None,
          "error": str or None,
          "changes": List[Dict[str, str]],   # every actual fix applied
          "warnings": List[Dict[str, str]],  # diagnostic-only observations
        }

    - "valid_raw"      : parsed cleanly AND matched schema with no fixing needed
    - "fixed_valid"     : needed structural repair, but now matches schema
    - "invalid_json"    : could not extract/parse any JSON object at all
    - "invalid_schema"  : parsed fine, repaired as much as safely possible,
                          still doesn't match schema (genuinely missing/
                          ambiguous content — NOT fabricated)
    """
    tracker = ChangeTracker()

    parsed = robust_parse(raw_str, tracker=tracker)
    if parsed is None:
        return {
            "status": "invalid_json",
            "original_parsed": None,
            "fixed_parsed": None,
            "validated": None,
            "error": "Could not extract/parse any JSON object from raw output.",
            "changes": tracker.changes,
            "warnings": tracker.warnings,
        }

    try:
        validated_raw = UnifiedOutput(**parsed)
        _check_duplicate_boxes(parsed, tracker, threshold=duplicate_box_threshold)
        return {
            "status": "valid_raw",
            "original_parsed": parsed,
            "fixed_parsed": parsed,
            "validated": validated_raw,
            "error": None,
            "changes": tracker.changes,
            "warnings": tracker.warnings,
        }
    except Exception:
        pass

    fixed = fix_prediction_structure(parsed, tracker=tracker)
    _check_duplicate_boxes(fixed, tracker, threshold=duplicate_box_threshold)
    try:
        validated_fixed = UnifiedOutput(**fixed)
        return {
            "status": "fixed_valid",
            "original_parsed": parsed,
            "fixed_parsed": fixed,
            "validated": validated_fixed,
            "error": None,
            "changes": tracker.changes,
            "warnings": tracker.warnings,
        }
    except Exception as e:
        return {
            "status": "invalid_schema",
            "original_parsed": parsed,
            "fixed_parsed": fixed,
            "validated": None,
            "error": str(e),
            "changes": tracker.changes,
            "warnings": tracker.warnings,
        }


# =============================================================================
# SECTION 7 — BATCH DRIVER (jsonl in -> repaired jsonl + report + manifest + broken log)
# =============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_jsonl(
    input_path: str,
    output_path: str,
    report_path: str,
    broken_path: str,
    manifest_path: str,
    duplicate_box_threshold: int = 3,
) -> Dict[str, Any]:
    records = load_jsonl(input_path)
    fixed_records = []
    status_list = []
    broken_records = []
    manifest_records = []

    change_type_counter = Counter()          # total occurrences, all records
    change_type_record_counter = Counter()   # distinct records affected
    warning_type_counter = Counter()
    warning_type_record_counter = Counter()

    for r in records:
        image_id = r.get("image_id", "")
        raw = r.get("raw_output", "")
        result = repair_and_validate(raw, duplicate_box_threshold=duplicate_box_threshold)

        # --- main repaired jsonl: ORIGINAL schema preserved, additive-only ---
        out_record = dict(r)
        if result["status"] == "fixed_valid":
            out_record["raw_output"] = json.dumps(result["fixed_parsed"], ensure_ascii=False)
            out_record["original_raw_output"] = raw
        out_record["repair_status"] = result["status"]
        fixed_records.append(out_record)

        status_list.append(result["status"])

        # --- aggregate change/warning tallies ---
        changes = result["changes"]
        warnings = result["warnings"]

        seen_change_types = set()
        for c in changes:
            change_type_counter[c["type"]] += 1
            seen_change_types.add(c["type"])
        for t in seen_change_types:
            change_type_record_counter[t] += 1

        seen_warning_types = set()
        for w in warnings:
            warning_type_counter[w["type"]] += 1
            seen_warning_types.add(w["type"])
        for t in seen_warning_types:
            warning_type_record_counter[t] += 1

        # --- per-record manifest entry (EVERY record, even fully clean ones) ---
        manifest_records.append({
            "image_id": image_id,
            "status": result["status"],
            "changes": changes,
            "warnings": warnings,
        })

        if result["status"] in ("invalid_json", "invalid_schema"):
            broken_records.append({
                "image_id": image_id,
                "status": result["status"],
                "error": result["error"],
                "raw_output": raw,
                "parsed": result["original_parsed"],
                "changes_attempted": changes,
                "warnings": warnings,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in fixed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    status_counts = Counter(status_list)
    total = len(status_list)

    change_type_summary = {
        t: {"total_occurrences": change_type_counter[t], "records_affected": change_type_record_counter[t]}
        for t in change_type_counter
    }
    warning_type_summary = {
        t: {"total_occurrences": warning_type_counter[t], "records_affected": warning_type_record_counter[t]}
        for t in warning_type_counter
    }

    report = {
        "total_records": total,
        "status_counts": dict(status_counts),
        "status_percentages": (
            {k: round(v / total * 100, 2) for k, v in status_counts.items()} if total else {}
        ),
        "change_type_summary": change_type_summary,
        "warning_type_summary": warning_type_summary,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(broken_path, "w", encoding="utf-8") as f:
        json.dump(broken_records, f, indent=2, ensure_ascii=False)

    return report


# =============================================================================
# SECTION 8 — CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Repair structural/schema issues in VLM JSONL predictions "
                    "WITHOUT changing any actual content, and track every fix applied."
    )
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", default=None,
                         help="Path for fixed predictions.jsonl (default: alongside input)")
    parser.add_argument("--report", default=None,
                         help="Path for repair_report.json (default: alongside input)")
    parser.add_argument("--broken", default=None,
                         help="Path for still_broken.json (default: alongside input)")
    parser.add_argument("--manifest", default=None,
                         help="Path for change_manifest.json (default: alongside input)")
    parser.add_argument("--duplicate_box_threshold", type=int, default=3,
                         help="Flag a box as 'duplicate_box_detected' if it repeats more "
                              "than this many times within one field of one record.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "predictions_repaired.jsonl"
    report_path = Path(args.report) if args.report else input_path.parent / "repair_report.json"
    broken_path = Path(args.broken) if args.broken else input_path.parent / "still_broken.json"
    manifest_path = Path(args.manifest) if args.manifest else input_path.parent / "change_manifest.json"

    print(f"Loading predictions from: {input_path}")
    report = process_jsonl(
        str(input_path), str(output_path), str(report_path), str(broken_path), str(manifest_path),
        duplicate_box_threshold=args.duplicate_box_threshold,
    )

    print("\n=== REPAIR STATUS SUMMARY ===")
    for status, count in report["status_counts"].items():
        pct = report["status_percentages"][status]
        print(f"  {status:<20}: {count:>6}  ({pct:>5.2f}%)")

    print("\n=== STRUCTURAL FIXES APPLIED (by total occurrences) ===")
    sorted_changes = sorted(
        report["change_type_summary"].items(),
        key=lambda kv: kv[1]["total_occurrences"], reverse=True,
    )
    if sorted_changes:
        for change_type, stats in sorted_changes:
            print(f"  {change_type:<40}: {stats['total_occurrences']:>6} occurrence(s) "
                  f"across {stats['records_affected']:>5} record(s)")
    else:
        print("  (none — no structural fixes were needed)")

    print("\n=== DIAGNOSTIC WARNINGS (not fixes — for manual review) ===")
    sorted_warnings = sorted(
        report["warning_type_summary"].items(),
        key=lambda kv: kv[1]["total_occurrences"], reverse=True,
    )
    if sorted_warnings:
        for warning_type, stats in sorted_warnings:
            print(f"  {warning_type:<40}: {stats['total_occurrences']:>6} occurrence(s) "
                  f"across {stats['records_affected']:>5} record(s)")
    else:
        print("  (none)")

    print(f"\nFixed predictions saved to:        {output_path}")
    print(f"Report saved to:                   {report_path}")
    print(f"Per-record change manifest saved to: {manifest_path}")
    print(f"Still-broken records saved to:     {broken_path}")