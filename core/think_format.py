"""The ``<think>``-block wire format for the ``violations_think`` task.

ONE definition of the format, three consumers with DIFFERENT strictness -- and the
strictness difference is the whole point:

  * ``data/preprocessor.py::_build_violations_think_target_json`` -- STRICT. Builds the
    SFT target from a dataset row's ``thinking`` column and refuses to emit a block
    whose verdicts disagree with that row's own labels. Such a target trains an
    outright contradiction ("the trench is protected, therefore rule_3 is violated"),
    and no reward, metric or repair stage downstream can see it -- so it has to fail
    before any GPU time is spent.
  * ``scripts/validate_think_dataset.py`` -- STRICT, but over a whole dataset: reports
    every offending row instead of raising on the first one.
  * ``evaluation/metrics_think.py`` -- TOLERANT. Reads what the MODEL actually emitted
    at inference, where a deviation is the measurement rather than a bug. Raising
    there would throw away a 3,004-image evaluation.

``parse_think_body`` therefore NEVER raises. It returns everything it found wrong in
``problems`` and lets each caller decide the policy.

THE FORMAT -- the *body*, i.e. what sits between the tags::

    first       the image caption, verbatim
    then        one line per rule, always all four, always rule_1 -> rule_4, on EVERY
                image including safe ones:
                    violated            rule_N: <reason, trailing period stripped> -> yes (<n>)
                    violated, no boxes  rule_N: <reason> -> yes
                    not violated        rule_N: no

Four rule lines on every image, safe ones included: omitting them on safe images would
leak the answer through the block's mere *presence*. The order is always rule_1 -> rule_4
and never sorted by violation status, which would leak it through *position*.

The CANONICAL body (what ``build_think_body`` emits, and what the dataset must hold) puts
the caption on one line, so it is exactly five lines. ``parse_think_body`` is deliberately
looser: it reads the caption as everything before the ``rule_1:`` line, however many lines
that is. The description is free prose that nothing scores, so a model wrapping it over
two lines is harmless -- the JSON still parses and no reported metric moves. Pinning the
line count there only turned one cosmetic newline into five cascading failures.

Nothing in the block is authored by us. Every token is ground truth -- GT caption, GT
reason, GT label, GT box count. That property is what makes the arm defensible, and it
is why a templated negative line ("no scaffold described") is deliberately NOT part of
this format: a regex-derived clause is invented text, and on a guarded trench it would
be flatly false.

Boxes never appear in the block, only their count. Coordinates ARE the answer; emitting
them twice doubles the exposure to the malformed-box-array failure mode that dominates
this project's parse failures, for no derivation benefit.
"""
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from core.constants import RULES

# The tags are added by render_think_block(), NOT stored in the dataset's `thinking`
# column. Keeping them out of the data means a tag change needs no data rebuild, and a
# half-written tag cannot enter the dataset silently.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The dataset column holding the block body. Part of the DATASET CONTRACT, defined
# here and nowhere else.
#
# Deliberately NOT a task-YAML key. The SFT target builders in
# data/preprocessor.py::_TARGET_BUILDERS are called as `builder(raw)` and never see a
# config, so a `think_field:` key in the task YAML could not reach the code that needs
# it -- it would read as configuration while being inert, which is precisely the
# ghost-variable class of bug this repo has already paid for five times (see
# CLAUDE.md's ghost-variable table). A future arm wanting a different column registers
# its own task with its own builder instead.
THINK_FIELD = "thinking"

_NOT_VIOLATED_LINE = "no"

# Tolerant on purpose (case, spacing, an optional count) because this same parser reads
# raw model output. Anchored at the end so a reason that happens to contain "-> yes"
# cannot be mistaken for the verdict.
_YES_LINE_RE = re.compile(
    r"^(?P<reason>.*?)\s*->\s*yes(?:\s*\(\s*(?P<n>\d+)\s*\))?\s*$",
    re.IGNORECASE,
)


def _normalize_newlines(text: str) -> str:
    """CRLF/CR -> LF, then strip leading/trailing blank lines.

    Not cosmetic: the dataset may well be baked on Windows, and a CRLF body would
    otherwise fail every line check with a completely unhelpful message.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def is_violation_asserted(v: Any) -> bool:
    """Does this ``rule_N_violation`` value assert a violation?

    A deliberate MIRROR of ``rewards/reward_utils.py::_is_violation_present``, not an
    import: that module pulls in ``torch`` at module scope, and ``core/`` must stay
    importable without it (the preprocessor and the dataset validator both run on a CPU
    login node). The two are pinned to agree on a table of shapes by
    ``tests/test_core/test_think_format.py`` -- if that test fails, this function is
    wrong, not the test.

    The rule follows the prompt contract literally: the prompt says "If NOT violated,
    output null", so only ``None`` (or an absent key) means safe. A bare ``{}`` is the
    one exception -- no keys, therefore no assertion.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "none", "null", "n/a", "")
    return bool(v)


def canonical_reason(text: Any) -> str:
    """A GT reason as it appears INSIDE the block: stripped, trailing period removed.

    The period is stripped only in the block, so ``... hard hat -> yes (1)`` reads
    cleanly. It stays intact in the JSON payload below, because ``reward_reasoning``
    and the LLM judge score against the exact GT string.
    """
    return str(text or "").strip().rstrip(".").strip()


def problem_bucket(problem: str) -> str:
    """A defect CLASS from a problem message, for tallying.

    Strips a leading ``rule_N: `` first. Bucketing on the text before the first colon
    alone collapses every per-rule defect into the rule's own NAME -- so a tally would
    report *which rule* was involved instead of *what was wrong*, and
    ``think_top_problem`` would read ``"rule_1"``. Whether the defect was an empty
    reason or an unreadable verdict is the part worth counting.

    Shared by ``evaluation/metrics_think.py`` and ``scripts/validate_think_dataset.py``
    so the two tallies cannot disagree.
    """
    s = str(problem)
    for rule in RULES:
        prefix = f"{rule}: "
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.split(":")[0].strip()[:70]


def _reject_unusable_text(text: Any, label: str) -> str:
    """Returns ``text`` stripped, or raises if it cannot appear inside a block.

    The four rejected shapes are exactly the ones ``think_row_problems`` refuses, so
    ``build_think_body`` can never emit a body its own validator would reject:

      * blank              -- there would be no line to write
      * a newline          -- the CANONICAL body is one line per element, and a reason
                              carrying a newline would split into two lines that no
                              longer parse as one verdict. (A newline in the caption is
                              harmless at INFERENCE -- parse_think_body accepts a
                              multi-line description -- but the baked dataset still has
                              to be deterministic.)
      * a brace            -- trained in, it teaches the model to emit ``{`` inside the
                              block, where structural_repair's no-fence fallback
                              (``_extract_outermost_braces``) could pick the block up
                              instead of the real JSON payload
      * a code fence       -- it would break fence extraction outright
    """
    s = str(text or "").strip()
    if not s:
        raise ValueError(f"{label} is blank; the think block needs it")
    if "\n" in s:
        raise ValueError(f"{label} contains a newline; each block line must be one line")
    if "{" in s or "}" in s:
        raise ValueError(f"{label} contains a brace, which cannot appear inside the block")
    if "```" in s:
        raise ValueError(f"{label} contains a code fence, which cannot appear inside the block")
    return s


def build_think_body(caption: Any, violations: Dict[str, Any]) -> str:
    """The canonical block body for one ground-truth row.

    This is the executable definition of the format. ``scripts/validate_think_dataset.py``
    uses it to check a pre-baked ``thinking`` column byte-for-byte, and any future data
    prep that wants to GENERATE that column should import this rather than re-implement
    it -- one implementation of the format in the repo, by construction.

    Args:
        caption: the row's ``image_caption``.
        violations: ``{"rule_1": None | {"bounding_box": [...], "reason": str}, ...}``.
            Box scale is irrelevant here; only the count is used.

    Raises:
        ValueError: if the caption is blank, or if a rule is asserted with no reason.
            A reason-less violation cannot produce a block line without inventing text,
            and it already scores 0 on ``reward_reasoning``, so it is rejected at
            build time rather than papered over. Drop or complete those rows in data
            prep.
    """
    cap = _reject_unusable_text(caption, "image_caption")

    lines: List[str] = [cap]
    for rule in RULES:
        v = violations.get(rule)
        if not is_violation_asserted(v):
            lines.append(f"{rule}: {_NOT_VIOLATED_LINE}")
            continue
        reason = canonical_reason(v.get("reason") if isinstance(v, dict) else "")
        if not reason:
            raise ValueError(
                f"{rule} is asserted but carries no reason; cannot build a block line "
                "without inventing text. Drop the row or supply a reason in data prep."
            )
        # The SAME guards as the caption, and for the same reason: this function is the
        # one data prep is told to import, so it must never produce a body that
        # think_row_problems then rejects. Without this, a reason carrying a newline
        # yields a 6-line body whose failure is reported as a positional line-count
        # complaint -- true, but useless for finding the offending reason.
        _reject_unusable_text(reason, f"{rule}'s reason")
        boxes = (v.get("bounding_box") or []) if isinstance(v, dict) else []
        suffix = f" -> yes ({len(boxes)})" if boxes else " -> yes"
        lines.append(f"{rule}: {reason}{suffix}")
    return "\n".join(lines)


def render_think_block(body: str) -> str:
    """Wraps a body in the tags, with the trailing newline the target needs.

    ``<think>\\n{body}\\n</think>\\n`` -- so the fenced JSON that follows starts on its
    own line.
    """
    return f"{THINK_OPEN}\n{_normalize_newlines(body)}\n{THINK_CLOSE}\n"


@dataclass(frozen=True)
class ParsedThink:
    """The result of reading a block body. ``problems`` empty means canonical shape."""

    caption: str
    verdicts: Dict[str, Optional[bool]]   # rule -> True/False, None if unreadable
    counts: Dict[str, Optional[int]]      # rule -> box count from "(n)", None if absent
    reasons: Dict[str, str]               # rule -> reason text, asserted rules only
    problems: Tuple[str, ...]

    @property
    def wellformed(self) -> bool:
        return not self.problems


def parse_think_body(body: Any) -> ParsedThink:
    """Reads a block body. NEVER raises -- every deviation lands in ``problems``.

    Tolerant by design, because this same parser reads raw model output at inference
    where deviation is the measurement. Callers wanting strictness check ``problems``
    (or, when they have ground truth, compare against ``build_think_body``).
    """
    problems: List[str] = []
    verdicts: Dict[str, Optional[bool]] = {r: None for r in RULES}
    counts: Dict[str, Optional[int]] = {r: None for r in RULES}
    reasons: Dict[str, str] = {}

    if not isinstance(body, str) or not body.strip():
        return ParsedThink("", verdicts, counts, reasons, ("body is empty or not a string",))

    lines = _normalize_newlines(body).split("\n")

    # The description may span any number of lines. It is free prose that nothing scores,
    # so a model wrapping it over two lines is harmless -- the JSON still parses and every
    # reported metric is unaffected. An earlier version fixed the block at exactly five
    # lines and matched the rule lines at indices 1-4, which turned one cosmetic newline
    # into five cascading failures and made the block's own health readout unreadable.
    #
    # The rule lines are still matched CONSECUTIVELY from rule_1, so order remains
    # enforced: sorting by violation status would leak the answer through position.
    start = next((i for i, l in enumerate(lines) if l.strip().startswith(f"{RULES[0]}:")), None)
    if start is None:
        problems.append(f"no {RULES[0]} line found")
        return ParsedThink(" ".join(l.strip() for l in lines).strip(),
                           verdicts, counts, reasons, tuple(problems))

    caption = " ".join(l.strip() for l in lines[:start]).strip()
    if not caption:
        problems.append("no description before the rule lines")

    for k, rule in enumerate(RULES):
        i = start + k
        if i >= len(lines):
            problems.append(f"{rule}: line missing")
            continue
        line = lines[i].strip()
        prefix = f"{rule}:"
        if not line.startswith(prefix):
            problems.append(f"rule lines out of order: expected {prefix!r}, got {line[:40]!r}")
            continue
        payload = line[len(prefix):].strip()
        if payload.lower() == _NOT_VIOLATED_LINE:
            verdicts[rule] = False
            continue
        m = _YES_LINE_RE.match(payload)
        if m is None:
            problems.append(f"{rule}: payload is neither {_NOT_VIOLATED_LINE!r} nor '... -> yes (n)': {payload[:60]!r}")
            continue
        verdicts[rule] = True
        reason = (m.group("reason") or "").strip()
        reasons[rule] = reason
        if not reason:
            problems.append(f"{rule}: asserted with an empty reason")
        if m.group("n") is not None:
            counts[rule] = int(m.group("n"))

    trailing = [l for l in lines[start + len(RULES):] if l.strip()]
    if trailing:
        problems.append(f"extra content after {RULES[-1]}: {trailing[0][:40]!r}")

    return ParsedThink(caption, verdicts, counts, reasons, tuple(problems))


def extract_think_block(text: Any) -> Tuple[Optional[str], bool]:
    """Pulls the block body out of a raw completion.

    Returns ``(body, closed)``:
      * ``(None, False)``  -- no opening tag at all.
      * ``(body, True)``   -- a properly closed ``<think>...</think>``.
      * ``(body, False)``  -- an opening tag with no closing tag, which is what a
        truncated completion looks like. The body is then everything up to the JSON
        fence if one follows, else the rest of the text. Reported rather than dropped,
        because "present but never closed" is a distinct and diagnostic failure.

    Only the FIRST opening tag is honoured: a second one is degenerate output, and the
    block-length and agreement metrics should describe the block the model actually
    committed to.
    """
    if not isinstance(text, str):
        return None, False
    i = text.find(THINK_OPEN)
    if i < 0:
        return None, False
    rest = text[i + len(THINK_OPEN):]
    j = rest.find(THINK_CLOSE)
    if j >= 0:
        return _normalize_newlines(rest[:j]), True
    fence = rest.find("```")
    body = rest[:fence] if fence >= 0 else rest
    return _normalize_newlines(body), False


def verdicts_from_violations(violations: Dict[str, Any]) -> Dict[str, bool]:
    """``{rule: asserted}`` for a mapping keyed by BARE rule name (``rule_1``).

    For a dataset row or a parsed prediction -- both of which key their violations
    ``rule_1_violation`` -- go through :func:`violations_from_row` first.
    """
    return {r: is_violation_asserted(violations.get(r)) for r in RULES}


# Dataset rows and parsed predictions both key their violations `rule_N_violation`;
# build_think_body and verdicts_from_violations take the bare `rule_N`. Conflating the
# two silently reads every violation as absent, which would report every genuinely
# violated row as a block/label contradiction -- so the conversion is one named
# function rather than an inline f-string at each call site.
VIOLATION_KEY_SUFFIX = "_violation"


def violations_from_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """``{rule_1: <value>, ...}`` from a row's/prediction's ``rule_N_violation`` keys."""
    return {r: raw.get(f"{r}{VIOLATION_KEY_SUFFIX}") for r in RULES}


def think_row_problems(raw: Dict[str, Any]) -> List[str]:
    """Everything wrong with one dataset row's ``thinking`` column, as plain messages.

    THE safeguard for this arm. The ``thinking`` column is pre-baked, so the dataset
    *is* the format: a bad row is frozen in, and by the time training starts nothing
    downstream can detect it. A block whose verdict contradicts its own label teaches
    the model to invert evidence, and every reward, metric and repair stage would
    report that run as normal.

    One implementation, two callers with different policies:
      * ``data/preprocessor.py::_build_violations_think_target_json`` raises on the
        first bad row, before any GPU time is spent.
      * ``scripts/validate_think_dataset.py`` collects them all and reports.

    Returns an empty list for a good row.

    The REASON comparison runs through ``canonical_reason`` on both sides, so a cosmetic
    difference (a trailing period data prep did not strip) passes while a genuinely
    different sentence fails.

    The CAPTION comparison is deliberately VERBATIM (a plain ``!=`` on the stripped
    strings). The block's first line is meant to be ``image_caption`` copied, not
    paraphrased or re-punctuated, and unlike a reason there is no formatting convention
    to normalise away -- so any difference at all means the two fields disagree about
    what the image shows.

    Exact byte-identity of the whole body against ``build_think_body`` is a separate,
    opt-in check: the validator's ``--strict``.
    """
    problems: List[str] = []

    body = raw.get(THINK_FIELD)
    if body is None:
        return [f"{THINK_FIELD!r} column is missing"]
    if not isinstance(body, str):
        return [f"{THINK_FIELD!r} is {type(body).__name__}, expected str"]
    if not body.strip():
        return [f"{THINK_FIELD!r} is blank"]

    # Trained-in braces would teach the model to emit them inside the block, where
    # structural_repair's no-fence fallback (_extract_outermost_braces) could then pick
    # the block up instead of the real JSON payload.
    if "{" in body or "}" in body:
        problems.append("contains a brace, which can divert structural repair's brace fallback")
    if "```" in body:
        problems.append("contains a code fence, which would break fence extraction")

    parsed = parse_think_body(body)
    problems.extend(parsed.problems)

    expected_caption = str(raw.get("image_caption") or "").strip()
    if not expected_caption:
        problems.append("image_caption is blank")
    elif parsed.caption and parsed.caption != expected_caption:
        problems.append("caption line does not match image_caption verbatim")

    violations = violations_from_row(raw)
    expected = verdicts_from_violations(violations)
    for rule in RULES:
        got = parsed.verdicts.get(rule)
        if got is None:
            continue                      # already reported by parse_think_body
        if got != expected[rule]:
            problems.append(
                f"{rule}: block says {'violated' if got else 'not violated'} but the "
                f"label says {'violated' if expected[rule] else 'not violated'}"
            )
            continue
        if not expected[rule]:
            continue
        v = violations[rule] or {}
        boxes = (v.get("bounding_box") or []) if isinstance(v, dict) else []
        want_count = len(boxes) if boxes else None
        if parsed.counts.get(rule) != want_count:
            problems.append(
                f"{rule}: block box count {parsed.counts.get(rule)!r} != "
                f"{want_count!r} from the label"
            )
        want_reason = canonical_reason(v.get("reason") if isinstance(v, dict) else "")
        if not want_reason:
            problems.append(f"{rule}: label asserts a violation with no reason")
        elif canonical_reason(parsed.reasons.get(rule, "")) != want_reason:
            problems.append(f"{rule}: block reason does not match the label's reason")

    return problems
