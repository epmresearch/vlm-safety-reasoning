"""Diagnostics for the ``<think>`` block emitted by the ``violations_think`` task.

WHY THIS FILE EXISTS. The block is invisible to every other metric family. Nothing
rewards it during GRPO, nothing parses it during evaluation (``strip_fences`` steps over
it to reach the JSON), and structural repair treats it as preamble. So without these
keys a run could emit a degenerate, truncated or self-contradicting block for all 3,004
images and every reported number would look completely normal.

Three things are worth measuring, in increasing order of interest:

1. **Did it emit a block at all, and did it close it?** An unclosed block is what a
   truncated completion looks like, and truncation is indistinguishable from a bad model
   in the reward signal (every component returns exactly 0.0 for both).
2. **Is the block well formed?** Five lines, four rule lines, in order.
3. **Do the block's verdicts agree with the JSON below it?** This is the interesting
   one. A model that writes ``rule_2: no`` and then reports a rule_2 violation in the
   JSON has produced reasoning that does not describe its own answer -- which is a
   different and more troubling failure than simply being wrong.

Deliberately NOT here: any judgement of whether the block's reasoning is *good*. The
reason text in the JSON is already scored by ``reward_reasoning``, the
``reasoning_text_similarity_*`` family and the LLM judge; scoring the copy in the block
as well would double-count the same sentence.

Everything is computed from what the MODEL emitted, which on the repaired file means
``original_raw_output`` when present -- see ``experiments/run_evaluation.py``. The parser
is deliberately tolerant (``core/think_format.py`` explains why); a deviation here is
the measurement, not a crash.
"""
from typing import Any, Dict, List, Optional, Sequence

from core.constants import RULES
from core.logging import get_logger
from core.think_format import (
    THINK_OPEN,
    extract_think_block,
    parse_think_body,
    problem_bucket,
    verdicts_from_violations,
    violations_from_row,
)

logger = get_logger(__name__)


def task_expects_think_block(task: str) -> bool:
    """Does this task's prompt ask for a ``<think>`` block?

    Derived from the prompt itself rather than hardcoded against a task name. Two
    reasons: ``core/tasks.py`` is explicit that no conditional anywhere should compare a
    task name to a literal string, and "were we asking for a block?" is the actual
    question -- which also means a future think-style arm is covered with no edit here.

    Gating on this (rather than on whether blocks happen to appear in the output) is
    what makes an all-zero result meaningful: for ``violations_think`` the keys are
    always present, so ``think_block_present_rate: 0.0`` reads as "the model stopped
    emitting blocks", not as "the metric never ran". For every other task the keys are
    absent entirely, never zero -- the same convention the LLM judge follows.
    """
    from data.prompt_templates import get_prompt_for_task
    try:
        return THINK_OPEN in get_prompt_for_task(task)
    except Exception as exc:                       # unregistered prompt_key, missing YAML
        logger.warning(f"Could not resolve the prompt for task {task!r}: {exc}")
        return False


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. No numpy dependency -- this runs inside evaluation."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def compute_think_metrics(
    model_texts: Sequence[Any],
    parsed_preds: Sequence[Optional[Dict[str, Any]]],
    prefix: str = "think_",
) -> Dict[str, Any]:
    """Block diagnostics for one run.

    Args:
        model_texts: what the model actually emitted, per record, in evaluation order.
        parsed_preds: the parsed prediction dicts in the same order, ``None`` where the
            JSON failed to parse or validate. Used only for the verdict-agreement
            comparison, which is skipped for a record whose JSON is unusable -- there is
            nothing to agree *with*.
        prefix: metric key prefix.

    Returns:
        A flat dict of metric keys. Rates whose denominator is zero are reported as
        ``0.0`` with the matching ``*_count`` key at 0, so a reader can always tell an
        empty denominator from a real zero.
    """
    n = len(model_texts)
    if n != len(parsed_preds):
        raise ValueError(
            f"model_texts ({n}) and parsed_preds ({len(parsed_preds)}) must align"
        )

    present = closed = wellformed = 0
    word_counts: List[int] = []
    problem_counts: List[int] = []
    problem_tally: Dict[str, int] = {}

    # Agreement is counted per (record, rule) slot, not per record: a block that gets
    # three rules right and one wrong should not be scored the same as one that gets all
    # four wrong.
    agree_hits = 0
    agree_total = 0
    per_rule_hits = {r: 0 for r in RULES}
    per_rule_total = {r: 0 for r in RULES}

    for text, pred in zip(model_texts, parsed_preds):
        body, was_closed = extract_think_block(text)
        if body is None:
            continue
        present += 1
        if was_closed:
            closed += 1

        parsed = parse_think_body(body)
        if parsed.wellformed:
            wellformed += 1
        problem_counts.append(len(parsed.problems))
        for p in parsed.problems:
            # problem_bucket, not p.split(":")[0]: the latter collapses every per-rule
            # defect into the rule's NAME, so the tally reports which rule rather than
            # what went wrong.
            bucket = problem_bucket(p)
            problem_tally[bucket] = problem_tally.get(bucket, 0) + 1
        word_counts.append(len(body.split()))

        if pred is None:
            continue
        json_verdicts = verdicts_from_violations(violations_from_row(pred))
        for rule in RULES:
            block_verdict = parsed.verdicts.get(rule)
            if block_verdict is None:       # unreadable line; nothing to compare
                continue
            per_rule_total[rule] += 1
            agree_total += 1
            if block_verdict == json_verdicts[rule]:
                per_rule_hits[rule] += 1
                agree_hits += 1

    def rate(hits: int, total: int) -> float:
        return float(hits) / total if total else 0.0

    words_sorted = sorted(word_counts)
    metrics: Dict[str, Any] = {
        f"{prefix}block_present_count": present,
        f"{prefix}block_present_rate": rate(present, n),
        # Denominator is the blocks that EXIST, not all records: mixing "never emitted
        # one" into "emitted a malformed one" would hide which failure is happening.
        f"{prefix}block_closed_rate": rate(closed, present),
        f"{prefix}block_wellformed_rate": rate(wellformed, present),
        f"{prefix}block_problem_count_mean": (
            sum(problem_counts) / len(problem_counts) if problem_counts else 0.0
        ),
        f"{prefix}block_words_mean": (
            sum(word_counts) / len(word_counts) if word_counts else 0.0
        ),
        f"{prefix}block_words_p50": _percentile(words_sorted, 0.50),
        f"{prefix}block_words_p95": _percentile(words_sorted, 0.95),
        f"{prefix}block_words_max": float(max(word_counts)) if word_counts else 0.0,
        f"{prefix}verdict_json_agreement_rate": rate(agree_hits, agree_total),
        f"{prefix}verdict_json_comparable_count": agree_total,
        f"{prefix}total_samples_count": n,
    }
    for rule in RULES:
        metrics[f"{prefix}verdict_json_agreement_rate_{rule}"] = rate(
            per_rule_hits[rule], per_rule_total[rule]
        )
        metrics[f"{prefix}verdict_json_comparable_count_{rule}"] = per_rule_total[rule]

    # One string key, mirroring the precedent set by reasoning_llm_judge_model_id.
    # Non-numeric values are skipped by the index's metric-row filter, so this reaches
    # metrics.json for a human without polluting the comparison tables.
    metrics[f"{prefix}top_problem"] = (
        max(problem_tally.items(), key=lambda kv: kv[1])[0] if problem_tally else ""
    )
    return metrics
