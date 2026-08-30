"""
Single source of truth for how a task name maps to the short prefix used in
variant/result-folder naming across the pipeline (e.g. "violations_only" -> "vo").

Adding a new task (e.g. "object_only", "caption_only") to the whole pipeline —
SLURM orchestration, inference/eval result folders, comparison tables, plots —
means adding ONE line here. No other file needs to know about it.
"""

TASK_PREFIXES = {
    "unified": "unified",
    "violations_only": "vo",
    # "object_only": "oo",
    # "caption_only": "co",
}


def task_prefix(task: str) -> str:
    """Short prefix for a task name, e.g. 'violations_only' -> 'vo'.

    Falls back to the task name itself for anything not yet registered, so a
    brand-new task still works end-to-end (just with a longer, unabbreviated
    prefix) before anyone gets around to adding it to TASK_PREFIXES.
    """
    return TASK_PREFIXES.get(task, task)


def variant_name(task: str, phase: str, tier: str, version: str) -> str:
    """Build a standard '<prefix>-<phase>-<tier>-<version>' variant name.

    Matches the naming convention introduced for the violations_only pipeline
    (scripts/submit_vo_pipeline.py). The unified pipeline's legacy result
    folders predate this convention (e.g. 'baseline_8b_v4', underscored, no
    task prefix) and are handled by their own lookup logic in
    experiments/compare_results.py and experiments/plot_metrics.py — this
    helper is for new, consistently-named pipeline output only.
    """
    return f"{task_prefix(task)}-{phase}-{tier}-{version}"


def merged_checkpoint_name(task: str, tier: str, version: str) -> str:
    """Build the name of the merged-SFT-into-base checkpoint directory used
    as the GRPO KL reference model: 'merged-<prefix>-sft-<tier>-<version>'.

    Task-namespaced so two pipelines (e.g. unified and violations_only)
    bumped to the same --version tag never write into the same checkpoint
    directory. Used by scripts/submit_vo_pipeline.py,
    scripts/submit_unified_pipeline.py, and experiments/run_inference.py's
    GRPO base-model auto-detect — all three must agree on this exact format.
    """
    return f"merged-{task_prefix(task)}-sft-{tier}-{version}"
