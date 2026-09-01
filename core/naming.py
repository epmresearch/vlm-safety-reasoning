"""
Single source of truth for how a task name maps to the short prefix used in
variant/result-folder naming across the pipeline (e.g. "violations_only" -> "vo").

The prefixes themselves live in core/tasks.py::TASK_REGISTRY, which is the one place
a new task is registered. This module only builds names out of them, so every layer
that has to agree on a string -- the SLURM submitter, the merge step, GRPO's KL
reference lookup in run_inference.py, the comparison tables, the plots -- agrees by
construction.

Every name produced here is namespaced by the task prefix. That is what makes it safe
to run all four pipelines concurrently on ARC at the same --version and tier: nothing
coordinates them at runtime, so isolation comes entirely from no two tasks ever
generating the same writable path. tests/test_core/test_name_isolation.py asserts it.
"""
from core.tasks import all_task_prefixes, get_task_spec

# Kept as a module-level dict for backward compatibility with existing importers.
# Derived from the registry -- do not edit here, add a TaskSpec in core/tasks.py.
TASK_PREFIXES = all_task_prefixes()


def task_prefix(task: str) -> str:
    """Short prefix for a task name, e.g. 'violations_only' -> 'vo'.

    Raises:
        ValueError: if the task is not registered in core/tasks.py. This used to
            fall back to the task name itself, which meant a typo'd --task
            produced a plausible-looking folder name and ran to completion
            against the wrong config.
    """
    return get_task_spec(task).prefix


def variant_name(task: str, phase: str, tier: str, version: str) -> str:
    """Build a standard '<prefix>-<phase>-<tier>-<version>' variant name.

    phase is one of 'baseline', 'sft', 'grpo'. Used for every pipeline artifact
    name by scripts/submit_pipeline.py.
    """
    return f"{task_prefix(task)}-{phase}-{tier}-{version}"


def baseline_run_name(task: str, tier: str, version: str) -> str:
    """Results-folder name for a baseline (no-adapter) inference run.

    Uniform across all tasks. The unified pipeline used to emit a legacy
    unprefixed 'baseline_<tier>_<version>' here, which was the one writable path
    in the repo not namespaced by task and forced compare_results.py and
    plot_metrics.py to special-case task == "unified". Normalized.
    """
    return variant_name(task, "baseline", tier, version)


def merged_checkpoint_name(task: str, tier: str, version: str) -> str:
    """Build the name of the merged-SFT-into-base checkpoint directory used
    as the GRPO KL reference model: 'merged-<prefix>-sft-<tier>-<version>'.

    Task-namespaced so two pipelines bumped to the same --version tag never write
    into the same checkpoint directory. Used by scripts/submit_pipeline.py and
    experiments/run_inference.py's GRPO base-model auto-detect -- both must agree
    on this exact format or GRPO silently trains against the wrong KL reference.
    """
    return f"merged-{task_prefix(task)}-sft-{tier}-{version}"


def results_dir_names(task: str, tier: str, version: str) -> dict:
    """The results/inference/ folder name for each pipeline phase of one run.

    Single source of truth for the analysis layer (compare_results.py,
    plot_metrics.py, generate_comparison_csv.py, plot_metrics_vo.py), so those
    scripts no longer each hand-build the strings.

    The `_best` / `_final` suffixes come from run_inference.py, which names a run
    ``<variant>_<checkpoint>``: SFT is evaluated from `best` (the checkpoint the
    merge step hands to GRPO), while GRPO has no eval dataset and therefore no
    best/, so it stays `final`. Baseline runs pass --run_name directly and carry
    no suffix.
    """
    return {
        "baseline": baseline_run_name(task, tier, version),
        "sft": f"{variant_name(task, 'sft', tier, version)}_best",
        "grpo": f"{variant_name(task, 'grpo', tier, version)}_final",
    }


def slurm_job_name(task: str, phase: str) -> str:
    """SBATCH --job-name for one pipeline phase, e.g. 'vlm-base-oo'."""
    short = {"baseline": "base", "sft": "sft", "merge": "merge-sft", "grpo": "grpo"}
    if phase not in short:
        raise ValueError(f"Unknown phase: {phase!r}. Known: {sorted(short)}")
    return f"vlm-{short[phase]}-{task_prefix(task)}"


def slurm_log_stem(task: str, phase: str) -> str:
    """Log filename stem for one pipeline phase, e.g. 'base_oo'."""
    short = {"baseline": "base", "sft": "sft", "merge": "merge_sft", "grpo": "grpo"}
    if phase not in short:
        raise ValueError(f"Unknown phase: {phase!r}. Known: {sorted(short)}")
    return f"{short[phase]}_{task_prefix(task)}"
