"""The parallel-safety guarantee, as an executable test.

All four task pipelines may run concurrently on ARC at the same --version and the
same tier. Nothing coordinates them at runtime: isolation comes entirely from every
writable path being namespaced by the task prefix. That is a naming property, so it
can be proven here rather than discovered by two jobs overwriting each other's
checkpoints.
"""
import pytest

from core.naming import (
    baseline_run_name,
    merged_checkpoint_name,
    results_dir_names,
    slurm_job_name,
    slurm_log_stem,
    variant_name,
)
from core.tasks import TASK_REGISTRY

TIERS = ["2b", "4b", "8b"]
VERSIONS = ["v1", "v2", "v10"]
TASKS = list(TASK_REGISTRY)


def _all_writable_names(task, tier, version):
    """Every name this task's pipeline writes to, for one tier and version."""
    names = {}

    # Checkpoint directories (per tier, under checkpoints/qwen3vl-<tier>/)
    names["ckpt_sft"] = variant_name(task, "sft", tier, version)
    names["ckpt_grpo"] = variant_name(task, "grpo", tier, version)
    names["ckpt_merged"] = merged_checkpoint_name(task, tier, version)

    # Inference results directories (under results/inference/)
    dirs = results_dir_names(task, tier, version)
    names["results_baseline"] = dirs["baseline"]
    names["results_sft"] = dirs["sft"]
    names["results_grpo"] = dirs["grpo"]
    assert dirs["baseline"] == baseline_run_name(task, tier, version)

    # SLURM job names and log filename stems, per phase
    for phase in ("baseline", "sft", "merge", "grpo"):
        names[f"job_{phase}"] = slurm_job_name(task, phase)
        names[f"log_{phase}"] = slurm_log_stem(task, phase)

    # Oversample manifest (experiments/run_sft.py names it by tier + variant)
    names["oversample_manifest"] = (
        f"oversample_manifest_{tier}_{variant_name(task, 'sft', tier, version)}.json"
    )

    # W&B eval run names, as the generic SLURM scripts build them
    names["wandb_baseline"] = f"qwen3-{tier}-{dirs['baseline']}-repaired"
    names["wandb_sft"] = f"qwen3-{tier}-{variant_name(task, 'sft', tier, version)}-repaired"
    names["wandb_grpo"] = f"qwen3-{tier}-{variant_name(task, 'grpo', tier, version)}-repaired"

    return names


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("version", VERSIONS)
def test_no_writable_name_is_shared_between_any_two_tasks(tier, version):
    """The core guarantee: at the SAME tier and SAME version, no two tasks
    produce a single identical writable name."""
    seen = {}
    for task in TASKS:
        for kind, name in _all_writable_names(task, tier, version).items():
            if name in seen:
                other_task, other_kind = seen[name]
                pytest.fail(
                    f"Name collision at tier={tier} version={version}: {name!r} is "
                    f"produced by both ({task}, {kind}) and ({other_task}, {other_kind})"
                )
            seen[name] = (task, kind)


def test_every_task_produces_the_same_set_of_artifact_kinds():
    """No task may be missing an artifact kind — a missing name is a name that
    falls back to some shared default somewhere."""
    kinds = [set(_all_writable_names(t, "8b", "v1")) for t in TASKS]
    assert all(k == kinds[0] for k in kinds)


@pytest.mark.parametrize("task", TASKS)
def test_names_are_tier_and_version_scoped(task):
    """Beyond cross-task isolation: the same task at a different tier or version
    must also not collide with itself."""
    seen = set()
    for tier in TIERS:
        for version in VERSIONS:
            for kind, name in _all_writable_names(task, tier, version).items():
                # Job names and log stems are intentionally tier/version-agnostic
                # (SLURM disambiguates with %j), so exclude them here.
                if kind.startswith(("job_", "log_")):
                    continue
                key = (kind, name)
                assert key not in seen, f"{task}: {kind} name {name!r} is reused"
                seen.add(key)


@pytest.mark.parametrize("task", TASKS)
def test_grpo_variant_contains_the_grpo_substring(task):
    """run_inference.py's merged-base auto-detect keys on the literal substring
    'grpo' in the variant name; without it, a GRPO checkpoint would be evaluated
    against the raw base model, silently dropping the SFT step."""
    assert "grpo" in variant_name(task, "grpo", "8b", "v1").lower()


@pytest.mark.parametrize("task", TASKS)
def test_merged_name_is_recoverable_from_the_grpo_variant(task):
    """The exact reverse-engineering run_inference.py performs. If this breaks,
    GRPO inference either aborts or (worse) uses the wrong KL base."""
    import re

    version = "v7"
    tier = "4b"
    grpo_variant = variant_name(task, "grpo", tier, version)
    match = re.search(r"-(v\d+)(?:_[^-]*)?$", grpo_variant)
    assert match, f"version tag not recoverable from {grpo_variant!r}"
    assert merged_checkpoint_name(task, tier, match.group(1)) == merged_checkpoint_name(
        task, tier, version
    )
