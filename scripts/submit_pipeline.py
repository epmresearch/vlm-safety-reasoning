#!/usr/bin/env python3
"""
Master SLURM orchestrator for ANY task pipeline.

One submitter, one set of generic phase scripts. Adding a task adds no
orchestration code at all — only a TaskSpec in core/tasks.py and its data
artifacts (task YAML, prompt, schema, target/GT builders).

Submits four jobs per tier with `afterok` dependencies:

    baseline  (independent)
    sft -> merge -> grpo

Usage:
    python scripts/submit_pipeline.py --task object_only  --tiers 2b 4b 8b --version v1
    python scripts/submit_pipeline.py --task caption_only --tiers 8b       --version v1

The per-task wrappers (submit_unified_pipeline.py, submit_vo_pipeline.py,
submit_oo_pipeline.py, submit_co_pipeline.py) just inject --task and call in here.

Every writable name this produces is namespaced by the task prefix from
core/naming.py, which is what makes it safe to fire all four pipelines back to
back at the same --version and tier. Nothing coordinates them at runtime;
isolation is purely a naming property, asserted by
tests/test_core/test_name_isolation.py.

On Windows there is no `sbatch`, so this degrades into printing the exact
commands it would run (with DUMMY_JOB_ID) — a useful dry run.
"""
from pathlib import Path
import subprocess
import argparse
import sys
import re
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.constants import VALID_TASKS
from core.naming import (
    baseline_run_name,
    merged_checkpoint_name,
    slurm_job_name,
    slurm_log_stem,
    task_prefix,
    variant_name,
)

# Memory configurations (adjust as needed based on HPC constraints).
# GRPO is memory intensive due to multiple reference models in vram.
MEM_CONFIG = {
    "baseline": {"2b": "150G", "4b": "150G", "8b": "150G"},
    "sft": {"2b": "150G", "4b": "150G", "8b": "150G"},
    "merge": {"2b": "80G", "4b": "80G", "8b": "80G"},
    "grpo": {"2b": "250G", "4b": "250G", "8b": "250G"},
}

TIME_CONFIG = {
    "baseline": "12:00:00",
    "sft": "12:00:00",
    "merge": "01:30:00",
    # 48h, not 24h: GRPO is 2 epochs, and per-step cost rose once images actually
    # reached the model. The post-training stages (inference on 3004 + repair + eval)
    # cost ~4-7h on top and cannot be compressed. Over-requesting only costs queue
    # priority; being killed at 90% costs the entire run.
    "grpo": "48:00:00",
}

PHASE_SCRIPTS = {
    "baseline": "scripts/hpc_baseline.sh",
    "sft": "scripts/hpc_sft.sh",
    "merge": "scripts/hpc_merge_sft.sh",
    "grpo": "scripts/hpc_grpo.sh",
}


def submit_job(script_path, args, dependencies=None, mem=None, time=None,
               job_name=None, log_stem=None):
    cmd = ["sbatch"]

    if dependencies:
        # dependencies can be a list of job ids
        deps_str = ":".join(str(d) for d in dependencies)
        cmd.append(f"--dependency=afterok:{deps_str}")

    if mem:
        cmd.append(f"--mem={mem}")

    if time:
        cmd.append(f"--time={time}")

    # Command-line --job-name / --output / --error beat the in-file #SBATCH
    # directives, which is how one generic script yields per-task job names and
    # log paths.
    if job_name:
        cmd.append(f"--job-name={job_name}")
    if log_stem:
        log_dir = "/home/%u/vlm-finetuning-project1/logs"
        cmd.append(f"--output={log_dir}/{log_stem}_%j.out")
        cmd.append(f"--error={log_dir}/{log_stem}_%j.err")

    cmd.append(script_path)
    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")

    # A dummy submission for local testing/Windows; on HPC this runs sbatch.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Expected output: "Submitted batch job 123456"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Successfully submitted job {job_id}")
            return job_id
        else:
            print(f"Warning: Could not parse job ID from sbatch output: {result.stdout}")
            return "UNKNOWN"
    except FileNotFoundError:
        print("sbatch command not found (are you on a SLURM cluster?). Printing command instead.")
        return "DUMMY_JOB_ID"
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        sys.exit(1)


def preload_model(tier):
    tier_map = {"2b": "2B", "4b": "4B", "8b": "8B"}
    model_name = f"unsloth/Qwen3-VL-{tier_map.get(tier.lower(), tier.upper())}-Instruct"
    print(f"Pre-downloading {model_name} to HuggingFace cache to prevent SLURM file-lock collisions...")
    try:
        subprocess.run(["hf", "download", model_name], check=True)
        print(f"Successfully cached {model_name}\n")
    except FileNotFoundError:
        print("Warning: huggingface-cli not found. Make sure your environment is activated before running this script.\n")
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to pre-download {model_name}. Jobs may experience cache locking issues.\n")


def build_parser(task_default=None):
    parser = argparse.ArgumentParser(
        description="Submit a full task pipeline (baseline / sft -> merge -> grpo) to SLURM"
    )
    parser.add_argument(
        "--task", default=task_default, required=task_default is None,
        choices=VALID_TASKS,
        help="Task pipeline to submit. Every generated name is namespaced by this "
             "task's prefix, so pipelines for different tasks can run concurrently "
             "at the same --version and tier.",
    )
    parser.add_argument("--tiers", nargs="+", default=["2b", "4b", "8b"], help="Model tiers to run")
    parser.add_argument(
        "--version", required=True,
        help="Run version tag (e.g. v1, v2). Stamped into every variant name this "
             "pipeline produces (baseline/SFT/merge/GRPO) — the only thing you need "
             "to change to start a fresh versioned run.",
    )
    parser.add_argument(
        "--skip-preload", action="store_true",
        help="Skip the login-node model pre-download. Only safe when the HF cache is "
             "already warm for every requested tier; the pre-download exists to stop "
             "concurrent SLURM jobs racing on HF cache locks.",
    )
    return parser


def run(task: str, tiers, version: str, skip_preload: bool = False):
    # run_inference.py reverse-engineers the merged-SFT base from the variant name using
    # the regex -(v\d+)(?:_[^-]*)?$. A free-form tag like "v5b" or "2025-08" yields an
    # empty version, a wrong merged path, and a SystemExit — but only AFTER GRPO training
    # has already finished. Fail here instead, before anything is submitted.
    if not re.fullmatch(r"v\d+", version):
        raise SystemExit(
            f"--version must match 'v<digits>' (e.g. v1, v12); got {version!r}. "
            "Other formats break the merged-checkpoint lookup in run_inference.py."
        )

    prefix = task_prefix(task)

    # SLURM cannot create the log directory itself: if it is missing, the job fails at
    # launch with no output anywhere to explain why.
    log_dir = Path(f"/home/{os.environ.get('USER', '')}/vlm-finetuning-project1/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: could not create SLURM log dir {log_dir}: {e}")

    for tier in tiers:
        print(f"\n{'='*60}\nScheduling {task} ({prefix}) pipeline for Qwen3-VL-{tier.upper()}\n{'='*60}")

        # 0. Pre-download the model on the login node
        if not skip_preload:
            preload_model(tier)

        # All names come from core.naming, so the submitter, the merge step,
        # run_inference.py's GRPO KL-reference lookup and the analysis scripts
        # cannot disagree about a single character.
        sft_variant = variant_name(task, "sft", tier, version)
        grpo_variant = variant_name(task, "grpo", tier, version)
        merged_variant = merged_checkpoint_name(task, tier, version)
        baseline_name = baseline_run_name(task, tier, version)

        print(f"  baseline results dir : {baseline_name}")
        print(f"  sft variant          : {sft_variant}")
        print(f"  merged KL base       : {merged_variant}")
        print(f"  grpo variant         : {grpo_variant}")

        # 1. Baseline Evaluation (no dependencies)
        baseline_job = submit_job(
            script_path=PHASE_SCRIPTS["baseline"],
            args=[task, tier, version],
            mem=MEM_CONFIG["baseline"].get(tier, "150G"),
            time=TIME_CONFIG["baseline"],
            job_name=slurm_job_name(task, "baseline"),
            log_stem=slurm_log_stem(task, "baseline"),
        )

        # 2. SFT + Evaluation (no dependencies, runs in parallel with Baseline)
        sft_job = submit_job(
            script_path=PHASE_SCRIPTS["sft"],
            args=[task, tier, sft_variant],
            mem=MEM_CONFIG["sft"].get(tier, "150G"),
            time=TIME_CONFIG["sft"],
            job_name=slurm_job_name(task, "sft"),
            log_stem=slurm_log_stem(task, "sft"),
        )

        # 3. Merge SFT adapter into base model (depends on SFT finishing).
        # This creates the correct KL reference model for GRPO.
        merge_job = submit_job(
            script_path=PHASE_SCRIPTS["merge"],
            args=[task, tier, sft_variant, merged_variant],
            dependencies=[sft_job] if sft_job != "UNKNOWN" else None,
            mem=MEM_CONFIG["merge"].get(tier, "80G"),
            time=TIME_CONFIG["merge"],
            job_name=slurm_job_name(task, "merge"),
            log_stem=slurm_log_stem(task, "merge"),
        )

        # 4. GRPO + Evaluation (depends on Merge finishing successfully)
        grpo_job = submit_job(
            script_path=PHASE_SCRIPTS["grpo"],
            args=[task, tier, grpo_variant, merged_variant],
            dependencies=[merge_job] if merge_job != "UNKNOWN" else None,
            mem=MEM_CONFIG["grpo"].get(tier, "250G"),
            time=TIME_CONFIG["grpo"],
            job_name=slurm_job_name(task, "grpo"),
            log_stem=slurm_log_stem(task, "grpo"),
        )

        print(
            f"Pipeline scheduled for {task} {tier}: Baseline({baseline_job}), "
            f"SFT({sft_job}) -> Merge({merge_job}) -> GRPO({grpo_job})"
        )


def main(task_default=None, argv=None):
    args = build_parser(task_default).parse_args(argv)
    run(args.task, args.tiers, args.version, skip_preload=args.skip_preload)


if __name__ == "__main__":
    main()
