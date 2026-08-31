#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
import sys
import re
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.naming import merged_checkpoint_name

def submit_job(script_path, args, dependencies=None, mem=None, time=None):
    cmd = ["sbatch"]
    
    if dependencies:
        # dependencies can be a list of job ids
        deps_str = ":".join(str(d) for d in dependencies)
        cmd.append(f"--dependency=afterok:{deps_str}")
        
    if mem:
        cmd.append(f"--mem={mem}")
        
    if time:
        cmd.append(f"--time={time}")
        
    cmd.append(script_path)
    cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # We use a dummy submission for local testing/Windows, but on HPC it will run sbatch
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
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to pre-download {model_name}. Jobs may experience cache locking issues.\n")

def main():
    parser = argparse.ArgumentParser(description="Submit Violations Only Pipeline to SLURM")
    parser.add_argument("--tiers", nargs="+", default=["2b", "4b", "8b"], help="Model tiers to run")
    parser.add_argument(
        "--version", required=True,
        help="Run version tag (e.g. v5, v6). Stamped into every variant name this "
             "pipeline produces (baseline/SFT/merge/GRPO) — the only thing you need "
             "to change to start a fresh versioned run."
    )
    args = parser.parse_args()

    # run_inference.py reverse-engineers the merged-SFT base from the variant name using
    # the regex -(v\d+)(?:_[^-]*)?$. A free-form tag like "v5b" or "2025-08" yields an
    # empty version, a wrong merged path, and a SystemExit — but only AFTER GRPO training
    # has already finished. Fail here instead, before anything is submitted.
    if not re.fullmatch(r"v\d+", args.version):
        raise SystemExit(
            f"--version must match 'v<digits>' (e.g. v5, v12); got {args.version!r}. "
            "Other formats break the merged-checkpoint lookup in run_inference.py."
        )

    # SLURM cannot create the log directory itself: if it is missing, the job fails at
    # launch with no output anywhere to explain why.
    log_dir = Path(f"/home/{os.environ.get('USER', '')}/vlm-finetuning-project1/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: could not create SLURM log dir {log_dir}: {e}")
    
    # Memory configurations (adjust as needed based on HPC constraints)
    # GRPO is memory intensive due to multiple reference models in vram
    mem_config = {
        "baseline": {"2b": "150G", "4b": "150G", "8b": "150G"},
        "sft": {"2b": "150G", "4b": "150G", "8b": "150G"},
        "grpo": {"2b": "250G", "4b": "250G", "8b": "250G"}
    }
    
    time_config = {
        "baseline": "12:00:00",
        "sft": "12:00:00",
        # 48h, not 24h: GRPO is now 2 epochs, and per-step cost rose once images actually
        # reached the model. The post-training stages (inference on 3004 + repair + eval)
        # cost ~4-7h on top and cannot be compressed. Over-requesting only costs queue
        # priority; being killed at 90% costs the entire run.
        "grpo": "48:00:00"
    }
    
    for tier in args.tiers:
        print(f"\n{'='*50}\nScheduling Pipeline for Qwen3-VL-{tier.upper()}\n{'='*50}")
        
        # 0. Pre-download the model on the login node
        preload_model(tier)
        
        # Dynamic variant naming — args.version is the single source of truth.
        # merged_variant is task-namespaced (merged-vo-sft-...) so it can never
        # collide with the unified pipeline's merged checkpoint at the same version.
        sft_variant = f"vo-sft-{tier}-{args.version}"
        merged_variant = merged_checkpoint_name("violations_only", tier, args.version)
        grpo_variant = f"vo-grpo-{tier}-{args.version}"

        # 1. Baseline Evaluation (No dependencies)
        baseline_mem = mem_config["baseline"].get(tier, "150G")
        baseline_job = submit_job(
            script_path="scripts/hpc_baseline_vo.sh",
            args=[tier, args.version],
            mem=baseline_mem,
            time=time_config["baseline"]
        )
        
        # 2. SFT + Evaluation (No dependencies, runs in parallel with Baseline)
        sft_mem = mem_config["sft"].get(tier, "150G")
        sft_job = submit_job(
            script_path="scripts/hpc_sft_vo.sh",
            args=[tier, sft_variant],
            mem=sft_mem,
            time=time_config["sft"]
        )
        
        # 3. Merge SFT adapter into base model (depends on SFT finishing)
        # This creates the correct KL reference model for GRPO.
        merge_mem = "80G"
        merge_job = submit_job(
            script_path="scripts/hpc_merge_sft_vo.sh",
            args=[tier, sft_variant, merged_variant],
            dependencies=[sft_job] if sft_job != "UNKNOWN" else None,
            mem=merge_mem,
            time="01:30:00"
        )
        
        # 4. GRPO + Evaluation (Depends on Merge finishing successfully)
        grpo_mem = mem_config["grpo"].get(tier, "250G")
        grpo_job = submit_job(
            script_path="scripts/hpc_grpo_vo.sh",
            args=[tier, grpo_variant, merged_variant],
            dependencies=[merge_job] if merge_job != "UNKNOWN" else None,
            mem=grpo_mem,
            time=time_config["grpo"]
        )
        
        print(f"Pipeline scheduled for {tier}: Baseline({baseline_job}), SFT({sft_job}) -> Merge({merge_job}) -> GRPO({grpo_job})")

if __name__ == "__main__":
    main()
