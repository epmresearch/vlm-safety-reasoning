#!/usr/bin/env python3
import subprocess
import argparse
import sys
import re

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
    args = parser.parse_args()
    
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
        "grpo": "24:00:00"
    }
    
    for tier in args.tiers:
        print(f"\n{'='*50}\nScheduling Pipeline for Qwen3-VL-{tier.upper()}\n{'='*50}")
        
        # 0. Pre-download the model on the login node
        preload_model(tier)
        
        # 1. Baseline Evaluation (No dependencies)
        baseline_mem = mem_config["baseline"].get(tier, "150G")
        baseline_job = submit_job(
            script_path="scripts/hpc_baseline_vo.sh",
            args=[tier],
            mem=baseline_mem,
            time=time_config["baseline"]
        )
        
        # 2. SFT + Evaluation (No dependencies, runs in parallel with Baseline)
        sft_mem = mem_config["sft"].get(tier, "150G")
        sft_job = submit_job(
            script_path="scripts/hpc_sft_vo.sh",
            args=[tier],
            mem=sft_mem,
            time=time_config["sft"]
        )
        
        # 3. Merge SFT adapter into base model (depends on SFT finishing)
        # This creates the correct KL reference model for GRPO.
        merge_mem = "80G"
        merge_job = submit_job(
            script_path="scripts/hpc_merge_sft_vo.sh",
            args=[tier],
            dependencies=[sft_job] if sft_job != "UNKNOWN" else None,
            mem=merge_mem,
            time="01:30:00"
        )
        
        # 4. GRPO + Evaluation (Depends on Merge finishing successfully)
        grpo_mem = mem_config["grpo"].get(tier, "250G")
        grpo_job = submit_job(
            script_path="scripts/hpc_grpo_vo.sh",
            args=[tier],
            dependencies=[merge_job] if merge_job != "UNKNOWN" else None,
            mem=grpo_mem,
            time=time_config["grpo"]
        )
        
        print(f"Pipeline scheduled for {tier}: Baseline({baseline_job}), SFT({sft_job}) -> Merge({merge_job}) -> GRPO({grpo_job})")

if __name__ == "__main__":
    main()
