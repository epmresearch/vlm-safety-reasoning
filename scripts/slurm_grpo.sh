#!/usr/bin/env bash
#SBATCH --job-name=vlm-grpo-2b
#SBATCH --account=YOUR_ARC_ACCOUNT       # ← replace with your allocation
#SBATCH --partition=gpu                   # ← check with: sinfo | grep gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G                         # needs RAM for dataset + sentence transformers
#SBATCH --gres=gpu:1                      # 1 GPU; for A100: gpu:a100:1 or gpu:a100l:1
#SBATCH --time=24:00:00                   # GRPO 1 epoch on 6308 samples ≈ 12-18h
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@ucalgary.ca

# ─── Reproducibility ──────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# ─── Create log directory ─────────────────────────────────────────────────────
mkdir -p logs

# ─── Load modules (match what you used in setup_arc.sh) ──────────────────────
module purge
module load gcc/12.3
module load cuda/12.2
module load python/3.11

# ─── Activate environment ─────────────────────────────────────────────────────
source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning"

# ─── Environment variables ────────────────────────────────────────────────────
export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="/scratch/$USER/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="/scratch/$USER/st_cache"

# WandB: use offline on ARC compute nodes (no internet), sync manually later
export WANDB_MODE=offline
export WANDB_DIR="/scratch/$USER/wandb"
mkdir -p "$WANDB_DIR"

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TRL tokenizer parallelism warning suppression
export TOKENIZERS_PARALLELISM=false

# ─── Load secrets (.env) ──────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ─── Configure paths ──────────────────────────────────────────────────────────
# Drive root for HPC: Use persistent home directory
HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"

# Override drive_root at runtime to point to HPC storage instead of Google Drive
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"
mkdir -p "$HPC_DRIVE_ROOT/checkpoints" "$HPC_DRIVE_ROOT/results" "$HPC_DRIVE_ROOT/logs"

# ─── Run GRPO ─────────────────────────────────────────────────────────────────
echo "Starting GRPO training..."
python -m experiments.run_grpo \
    --tier 2b \
    --variant unified-grpo-v1 \
    --sft_variant unified-sft-v1

echo "GRPO training completed: $(date)"

# ─── Sync WandB offline runs (optional, if you want to upload later) ──────────
# wandb sync "$WANDB_DIR/offline-run-*"