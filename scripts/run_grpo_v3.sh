#!/usr/bin/env bash
#SBATCH --job-name=vlm-grpo-v3
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/grpo_v3_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/grpo_v3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load gcc/13.3.0
module load python/3.12.5

source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning"

export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="$HOME/scratch/st_cache"

export WANDB_MODE=offline
export WANDB_DIR="$HOME/scratch/wandb"
mkdir -p "$WANDB_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"
mkdir -p "$HPC_DRIVE_ROOT/checkpoints" "$HPC_DRIVE_ROOT/results" "$HPC_DRIVE_ROOT/logs"

echo "Starting FRESH GRPO training (V3) from SFT Checkpoint..."
python -m experiments.run_grpo \
    --tier 2b \
    --variant unified-grpo-v3 \
    --sft_variant unified-sft-v1

echo "GRPO V3 training completed: $(date)"
