#!/usr/bin/env bash
#SBATCH --job-name=vlm-grpo-test
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h200:1                 # Using H200
#SBATCH --time=00:30:00                   # Quick 30-minute limit
#SBATCH --output=logs/grpo_test_%j.out
#SBATCH --error=logs/grpo_test_%j.err

echo "Test Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

mkdir -p logs

module purge
module load gcc/12.3
module load cuda/12.2
module load python/3.11

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

echo "Starting GRPO SMOKE TEST..."
python -m experiments.run_grpo \
    --tier 2b \
    --variant unified-grpo-test-v1 \
    --sft_variant unified-sft-v1 \
    --max_samples 10

echo "GRPO test completed: $(date)"
