#!/usr/bin/env bash
#SBATCH --job-name=vlm-merge-sft
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/merge_sft_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/merge_sft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

TIER=$1
SFT_VARIANT=$2
MERGED_VARIANT_NAME=$3
if [ -z "$TIER" ] || [ -z "$SFT_VARIANT" ] || [ -z "$MERGED_VARIANT_NAME" ]; then
    echo "Error: Arguments missing (Usage: hpc_merge_sft_vo.sh <tier> <sft_variant> <merged_variant_name>)"
    exit 1
fi

echo "Merge job started: $(date)"
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
export TOKENIZERS_PARALLELISM=false

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

ADAPTER_PATH="$HPC_DRIVE_ROOT/checkpoints/qwen3vl-${TIER}/${SFT_VARIANT}/final"
MERGED_OUTPUT="$HPC_DRIVE_ROOT/checkpoints/qwen3vl-${TIER}/${MERGED_VARIANT_NAME}"

echo "======================================================================"
echo "Merging SFT adapter for tier ${TIER}"
echo "  Adapter path : $ADAPTER_PATH"
echo "  Merged output: $MERGED_OUTPUT"
echo "======================================================================"

python scripts/merge_sft_adapter.py \
    --tier ${TIER} \
    --adapter_path "${ADAPTER_PATH}" \
    --output_path "${MERGED_OUTPUT}"

# Quick verification — confirm the merged model directory exists and has config
if [ -f "${MERGED_OUTPUT}/config.json" ]; then
    echo "VERIFY OK: config.json found in merged output directory"
else
    echo "VERIFY FAIL: config.json NOT found — merge may have failed!"
    exit 1
fi

echo "======================================================================"
echo "Merge complete: $(date)"
echo "Merged model saved to: ${MERGED_OUTPUT}"
echo "======================================================================"
