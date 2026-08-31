#!/usr/bin/env bash
#SBATCH --job-name=vlm-sft-vo
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/sft_vo_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/sft_vo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# Fail fast. Without this, a failed training step still ran inference/repair/eval and the
# job could exit 0 — so the SLURM afterok dependency would start the next stage against a
# missing or stale adapter. NOTE: deliberately not `set -u`; several lines legitimately
# expand possibly-unset vars (PYTHONPATH, SLURM_JOB_ID).
set -eo pipefail

TIER=$1
VARIANT=$2
if [ -z "$TIER" ] || [ -z "$VARIANT" ]; then
    echo "Error: Arguments missing (Usage: hpc_sft_vo.sh <tier> <variant>)"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load gcc/13.3.0
module load python/3.12.5

export PATH="$HOME/scratch/jdk-21.0.2/bin:$PATH"
export JAVA_HOME="$HOME/scratch/jdk-21.0.2"

source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning" || { echo "FATAL: repo checkout not found at $HOME/vlm-safety-reasoning"; exit 1; }

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

echo "======================================================================"
echo "[STEP 1/4] Running SFT on ${TIER} model (Violations Only Task)"
echo "======================================================================"
python -m experiments.run_sft \
    --tier ${TIER} \
    --variant ${VARIANT} \
    --task violations_only

echo "======================================================================"
echo "[STEP 2/4] Running Inference on Best SFT Checkpoint"
echo "======================================================================"
# 'best' (lowest eval_loss), not 'final' (last step) — this is the checkpoint the merge
# step feeds to GRPO, so the reported SFT numbers must describe that same checkpoint.
python -m experiments.run_inference \
    --tier ${TIER} \
    --variant ${VARIANT} \
    --checkpoint best \
    --batch_size 32 \
    --task violations_only

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/${VARIANT}_best"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"

echo "======================================================================"
echo "[STEP 3/4] Running Structural Repair"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl" \
    --task violations_only

echo "======================================================================"
echo "[STEP 4/4] Running Full Evaluation Pipeline"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/evaluation_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_spice \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-${TIER}-${VARIANT}-repaired" \
    --task violations_only

echo "======================================================================"
echo "SFT and Evaluation completed successfully: $(date)"
echo "======================================================================"
