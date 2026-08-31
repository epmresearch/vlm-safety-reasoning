#!/usr/bin/env bash
#SBATCH --job-name=vlm-base-unified
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/base_unified_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/base_unified_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# Fail fast. Without this, a failed training step still ran inference/repair/eval and the
# job could exit 0 — so the SLURM afterok dependency would start the next stage against a
# missing or stale adapter. NOTE: deliberately not `set -u`; several lines legitimately
# expand possibly-unset vars (PYTHONPATH, SLURM_JOB_ID).
set -eo pipefail

TIER=$1
VERSION=$2
if [ -z "$TIER" ] || [ -z "$VERSION" ]; then
    echo "Error: Arguments missing (Usage: hpc_baseline_unified.sh <tier> <version>, e.g. 2b v5)"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load gcc/13.3.0
module load python/3.12.5

# Inject Portable Java for pycocoevalcap metrics
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

# Parity with the sft/merge/grpo scripts: without this the baseline job runs with no
# HF_TOKEN / WANDB_API_KEY.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

echo "======================================================================"
echo "[PHASE 1/3] Running Inference on ${TIER} Baseline Checkpoint (Unified)"
echo "======================================================================"
# Matches the legacy unified naming convention (baseline_<tier>_<version>,
# no task prefix) that experiments/compare_results.py and plot_metrics.py
# already expect for task=unified.
python -m experiments.run_inference \
    --tier ${TIER} \
    --run_name baseline_${TIER}_${VERSION} \
    --batch_size 32 \
    --task unified

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/baseline_${TIER}_${VERSION}"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"

echo "======================================================================"
echo "[PHASE 2/3] Running Structural Repair (Unified)"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl" \
    --task unified

echo "======================================================================"
echo "[PHASE 3/3] Running Full Evaluation Pipeline (Unified)"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/evaluation_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_spice \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-${TIER}-baseline-${VERSION}-repaired" \
    --task unified

echo "======================================================================"
echo "Baseline Unified ${TIER} Evaluation completed successfully: $(date)"
echo "======================================================================"
