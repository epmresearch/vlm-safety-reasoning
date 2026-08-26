#!/usr/bin/env bash
#SBATCH --job-name=test-eval-grpo
#SBATCH --partition=gpu-h100              # ← Using the fastest partition for inference!
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G                         # needs RAM for dataset + sentence transformers
#SBATCH --gres=gpu:h100:1                 # ← Requesting 1x H100 GPU
#SBATCH --time=00:30:00                   # 30 mins is plenty for 10 samples
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/test_eval_grpo_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/test_eval_grpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# ─── Reproducibility ──────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# ─── Create log directory ─────────────────────────────────────────────────────

# ─── Load modules ─────────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0
module load python/3.12.5
module load java || true 

# ─── Activate environment ─────────────────────────────────────────────────────
source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning"

# ─── Environment variables ────────────────────────────────────────────────────
export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="$HOME/scratch/st_cache"

export WANDB_MODE=offline
export WANDB_DIR="$HOME/scratch/wandb"
mkdir -p "$WANDB_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Load secrets (.env) ──────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"
mkdir -p "$HPC_DRIVE_ROOT/checkpoints" "$HPC_DRIVE_ROOT/results" "$HPC_DRIVE_ROOT/logs"
mkdir -p "$HPC_DRIVE_ROOT/tools"


# ==============================================================================
# PHASE 1: INFERENCE (SMOKE TEST = 10 SAMPLES)
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 1/3] Running Smoke Test Inference (Baseline Model)"
echo "======================================================================"
# We run the BASELINE model (no variant) so you can test this script RIGHT NOW 
# without waiting for your GRPO training to finish!
python -m experiments.run_inference \
    --tier 2b \
    --batch_size 10 \
    --max_samples 10 \
    --run_name "smoke_test_baseline"

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/smoke_test_baseline"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"


# ==============================================================================
# PHASE 2: STRUCTURAL REPAIR
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 2/3] Running Structural JSON Repair"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl"


# ==============================================================================
# PHASE 3: MULTI-MODAL EVALUATION
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 3/3] Running Full Evaluation Pipeline (10 samples)"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/grpo_after_repair_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_java_switch \
    --max_samples 10 \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "smoke-test-eval"

echo ""
echo "======================================================================"
echo "Smoke Test Evaluation completed successfully: $(date)"
echo "Metrics saved to: $EVAL_OUT_DIR/metrics.json"
echo "======================================================================"
