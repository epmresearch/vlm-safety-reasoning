#!/usr/bin/env bash
#SBATCH --job-name=vlm-eval-grpo
#SBATCH --partition=gpu-h100              # ← Using the fastest partition for inference!
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G                         # needs RAM for dataset + sentence transformers
#SBATCH --gres=gpu:h200:1                 # ← Requesting 1x H200 GPU
#SBATCH --time=08:00:00                   # Inference + Eval on 3004 samples takes ~4-6 hours
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/eval_grpo_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/eval_grpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# ─── Reproducibility ──────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# ─── Create log directory ─────────────────────────────────────────────────────
mkdir -p logs

# ─── Load modules ─────────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0
module load python/3.12.5

# Load Java (Required for SPICE / METEOR / CIDEr-D metrics in Captioning Suite)
# If this generic load fails, try `module load java/1.8` or similar on your cluster
module load java || true 

# ─── Activate environment ─────────────────────────────────────────────────────
source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning"

# ─── Environment variables ────────────────────────────────────────────────────
export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="$HOME/scratch/st_cache"

# WandB: use offline on ARC compute nodes (no internet)
export WANDB_MODE=offline
export WANDB_DIR="$HOME/scratch/wandb"
mkdir -p "$WANDB_DIR"

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Load secrets (.env) ──────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ─── Configure paths ──────────────────────────────────────────────────────────
HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"
mkdir -p "$HPC_DRIVE_ROOT/checkpoints" "$HPC_DRIVE_ROOT/results" "$HPC_DRIVE_ROOT/logs"

# Ensure tools directory exists for SPICE cache
mkdir -p "$HPC_DRIVE_ROOT/tools"


# ==============================================================================
# PHASE 1: INFERENCE
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 1/3] Running Inference on GRPO (Final Checkpoint)"
echo "======================================================================"
# Note: we use unified-grpo-v1 and checkpoint 'final'. 
# This runs the model across all 3004 test samples.
python -m experiments.run_inference \
    --tier 2b \
    --variant unified-grpo-v1 \
    --checkpoint final \
    --batch_size 32

# The above command outputs predictions to:
PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/unified-grpo-v1_final"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"


# ==============================================================================
# PHASE 2: STRUCTURAL REPAIR
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 2/3] Running Structural JSON Repair"
echo "======================================================================"
# This safely fixes trailing commas, unclosed brackets, etc. without 
# hallucinating meaning, producing predictions_repaired.jsonl
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE"


# ==============================================================================
# PHASE 3: MULTI-MODAL EVALUATION
# ==============================================================================
echo ""
echo "======================================================================"
echo "[PHASE 3/3] Running Full Evaluation Pipeline (200+ Metrics)"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/grpo_after_repair_results"

# Note: We pass --skip_java_switch because we are on a compute node where 
# update-alternatives (which requires sudo) does not work. We rely on module load java.
python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_java_switch \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-2b-grpo-final-repaired"


echo ""
echo "======================================================================"
echo "GRPO Evaluation completed successfully: $(date)"
echo "Metrics saved to: $EVAL_OUT_DIR/metrics.json"
echo "======================================================================"
