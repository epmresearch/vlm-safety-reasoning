#!/usr/bin/env bash
#SBATCH --job-name=vlm-base-vo
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/base_vo_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/base_vo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

TIER=$1
if [ -z "$TIER" ]; then
    echo "Error: TIER argument missing (e.g., 2b, 4b, 8b)"
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
cd "$HOME/vlm-safety-reasoning"

export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="$HOME/scratch/st_cache"

export WANDB_MODE=offline
export WANDB_DIR="$HOME/scratch/wandb"
mkdir -p "$WANDB_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

echo "======================================================================"
echo "[PHASE 1/3] Running Inference on ${TIER} Baseline Checkpoint (Violations Only)"
echo "======================================================================"
# We use a dummy variant name to save the results in a unique folder
python -m experiments.run_inference \
    --tier ${TIER} \
    --run_name vo-baseline-${TIER}-v4 \
    --batch_size 32 \
    --task violations_only

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/vo-baseline-${TIER}-v4"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"

echo "======================================================================"
echo "[PHASE 2/3] Running Structural Repair (Violations Only)"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl" \
    --task violations_only

echo "======================================================================"
echo "[PHASE 3/3] Running Full Evaluation Pipeline (Violations Only)"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/evaluation_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_spice \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-${TIER}-vo-baseline-v4-repaired" \
    --task violations_only

echo "======================================================================"
echo "Baseline VO ${TIER} Evaluation completed successfully: $(date)"
echo "======================================================================"
