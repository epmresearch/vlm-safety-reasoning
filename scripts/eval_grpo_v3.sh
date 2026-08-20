#!/usr/bin/env bash
#SBATCH --job-name=vlm-eval-grpo-v3
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/eval_grpo_v3_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/eval_grpo_v3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load gcc/13.3.0
module load python/3.12.5

# Inject Portable Java for pycocoevalcap metrics (METEOR/CIDEr)
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
echo "[PHASE 1/3] Running Inference on GRPO V3 (Checkpoint 836)"
echo "======================================================================"
python -m experiments.run_inference \
    --tier 2b \
    --variant unified-grpo-v3 \
    --checkpoint checkpoint-836 \
    --batch_size 32

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/unified-grpo-v3_checkpoint-836"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"

echo "======================================================================"
echo "[PHASE 2/3] Running Structural Repair"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl"

echo "======================================================================"
echo "[PHASE 3/3] Running Full Evaluation Pipeline"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/evaluation_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_spice \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-2b-grpo-v3-repaired"

echo "======================================================================"
echo "GRPO V3 Evaluation completed successfully: $(date)"
echo "Metrics saved to: $EVAL_OUT_DIR/metrics.json"
echo "======================================================================"
