#!/usr/bin/env bash
#SBATCH --job-name=vlm-grpo-vo
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/grpo_vo_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/grpo_vo_%j.err
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
export TOKENIZERS_PARALLELISM=false

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

VARIANT="vo-grpo-${TIER}-v1"
SFT_VARIANT="vo-sft-${TIER}-v1"

echo "======================================================================"
echo "[STEP 1/4] Running GRPO on ${TIER} model (Violations Only Task)"
echo "======================================================================"
python -m experiments.run_grpo \
    --tier ${TIER} \
    --variant ${VARIANT} \
    --sft_variant ${SFT_VARIANT} \
    --task violations_only

echo "======================================================================"
echo "[STEP 2/4] Running Inference on Final GRPO Checkpoint"
echo "======================================================================"
python -m experiments.run_inference \
    --tier ${TIER} \
    --variant ${VARIANT} \
    --checkpoint final \
    --batch_size 32 \
    --task violations_only

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/${VARIANT}_final"
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
    --wandb_run_name "qwen3-${TIER}-vo-grpo-v1-repaired" \
    --task violations_only

echo "======================================================================"
echo "GRPO and Evaluation completed successfully: $(date)"
echo "======================================================================"
