#!/usr/bin/env bash
#SBATCH --job-name=vlm-merge-sft
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h200:1
# NOTE: partition gpu-h100 contains BOTH H100 (mgh1,mgh3-5) and H200 (egh2) nodes.
# The GRES type is what actually selects the card. configs/grpo.yaml is tuned for
# the H200's 141 GB (per_device_train_batch_size: 16 is documented as OOMing at
# 92.97/93.12 GiB on a 93 GB H100), so the H200 must be requested explicitly.
# Only egh2 has H200s (2 of them) — expect queue waits.
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/%x_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# ---------------------------------------------------------------------------
# GENERIC, TASK-PARAMETERIZED. One script serves every task pipeline
# (unified / violations_only / object_only / caption_only); the task is the
# first positional argument and is threaded to every Python call explicitly.
#
# The --job-name and --output/--error directives above are per-task DEFAULTS.
# scripts/submit_pipeline.py overrides all three on the sbatch command line
# (which beats the in-file directives, the same way it already overrides --mem
# and --time), so each task's jobs and logs are namespaced by its prefix
# (merge_sft_oo_<jobid>.out...).
# %x in the paths above means the job name, so even a hand-run
# `sbatch --job-name=... ` stays namespaced.
# ---------------------------------------------------------------------------

# Fail fast. Without this, a failed training step still ran inference/repair/eval and the
# job could exit 0 — so the SLURM afterok dependency would start the next stage against a
# missing or stale adapter. NOTE: deliberately not `set -u`; several lines legitimately
# expand possibly-unset vars (PYTHONPATH, SLURM_JOB_ID).
set -eo pipefail

TASK=$1
TIER=$2
SFT_VARIANT=$3
MERGED_VARIANT_NAME=$4
if [ -z "$TASK" ] || [ -z "$TIER" ] || [ -z "$SFT_VARIANT" ] || [ -z "$MERGED_VARIANT_NAME" ]; then
    echo "Error: Arguments missing (Usage: hpc_merge_sft.sh <task> <tier> <sft_variant> <merged_variant_name>)"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Task: $TASK"
# `|| true`: informational only. Under `set -e` a transient non-zero exit
# here would abort the job before a single useful line was logged.
nvidia-smi || true

module purge
module load gcc/13.3.0
module load python/3.12.5

# Portable Java for pycocoevalcap (METEOR / CIDEr-D / SPICE). Exported
# unconditionally; evaluation only *requires* a JVM for tasks that score text
# (see evaluation/evaluator.py's capability gate), so object_only runs fine
# without one, but there is no cost to having it on PATH.
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

# Without this the job runs with no HF_TOKEN / WANDB_API_KEY.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

# Validate the task against core/tasks.py::TASK_REGISTRY before doing any work.
# A typo would otherwise surface much later as a FileNotFoundError on
# configs/tasks/<typo>.yaml, after the model had already been loaded.
python -c "from core.tasks import validate_task; validate_task('$TASK')" \
    || { echo "FATAL: unknown --task '$TASK' (see core/tasks.py::TASK_REGISTRY)"; exit 1; }

ADAPTER_PATH="$HPC_DRIVE_ROOT/checkpoints/qwen3vl-${TIER}/${SFT_VARIANT}/best"
MERGED_OUTPUT="$HPC_DRIVE_ROOT/checkpoints/qwen3vl-${TIER}/${MERGED_VARIANT_NAME}"

echo "======================================================================"
echo "Merging SFT adapter for tier ${TIER} (task=$TASK)"
echo "  Adapter path : $ADAPTER_PATH"
echo "  Merged output: $MERGED_OUTPUT"
echo "======================================================================"

if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ] && [ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]; then
    echo "ERROR: no SFT adapter found at ${ADAPTER_PATH}."
    echo "The SFT job must have written best/ before this stage runs."
    exit 1
fi

# --task is passed EXPLICITLY and merge_sft_adapter.py now *requires* it. It used
# to default to "violations_only", which was right for exactly one caller and
# silently wrong for every other — a merge that loaded another task's
# max_seq_length / pixel bounds while looking perfectly healthy.
python scripts/merge_sft_adapter.py \
    --tier ${TIER} \
    --adapter_path "${ADAPTER_PATH}" \
    --output_path "${MERGED_OUTPUT}" \
    --task "$TASK"

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
