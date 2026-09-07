#!/usr/bin/env bash
#SBATCH --job-name=vlm-grpo
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h100:1
# GPU POLICY (decided 2026-09-06): GRPO now runs on H100, same as every other stage.
# Moved off the H200 that it used from 2026-09-05 to 2026-09-06. Why the reversal:
# the H200-only policy was originally justified by a 92.97/93.12 GiB OOM on a 93GB
# H100 -- but that OOM was measured BEFORE the pixel-bounds key-rename fix, under
# images that were silently uncapped (up to 14.6 MP). models/grpo_trainer.py::run_grpo
# passes sft_cfg (which unconditionally carries configs/sft.yaml's
# image_max_pixels: 1204224) into load_model_for_training, so the 1.2MP cap SFT trains
# under is ALREADY applied to GRPO too -- confirmed live: the real vo-2b GRPO run
# (job 47873931, on H200) peaked at ~37-38GB via repeated nvidia-smi checks, far below
# even a single H100's ~80-93GB. H100 is also far more plentiful (10 cards across 4
# nodes vs H200's 2 cards on 1 node), so this also cuts queue time substantially.
# CAVEAT: only 2b has been measured. configs/grpo.yaml's per_device_train_batch_size: 16
# was originally sized with the (now-corrected) H200 headroom in mind and has NOT been
# re-tested at 4b/8b under the real 1.2MP cap. Watch nvidia-smi / the GPUMemoryLoggingCallback
# W&B panel early in the first 4b/8b GRPO run; if it OOMs, the documented fallback is
# dropping per_device_train_batch_size to 8 in configs/grpo.yaml (previously verified
# safe, see tests/test_grpo/test_grpo_config.py).
# NOTE: partition gpu-h100 contains BOTH H100 (mgh1,mgh3-5) and H200 (egh2) nodes.
# The GRES type is what actually selects the card, NOT the partition -- and
# --time is a PARTITION property, so it binds identically regardless of GRES type.
# Jobs submitted before this change keep whatever GRES was baked in at THEIR OWN
# submission time -- sbatch reads #SBATCH directives once, at submit, not at run
# time -- so any GRPO job already queued/running under the old script still uses H200.
#SBATCH --time=24:00:00
# Was 48:00:00 -- gpu-h100's real MaxTime is 1-00:00:00 (24h), confirmed via
# `scontrol show partition gpu-h100`. A 48h ask is rejected at submission, not at
# runtime, so this was silently unsubmittable. 24h has no confirmed headroom for
# 4b/8b (the only recorded GRPO walltime is from an invalid prompt-only run) --
# if the wall is hit, models/grpo_trainer.py auto-resumes from the last checkpoint
# (save_steps: 20), so re-submitting this same job continues rather than restarts.
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
# (grpo_oo_<jobid>.out...).
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
GRPO_VARIANT=$3
MERGED_VARIANT_NAME=$4
if [ -z "$TASK" ] || [ -z "$TIER" ] || [ -z "$GRPO_VARIANT" ] || [ -z "$MERGED_VARIANT_NAME" ]; then
    echo "Error: Arguments missing (Usage: hpc_grpo.sh <task> <tier> <grpo_variant> <merged_variant_name>)"
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

WANDB_TAG="$GRPO_VARIANT"
MERGED_BASE="$HPC_DRIVE_ROOT/checkpoints/qwen3vl-${TIER}/${MERGED_VARIANT_NAME}"

echo "======================================================================"
echo "[STEP 1/4] Running GRPO on ${TIER} model (task=$TASK)"
echo "======================================================================"

# The merged model MUST exist — it provides the correct KL reference model. TRL
# computes its reference via disable_adapter(), so without the merge the reference
# is the raw pretrained base rather than the SFT policy. run_grpo.py refuses too;
# this guard just fails before the GPU is even touched.
if [ -d "${MERGED_BASE}" ] && [ -f "${MERGED_BASE}/config.json" ]; then
    echo "Found merged SFT base model at: ${MERGED_BASE}"
    echo "KL reference will correctly point to SFT policy."
    python -m experiments.run_grpo \
        --tier ${TIER} \
        --variant ${GRPO_VARIANT} \
        --task "$TASK" \
        --base_model_override "${MERGED_BASE}"
else
    echo "ERROR: Merged model not found at ${MERGED_BASE}!"
    echo "Run scripts/hpc_merge_sft.sh first, or check that the merge job completed successfully."
    echo "Refusing to start GRPO with wrong KL reference model — aborting."
    exit 1
fi

echo "======================================================================"
echo "[STEP 2/4] Running Inference on Final GRPO Checkpoint"
echo "======================================================================"
python -m experiments.run_inference \
    --tier ${TIER} \
    --variant ${GRPO_VARIANT} \
    --checkpoint final \
    --batch_size 32 \
    --task "$TASK" \
    --base_model_override "${MERGED_BASE}"

PREDS_DIR="$HPC_DRIVE_ROOT/results/inference/${GRPO_VARIANT}_final"
PREDS_FILE="$PREDS_DIR/predictions.jsonl"

echo "======================================================================"
echo "[STEP 3/4] Running Structural Repair (task=$TASK)"
echo "======================================================================"
python preprocessing/structural_repair.py \
    --input "$PREDS_FILE" \
    --output "$PREDS_DIR/repair_applied/predictions_repaired.jsonl" \
    --task "$TASK"

echo "======================================================================"
echo "[STEP 4/4] Running Full Evaluation Pipeline (task=$TASK)"
echo "======================================================================"
REPAIRED_FILE="$PREDS_DIR/repair_applied/predictions_repaired.jsonl"
EVAL_OUT_DIR="$PREDS_DIR/evaluation_results"

python -m experiments.run_evaluation \
    --predictions_path "$REPAIRED_FILE" \
    --output_dir "$EVAL_OUT_DIR" \
    --skip_spice \
    --skip_java_switch \
    --wandb_project "vlm-safety-evals" \
    --wandb_run_name "qwen3-${TIER}-${WANDB_TAG}-repaired" \
    --task "$TASK"

echo "======================================================================"
echo "GRPO and evaluation completed successfully: $(date)"
echo "======================================================================"
