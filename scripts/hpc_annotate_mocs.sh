#!/usr/bin/env bash
#SBATCH --job-name=vlm-annot-mocs
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:h100:1
# H100, not H200. Qwen3-VL-32B in bf16 is ~66 GB of an 80 GB card, which fits with
# ~14 GB for the vision encoder, KV cache and generation at batch 2. H100 is also far
# more plentiful on this cluster (10 cards across 4 nodes vs H200's 2 cards on one
# node), so the queue wait is much shorter -- and this is a ~2 h job, not a 12 h one.
# If it OOMs: drop --batch-size to 1, or pass --load-4bit (~20 GB).
# NOTE: partition gpu-h100 contains BOTH H100 (mgh1,mgh3-5) and H200 (egh2) nodes --
# the GRES type selects the card, not the partition.
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/%x_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

# ---------------------------------------------------------------------------
# MOCS auto-annotation pass. Produces PROPOSALS for human review; produces no
# training data and touches nothing the training pipeline reads.
#
# This is NOT part of the baseline/sft/merge/grpo pipeline and is deliberately
# absent from scripts/submit_pipeline.py::PHASE_SCRIPTS -- submit it by hand.
#
# Reads : $VLM_DATA_ROOT/datasets/mocs_annotation/{selection.json,fewshot.json}
# Writes: $VLM_DATA_ROOT/datasets/mocs_annotation/{proposals.jsonl,progress.json,
#                                                   annotate_manifest.json}
#
# RE-SUBMITTING THIS JOB IS SAFE AND IS THE RECOVERY PATH. annotate.py skips every
# image already present in proposals.jsonl, so a walltime kill or a NODE_FAIL costs
# at most the batch in flight. Just sbatch it again.
#
# Usage:
#   sbatch scripts/hpc_annotate_mocs.sh                 # batch 2, all selected images
#   sbatch scripts/hpc_annotate_mocs.sh 4               # batch 4
#   sbatch scripts/hpc_annotate_mocs.sh 2 8             # smoke test: 8 images only
# ---------------------------------------------------------------------------

# Fail fast. Deliberately not `set -u`: PYTHONPATH and SLURM_JOB_ID are legitimately
# expanded while possibly unset, matching the other phase scripts.
set -eo pipefail

BATCH_SIZE=${1:-2}
LIMIT=${2:-}
MODEL=${MOCS_ANNOT_MODEL:-Qwen/Qwen3-VL-32B-Instruct}

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Limit: ${LIMIT:-<none, all selected images>}"
# `|| true`: informational only. Under `set -e` a transient non-zero here would abort
# the job before a single useful line was logged.
nvidia-smi || true

module purge
module load gcc/13.3.0
module load python/3.12.5

source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning" || { echo "FATAL: repo checkout not found at $HOME/vlm-safety-reasoning"; exit 1; }

export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Without this the job runs with no HF_TOKEN.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"

OUT_DIR="$HPC_DRIVE_ROOT/datasets/mocs_annotation"
SELECTION="$OUT_DIR/selection.json"
FEWSHOT="$OUT_DIR/fewshot.json"

# Guard before the GPU is touched, in the style of hpc_merge_sft.sh / hpc_grpo.sh.
# Both inputs are produced by CPU-only login-node steps; a missing one means a step
# was skipped, and discovering that after a queue wait is a wasted allocation.
if [ ! -f "$SELECTION" ]; then
    echo "ERROR: no selection file at $SELECTION"
    echo "Run this on the login node first:"
    echo "  python -m mocs_annotation.select_images \\"
    echo "      --annotations \$VLM_DATA_ROOT/datasets/filtered/annotation_val.json \\"
    echo "      --images-root \$VLM_DATA_ROOT/datasets/filtered/instances_val \\"
    echo "      --out         $SELECTION"
    exit 1
fi

if [ ! -f "$FEWSHOT" ]; then
    echo "ERROR: no few-shot file at $FEWSHOT"
    echo "Run this on the login node first:"
    echo "  python -m mocs_annotation.build_fewshot --out $FEWSHOT"
    echo "Refusing to run without it: an unconditioned teacher emits the wrong JSON"
    echo "shape and the wrong reason register, which silently poisons the harvest."
    exit 1
fi

echo "======================================================================"
echo "Annotating MOCS candidates with $MODEL"
echo "  selection : $SELECTION"
echo "  few-shot  : $FEWSHOT"
echo "  output    : $OUT_DIR"
echo "======================================================================"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python -m mocs_annotation.annotate \
    --selection "$SELECTION" \
    --fewshot "$FEWSHOT" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    $LIMIT_ARG

echo "======================================================================"
echo "Annotation pass finished: $(date)"
echo "Proposals: $OUT_DIR/proposals.jsonl"
echo "Progress : $OUT_DIR/progress.json"
echo ""
echo "Next (login node, CPU only) -- build the review queue and read the yield table:"
echo "  python -m mocs_annotation.export_review \\"
echo "      --proposals $OUT_DIR/proposals.jsonl \\"
echo "      --selection $SELECTION \\"
echo "      --out       $OUT_DIR/review.csv"
echo "======================================================================"
