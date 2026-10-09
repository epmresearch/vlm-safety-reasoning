#!/bin/bash
#SBATCH --job-name=vlm-build-v3-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/build_v3_data_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/build_v3_data_%j.err
#
# Builds datasets/augmented_v3 + datasets/grpo_pool_v3 as a CPU batch job.
#
# WHY A BATCH JOB RATHER THAN A LOGIN NODE. The augmentation step holds every
# augmented PIL image in one list, then encodes each to PNG bytes (Image.fromarray
# leaves .format=None, so HF's encode_pil_image picks PNG, not JPEG), then copies
# those bytes into Arrow -- three live copies of the pixel data at once, with nothing
# freed in between. At ConstructionSite's real resolutions (configs/sft.yaml:107 notes
# 4K/14MP outliers) 450 augmented copies is a multi-GB peak. data/augment_rare_classes.py
# documents the same hazard at its line 26, and scripts/augment_data.sh asks 50G for the
# v2 build. A login-node cgroup kill gives no traceback and no partial output.
#
# No --gres and no --partition: this needs no GPU, and scripts/augment_data.sh -- the
# known-good CPU job on this cluster -- names no partition either, so the default is right.
#
# Usage:
#   sbatch scripts/build_v3_data.sh
#   sbatch scripts/build_v3_data.sh --smoke 400          # rehearsal, writes *_smoke
#   sbatch scripts/build_v3_data.sh --rule4-box-policy keep
# Anything you pass is forwarded verbatim to data/build_v3_datasets.py.

set -eo pipefail

module purge
module load gcc/13.3.0
module load python/3.12.5
source "$HOME/envs/vlm_grpo/bin/activate"

REPO="$HOME/vlm-safety-reasoning"
cd "$REPO" || { echo "FATAL: $REPO not found"; exit 1; }

export PYTHONPATH="$REPO:$PYTHONPATH"
export VLM_DATA_ROOT="$HOME/vlm-finetuning-project1"
export HF_HOME="$HOME/scratch/hf_cache"
export HF_DATASETS_CACHE="$HOME/scratch/hf_datasets_cache"
# albumentations does a synchronous urllib GET to pypi.org on import (timeout 2s, longer
# if DNS hangs). Compute nodes have no internet.
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false

REVIEW="${VLM_DATA_ROOT}/datasets/review_results.json"
if [ ! -f "$REVIEW" ]; then
    echo "FATAL: $REVIEW not found."
    echo "  scp it from your workstation first:"
    echo "  scp review_results.json <user>@arc:~/vlm-finetuning-project1/datasets/"
    exit 1
fi

echo "=============================================================="
echo " build_v3_data   $(date)"
echo " node        : $(hostname)"
echo " review      : $REVIEW"
echo " data root   : $VLM_DATA_ROOT"
python -c "import datasets, albumentations as A; print(f' datasets    : {datasets.__version__}  (ARC pin is 4.3.0)'); print(f' albumentations: {A.__version__}  (needs >=1.4.14 for quality_range)')"
echo " extra args  : $*"
echo "=============================================================="

python data/build_v3_datasets.py --review "$REVIEW" "$@"

echo
echo "=============================================================="
echo " done $(date). Next, before any GPU time:"
echo "   python scripts/dataset_report.py --roots processed augmented_v3 grpo_pool_v3"
echo "   python scripts/validate_think_dataset.py --subdir datasets/augmented_v3 --strict"
echo "=============================================================="
