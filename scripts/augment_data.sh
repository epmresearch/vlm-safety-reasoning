#!/usr/bin/env bash
#SBATCH --job-name=vlm-augment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/vlm-finetuning-project1/logs/augment_%j.out
#SBATCH --error=/home/%u/vlm-finetuning-project1/logs/augment_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nabeel.shan@ucalgary.ca

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load gcc/13.3.0
module load python/3.12.5

source "$HOME/envs/vlm_grpo/bin/activate"
cd "$HOME/vlm-safety-reasoning"

export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH"
export HF_HOME="$HOME/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HOME/scratch/hf_cache"
export HF_DATASETS_CACHE="$HOME/scratch/hf_datasets_cache"

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

HPC_DRIVE_ROOT="/home/$USER/vlm-finetuning-project1"
export VLM_DATA_ROOT="$HPC_DRIVE_ROOT"
mkdir -p "$HPC_DRIVE_ROOT/logs"

echo "======================================================================"
echo "Running Offline Data Augmentation (Rule 3 & Rule 4)"
echo "======================================================================"

python -m data.augment_rare_classes

echo "======================================================================"
echo "Data augmentation completed successfully: $(date)"
echo "======================================================================"
