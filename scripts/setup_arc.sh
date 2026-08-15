#!/usr/bin/env bash
# setup_arc.sh — Set up the VLM Safety Reasoning environment on UCalgary ARC HPC
# Run once from the LOGIN NODE before submitting any jobs.
# Usage: bash scripts/setup_arc.sh

set -euo pipefail

echo ">>> Setting up VLM Safety Reasoning on ARC HPC"

# ─── 1. Clone / pull repo (do this from login node once) ────────────────────
REPO_DIR="$HOME/vlm-safety-reasoning"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/epmresearch/vlm-safety-reasoning.git "$REPO_DIR"
else
    echo "Repo already present, pulling..."
    cd "$REPO_DIR" && git pull origin main
fi
cd "$REPO_DIR"

# ─── 2. Load modules ─────────────────────────────────────────────────────────
# Check available CUDA: `module spider cuda`
module load gcc/13.3.0
module load python/3.12.5

# ─── 3. Create virtualenv in scratch (fast I/O, not backed up) ───────────────
VENV="$HOME/envs/vlm_grpo"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ─── 4. Install packages ──────────────────────────────────────────────────────
pip install --upgrade pip wheel

# Core ML stack — adjust versions to what was tested on Colab
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Project dependencies
pip install \
    unsloth_zoo \
    datasets \
    huggingface_hub \
    wandb \
    sentence-transformers \
    bert_score \
    shapely \
    pydantic \
    pycocoevalcap \
    pycocotools \
    loguru \
    python-dotenv \
    pandas \
    numpy \
    Pillow \
    PyYAML \
    qwen-vl-utils \
    scikit-learn \
    nltk \
    matplotlib \
    tqdm \
    pytest

# Pin critical ML stack versions at the very end to prevent unsloth_zoo from upgrading them!
pip install transformers==4.46.3 peft==0.14.0 accelerate==1.2.0 trl==0.12.2

# ─── 5. Pre-download HF models to scratch (no internet on compute nodes) ─────
HF_CACHE="$HOME/scratch/hf_cache"
ST_CACHE="$HOME/scratch/st_cache"
mkdir -p "$HF_CACHE" "$ST_CACHE"

export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$ST_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

echo ">>> Pre-downloading sentence transformer (needs internet on login node)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo ">>> Pre-downloading CLIP model for CLIPScore..."
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

echo ">>> Pre-downloading BERTScore model..."
python -c "from bert_score import score; score(['test'], ['test'], lang='en', verbose=False)"

echo ">>> (Qwen3-VL-2B model will be downloaded by the first SLURM job — ensure compute nodes have HF cache access via scratch)"

# ─── 6. Create .env file ──────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
    cat > "$REPO_DIR/.env" << 'EOF'
HF_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
EOF
    echo ">>> Created .env file — fill in your tokens!"
fi

# ─── 7. Verify setup ──────────────────────────────────────────────────────────
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import trl; print('TRL:', trl.__version__)"
python -c "import unsloth; print('Unsloth OK')"

echo ""
echo ">>> Setup complete!"
echo ">>> Next step: edit slurm_grpo.sh with your account, partition, and paths"
echo ">>> Then submit: sbatch scripts/slurm_grpo.sh"