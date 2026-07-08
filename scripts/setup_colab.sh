#!/usr/bin/env bash
# Full Colab session bootstrap. Run this as the first cell of every session.
set -e

REPO_URL="https://github.com/epmresearch/vlm-safety-reasoning.git"
REPO_DIR="vlm-safety-reasoning"
DRIVE_ROOT="/content/drive/MyDrive/vlm-finetuning-project1"

if [ -d "$REPO_DIR" ]; then
    echo ">>> Repo already present, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    echo ">>> Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo ">>> Copying secrets from Drive..."
cp "$DRIVE_ROOT/secrets/.env" .env

echo ">>> Installing requirements..."
pip install -q -r requirements.txt

echo ">>> Exporting environment variables..."
export $(grep -v '^#' .env | xargs)

echo ">>> Setup complete. Repo at: $(pwd)"