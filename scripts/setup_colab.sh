#!/usr/bin/env bash
# Full Colab session bootstrap. Run this as the first cell of every session.
set -e

REPO_DIR="vlm-safety-reasoning"
DRIVE_ROOT="/content/drive/MyDrive/vlm-finetuning-project1"

echo ">>> Copying secrets from Drive..."
# Make sure Google Drive is actually mounted before running this!
cp "$DRIVE_ROOT/secrets/.env" .env

echo ">>> Exporting environment variables..."
export $(grep -v '^#' .env | xargs)

echo ">>> Configuring Git Identity..."
# Uses the variables we will put in your .env file
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"

# Dynamically construct URL using your Token and Username
AUTH_REPO_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/epmresearch/vlm-safety-reasoning.git"

if [ -d "$REPO_DIR" ]; then
    echo ">>> Repo already present, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    echo ">>> Cloning repo..."
    git clone "$AUTH_REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo ">>> Installing requirements..."
pip install -q -r requirements.txt

echo ">>> Setup complete. Repo at: $(pwd)"