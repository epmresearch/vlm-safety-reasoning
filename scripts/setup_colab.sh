#!/usr/bin/env bash
# This script configures the Colab environment AFTER the repo is cloned
set -e

DRIVE_ROOT="/content/drive/MyDrive/vlm-finetuning-project1"

echo ">>> Copying secrets from Drive to local workspace..."
cp "$DRIVE_ROOT/secrets/.env" .env

echo ">>> Exporting environment variables..."
export $(grep -v '^#' .env | xargs)

echo ">>> Configuring Git Identity for commits..."
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"

# Update the hidden config so any future 'git push' uses the token
AUTH_REPO_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/epmresearch/vlm-safety-reasoning.git"
git remote set-url origin "$AUTH_REPO_URL"

echo ">>> Installing Python requirements..."
pip install -q -r requirements.txt

echo ">>> Setup complete!"