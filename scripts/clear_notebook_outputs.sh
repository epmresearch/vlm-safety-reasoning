#!/usr/bin/env bash
# Run before every commit to strip notebook outputs from version control.
set -e

for f in notebooks/*.ipynb; do
    if [ -f "$f" ]; then
        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$f"
        echo "Cleared outputs: $f"
    fi
done