# VLM Safety Reasoning

A research project focused on utilizing Vision-Language Models (VLMs) for automated construction safety inspections.

## Overview
This project fine-tunes state-of-the-art VLMs (e.g., Qwen3-VL via Unsloth) using a **unified single-prompt strategy**. Rather than running multiple models or separate passes for detection, captioning, and safety analysis, our pipeline trains the model to emit a single, highly structured JSON object capturing the entire scene.

### Key Outputs
- **Scene Captioning**: Dense description of the construction environment.
- **Object Grounding**: Bounding boxes `[xmin, ymin, xmax, ymax]` for key entities like helmets and vests.
- **Safety Violations**: Rule-based safety analysis, severity, and recommendations.

## Project Workflow
1. **Local IDE**: Develop code, configuration, and structural logic.
2. **GitHub**: Version control and sync code.
3. **Google Colab (A100)**: Execute training and evaluation using scripts dynamically pulled from this repo.
4. **Google Drive**: Persist datasets, model checkpoints, and evaluation results (`/content/drive/MyDrive/vlm-finetuning-project1/`).

## Setup

### For Local Development
```bash
pip install -r requirements.txt
```

### For Google Colab
Run the `scripts/setup_colab.sh` in your first notebook cell to clone this repository, mount Google Drive, and install necessary dependencies (Unsloth, bitsandbytes, peft).

## Running Experiments

All experiment entry points are in the `experiments/` directory.

- **Baseline Evaluation**: `python experiments/run_baseline.py --tier 2b`
- **Unified SFT Training**: `python experiments/run_sft.py --tier 2b`
- **Results Comparison**: `python experiments/compare_results.py --tier 2b`

*Note: You can swap `--tier 2b` with `4b` or `8b` as needed.*

## Notebooks
The `notebooks/` directory contains `.py` scripts structured as cell blocks for easy copying into Google Colab, covering dataset preprocessing, baseline inference, training, and evaluation.