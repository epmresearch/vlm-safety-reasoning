# Project Architecture

## Core Concept
This project implements a single-prompt unified training strategy for Vision-Language Models (VLMs) applied to construction safety inspection. Rather than using task-specific adapters, a single model is trained to generate a structured JSON output containing:
1. A dense caption of the scene
2. Bounding boxes and labels for key safety objects (helmets, vests)
3. Safety violations with severity and recommendations

## Directory Structure
- `core/`: Constants, configurations, logging, and utilities (W&B, I/O).
- `data/`: Dataset loading, prompt templating, caching, and preprocessing (creating the unified SFT messages).
- `models/`: HuggingFace model loading (Unsloth), unified SFT training, and inference wrappers.
- `evaluation/`: Unified evaluation pipeline (structural checks, BERTScore, CLIPScore, IoU, violation matching).
- `experiments/`: Entry points for running baseline, SFT, and evaluations.
- `notebooks/`: Colab-ready `.py` scripts designed to be executed as cells for distributed training in Google Drive.
- `scripts/`: Utility scripts for setting up environments, pushing to HuggingFace, and generating paper figures.
- `tests/`: Unit tests for critical components.

## Pipeline Flow
1. **Data Loading**: Raw Construction Site dataset is loaded via HuggingFace Hub.
2. **Preprocessing**: Images and metadata are converted into Qwen conversational format containing the `UNIFIED_INSPECTION_PROMPT`.
3. **Training**: `UnslothVisionDataCollator` trains the model to produce the specific JSON schema.
4. **Evaluation**: Outputs are parsed via Pydantic (`UnifiedOutput`) and evaluated across multiple dimensions.