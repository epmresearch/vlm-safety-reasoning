# Experiment Log

## Phase 1: Unified Prompt Refactoring
- **Date:** (Current)
- **Goal:** Shift from multi-task pipelines to a single unified JSON generation prompt.
- **Model:** `unsloth/Qwen3-VL-2B-Instruct`
- **Result:** Successfully implemented the unified training and evaluation pipeline. Structural checks confirm the model can be guided to output robust JSON matching a predefined Pydantic schema.

## Next Phase
- Run `baseline` evaluation for 2B, 4B, 8B on the 250 validation samples.
- Execute SFT `unified-sft-v1` for 2B, 4B, 8B.
- Aggregate metrics using `experiments/compare_results.py` and `scripts/generate_figures.py`.