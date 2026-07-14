# Research Journal

## Goal
To develop a computationally efficient, highly capable VLM for automated construction safety monitoring using a unified single-prompt training strategy.

## Phase 1 Updates
- Transitioned from multi-task separated models to a **unified single-prompt strategy**.
- Model: Qwen3-VL-2B-Instruct.
- Framework: Unsloth for fast, memory-efficient LoRA SFT.
- Successfully implemented JSON-constrained output structure.

## Next Steps
- Scale up to 4B and 8B models.
- Introduce GRPO (Generative Reward Policy Optimization) to align the model's reasoning process and output formatting further.
- Conduct a robust error analysis and edge-case evaluation.
