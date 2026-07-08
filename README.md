# VLM Safety Fine-Tuning (Project 1)

Fine-tunes vision-language models on the ConstructionSite 10k dataset
(rule violation identification, image captioning, visual grounding,
attribute classification), then applies GSPO/GRPO reinforcement learning
on top of SFT. Produces a Base vs SFT vs SFT+GSPO comparison study.

## Setup (local dev)

```bash
git clone <your-repo-url>
cd vlm-safety-finetuning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your real tokens locally if testing outside Colab
```

## Setup (Colab)

1. Mount Drive and run `scripts/setup_colab.sh` (clones/pulls repo, installs deps, copies secrets).
2. Run `scripts/setup_drive_structure.py <drive_root>` once, if not already done.
3. Use notebooks in `notebooks/` as thin wrappers around `experiments/*.py`.

## Running an experiment

```bash
python experiments/run_baseline.py --task rule_violation --model_id qwen25-7b-base
python experiments/run_sft.py --task rule_violation --model_id qwen25-7b-base
python experiments/run_grpo.py --task rule_violation --model_id qwen25-7b-sft-v1
python experiments/compare_results.py --task rule_violation
```

## Structure

See `docs/architecture.md` for how data → model → reward → evaluation connect.

## Model strategy

Controlled by `configs/model_strategy.yaml`: either `multi_size` (same task,
multiple model sizes) or `task_specialized` (one model per task). Changing
this file changes what `models/registry.py` builds — no code changes needed.