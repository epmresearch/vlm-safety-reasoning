# System Context & Role

You are an expert Machine Learning Engineer, Researcher and Python Architect specializing in Vision-Language Models (VLMs), PyTorch, HuggingFace TRL, Unsloth, and ARC-HPC (Slurm) deployments. You excel at building modular, robust, and scalable machine learning pipelines from first principles and identifying bugs, problems and issues in the codebase.

# Overview

I was having an existing, fully functional VLM fine-tuning pipeline (Baseline, SFT and GRPO) designed for construction safety inspection.


# Add `object_only` and `caption_only` Pipelines
Add it as only am **Add-On, Not Replacement** way. The existing "Unified Task" and "Violation Only" codebase should work 100% intact, operational, and unchanged in its behavior. we will extend the architecture, not rewrote the core.

---

## Context

You are working inside the repo at `~/vlm-safety-reasoning` (a VLM construction-safety fine-tuning project using Qwen3-VL + Unsloth on ARC/SLURM).

The project already has **two fully live pipelines**:
- `unified` (prefix `unified`) — `submit_unified_pipeline.py` + 4 `hpc_*_unified.sh` scripts
- `violations_only` (prefix `vo`) — `submit_vo_pipeline.py` + 4 `hpc_*_vo.sh` scripts

Each pipeline covers all three model tiers (`2b`/`4b`/`8b`) and four phases: `baseline ‖ sft → merge → grpo`. All inference and evaluation.

You must add **two new fully-live, parallel-safe pipelines**:
- `object_only` (prefix `oo`) — the model outputs **only** the 3 object grounding classes (`excavator`, `rebar`, `worker_with_white_hard_hat`) as bounding-box lists. No caption, no violations.
- `caption_only` (prefix `co`) — the model outputs **only** a scene caption string. No objects, no violations.

Each new pipeline must be **exactly as complete and robust**  — nothing less. Zero conflicts with any existing pipeline, and both new pipelines must be safe to run concurrently with each other and with the existing two at the same `--version` and tier.

Read `CLAUDE.md` first — it is the authoritative design document and contains critical invariants, known traps, and the full namespace isolation contract. Understand it deeply before touching a single file. This file may be helpfull but i do not know if all things in this file is perfect or not.

---

## IF you find any bugs, problems, and any issues in current codebase then fix them as well and make everything perfect.

---

## What Needs to Be Done

Read every relevant file in the codebase before touching anything. The existing `violations_only` and `unified` pipelines — understand them end to end across every layer of the stack, then mirror it exactly for `object_only` and `caption_only`.

**Do not add placeholders, stubs, or TODO comments. Every change must be production-complete and immediately runnable.**

---

### The Two New Tasks

**`object_only` (prefix `oo`):** The model outputs only the 3 object grounding classes — `excavator`, `rebar`, `worker_with_white_hard_hat` — as bounding-box lists scaled 0–1000. No caption. No violations.

**`caption_only` (prefix `co`):** The model outputs only a scene description as plain text, no json stuff. No bounding boxes. No violations.

---

### Scope of Work

Audit every layer of the stack that is task-aware and extend it for both new tasks. At minimum, this includes (this is just I think needed to do, but never only rely on this, Do not rely on me, do you own independent thining and analysis and implementation):

- **Registry / naming layer** — task constants, VALID_TASKS, TASK_PREFIXES (both new prefixes are already commented in `core/naming.py`; uncomment them).
- **Task configs** — `configs/tasks/object_only.yaml` and `configs/tasks/caption_only.yaml`, following `violations_only.yaml` structure. Choose reward components and weights appropriate to each task's sole objective: grounding IoU for `object_only`, caption quality for `caption_only`. Reason about token budgets relative to output complexity.
- **Prompts** — new prompt constants in `data/prompt_templates.py` registered in `PROMPT_REGISTRY`. Match the established style: clear instructions, bounding box scale note where relevant, ```json fence, exact format example.
- **Output schemas** — new Pydantic models in `data/schemas.py` registered in `SCHEMA_REGISTRY`. `object_only` has only the 3 grounding list fields. `caption_only` has only a required `caption` string.
- **Preprocessor routers** — new SFT target JSON builders and ground-truth dict builders in `data/preprocessor.py`, wired into the `build_target_json` and `build_gt_dict` routers. Box scaling, field inclusion, and gt dict structure must be correct for each task's eval and reward pipeline.
- **Evaluator metric gating** — `evaluation/evaluator.py` currently gates metrics with `task != "violations_only"` binary checks. Replace with explicit task-set membership so each metric family (captioning, grounding, violations, reasoning) runs for exactly the tasks that produce the required output fields. Gate the Java availability check the same way — `object_only` never needs Java.
- **Structural repair** — `preprocessing/structural_repair.py` has its own local Pydantic re-declarations (CLAUDE.md warns to keep in sync with `data/schemas.py`). Add the two new schema classes locally. Extend the task dispatch for post-repair validation. Critically: audit the repair transform pipeline and add task-based skipping so it never tries to inject or normalize fields absent from that task's schema — a clean `object_only` or `caption_only` output must never land in "still broken".
- **Reward pipeline** — verify the full chain `get_reward_funcs_for_task → _strict_parse_for_task → SCHEMA_REGISTRY` is fully generic. Fix anything hardcoded to `"unified"` or `"violations_only"`.
- **`preflight_grpo.py`** — verify every step is task-parameterized so `--task object_only` and `--task caption_only` work correctly end to end.
- **8 HPC shell scripts** — `hpc_{baseline,sft,merge_sft,grpo}_{oo,co}.sh`, exactly matching the `_vo` equivalents in structure (`set -eo pipefail`, env setup, guards, `.env` sourcing, Java PATH injection where needed). Every Python call must pass `--task` explicitly. All SBATCH names and log paths must be uniquely prefixed `oo` or `co`. The `grpo_` scripts need the merged-model existence guard. The `merge_sft_` scripts must never rely on a `--task` default. Baseline scripts must use the prefixed run_name format, not the legacy underscore form.
- **2 pipeline submitters** — `submit_oo_pipeline.py` and `submit_co_pipeline.py`, exactly mirroring `submit_vo_pipeline.py`: correct `--version` validation, log-dir creation, `merged_checkpoint_name()` usage, variant naming, resource configs, and Windows graceful degradation.
- **`CLAUDE.md`** — update the task table (both tasks live), pipeline commands section, naming table, and "Adding a new task" checklist.
- **And a lot of others that I am not telling you, that's why do not rely on me**.

---

## Isolation Guarantee — Verify Before Finishing

After all changes, trace each artifact for both new pipelines and confirm globally unique names that never conflict with `unified` or `violations_only`:
If any name collides, fix it.

---

## Testing After Implementation

Run the full test suite first:

```powershell
python -m pytest tests/ -v
```

All tests must pass. Then run targeted smoke tests:
Fix any failures before declaring done.

---

## Completion Report

When finished, output a structured report covering:

1. **Every file created or modified**, with a one-line summary of what changed and why.
2. **How `object_only` works end to end**: prompt → SFT target JSON → schema → GRPO rewards → structural repair → eval metrics → submission.
3. **How `caption_only` works end to end**:.
4. **Isolation proof**: confirm no writable path is shared between any two of the four pipelines at the same `--version` and tier.
5. **Any non-obvious design choices** and their rationale.

---

Do not guess at file contents. Read each file before modifying it.

# IMPORTANT: If you think that some codebase structure is bad and is not modular and will require to do a lot of changings at many places to run different experiment, then fix them and do them in a better way.

Note: For now do not edit any codebase, write the plan first and put any open question that you may have and i will answer them and then approve the plan.