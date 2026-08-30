# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project fine-tuning Qwen3-VL (2B/4B/8B via Unsloth) for construction safety inspection on the
`LouisChen15/ConstructionSite` dataset (7009 train / 3004 test). The model emits **one flat JSON object** per
image, and training is two-phase: **LoRA SFT → merge adapter → GRPO** on the merged model.

**This repo runs a family of parallel pipelines, not one pipeline.** Each *task* is a full, independent
baseline→SFT→merge→GRPO→eval pipeline over the same images and the same base model, differing only in what the
model is asked to output:

| Task | Prefix | Output | Status |
|---|---|---|---|
| `unified` | `unified` | caption + 3 object classes + 4 rule violations | live |
| `violations_only` | `vo` | 4 rule violations only | live |
| `object_only` | `oo` | object grounding only | planned (commented in `TASK_PREFIXES`) |
| `caption_only` | `co` | caption only | planned (commented in `TASK_PREFIXES`) |

Any number of these may run **concurrently on ARC** at the same `--version` and the same tier without colliding.
That isolation is a designed property, not an accident — see [Parallel-safety](#parallel-safety-the-isolation-guarantee).

## Environment

Two very different environments; know which one you're in.

| | Local (Windows dev box) | HPC (ARC / SLURM) |
|---|---|---|
| Python | 3.10.11, `.\venv\Scripts\Activate.ps1` | `module load gcc/13.3.0 python/3.12.5`, `source $HOME/envs/vlm_grpo/bin/activate` |
| Data root | `./vlm_data_root` | `VLM_DATA_ROOT=/home/$USER/vlm-finetuning-project1` |
| GPU / unsloth | none | H100/H200, `--gres=gpu:h100:1` |

**Runs locally:** the test suite, `preprocessing/structural_repair.py`, all plotting/analysis scripts, anything
importing only `core/`, `data/schemas.py`, `rewards/`. **HPC only:** `experiments/run_{sft,grpo,inference}.py`
(`run_sft.py` imports `unsloth` at line 5, before transformers — it cannot even be imported on Windows),
`merge_sft_adapter.py`, `augment_rare_classes.py`, all `scripts/hpc_*.sh`.

`requirements.txt` has loose lower bounds and is **not** the authoritative version set — the working pins live in
`scripts/setup_arc.sh` (`transformers==5.4.0`, `trl==0.23.0`, `datasets==4.3.0`, `unsloth_zoo==2026.8.12`).

There is no linter, formatter, Makefile, or CI. `.gitattributes` forces LF on `*.sh`/`*.py`/`*.yaml` to prevent
SLURM CRLF errors — don't defeat it from Windows.

## Commands

### Tests

```powershell
python -m pytest tests/ -v                                       # all (~207 tests, no GPU needed)
python -m pytest tests/test_rewards/test_unified_reward.py -v     # single file
python -m pytest tests/test_evaluation/test_output_parser.py::test_strip_fences -v   # single test
python -m pytest tests/ -k "violation and not grounding" -v
```

Use `python -m pytest`, not bare `pytest`: `tests/` has no `__init__.py` and `conftest.py` does no `sys.path`
insertion, so first-party imports (`from rewards...`) only resolve when CWD is on the path. On HPC the SBATCH
scripts export `PYTHONPATH`, so bare `pytest` works there. No pytest config file exists; no markers are registered.

### Full pipeline (HPC, from repo root on the login node)

```bash
python scripts/submit_unified_pipeline.py --tiers 2b 4b 8b --version v5
python scripts/submit_vo_pipeline.py      --tiers 8b       --version v5
```

One submitter per task; fire as many as you like back to back, same `--version`, same tiers — they are
namespace-isolated end to end (see [Parallel-safety](#parallel-safety-the-isolation-guarantee)).

`--version` is **required** and is the single source of truth for every generated name. Each submitter
pre-downloads the model on the login node, then submits 4 jobs per tier with `afterok` dependencies:
`baseline` (independent) ‖ `sft → merge → grpo`. It overrides walltime/memory on the `sbatch` command line
(grpo: `--mem=250G --time=24:00:00`), which **beats** the `--time` written in the `hpc_*.sh` files.

On Windows these submitters degrade gracefully — `sbatch` is missing, so they print the exact commands with
`DUMMY_JOB_ID`. Useful as a dry run.

### Individual stages

```bash
python -m experiments.run_sft --tier 8b --variant vo-sft-8b-v5 --task violations_only
python scripts/merge_sft_adapter.py --tier 8b --task violations_only \
  --adapter_path "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/vo-sft-8b-v5/final" \
  --output_path  "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-vo-sft-8b-v5"
python -m experiments.run_grpo --tier 8b --variant vo-grpo-8b-v5 --task violations_only \
  --base_model_override "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-vo-sft-8b-v5"
python -m experiments.run_inference --tier 8b --variant vo-sft-8b-v5 --checkpoint final --task violations_only
python preprocessing/structural_repair.py --input "$PREDS/predictions.jsonl" \
  --output "$PREDS/repair_applied/predictions_repaired.jsonl" --task violations_only
python -m experiments.run_evaluation --predictions_path "$PREDS/repair_applied/predictions_repaired.jsonl" \
  --skip_spice --task violations_only
```

`python scripts/preflight_grpo.py --tier 2b --task violations_only` runs a 6-step sanity check before burning a
GRPO job — use it.

Inference → **structural repair → evaluation** is a fixed chain; never evaluate raw `predictions.jsonl`.
Evaluation needs a JVM for METEOR/CIDEr-D (pipelines always pass `--skip_spice`; pass `--skip_java_switch`
off-Linux).

### Data prep

```bash
python -m data.augment_rare_classes     # → datasets/augmented   (SFT input; sbatch scripts/augment_data.sh)
python data/build_grpo_pool.py          # → datasets/grpo_pool   (GRPO input; no args, no --version)
```

### Local analysis (run from repo root — these use relative `Path("evaluation_results")`)

```powershell
python -m experiments.generate_comparison_csv --version v4   # current, working comparison tool
python -m experiments.plot_metrics_vo --version v4
python analyze_metrics.py
```

## Architecture

### Config layering

`core/config.py::load_config(task, training_kind)` merges, **last wins**:

```
base.yaml → model_registry.yaml → {sft,grpo}.yaml → tasks/<task>.yaml
```

`merge_configs` is a shallow merge, nested dicts one level deep. Task YAML is last on purpose (e.g.
`violations_only.yaml`'s `max_completion_length: 1024` beats `grpo.yaml`'s `1000`). `grpo.yaml`'s top-level
`per_device_train_batch_size` beats `model_registry.yaml`'s per-tier nested one.

`run_grpo` additionally mutates `sft_cfg` in place (`max_seq_length`, `load_in_4bit`, gradient checkpointing,
pixel bounds) — **reading `configs/sft.yaml` will not tell you what GRPO actually loaded.** Read the
`run_manifest.json` written into the checkpoint dir.

### Tasks — what `--task` actually controls

`--task` is threaded through every stage and is the axis that distinguishes one pipeline from another. It is a
plain argparse string, **not validated** against `VALID_TASKS` anywhere — a typo surfaces late as
`FileNotFoundError: configs/tasks/<typo>.yaml`.

| Dimension | Mechanism |
|---|---|
| Prompt | `prompt_key` in task YAML → `data/prompt_templates.py::PROMPT_REGISTRY` |
| SFT target JSON | `data/preprocessor.py::build_target_json(raw, task)` |
| GRPO/eval ground truth | `data/preprocessor.py::build_gt_dict(raw, task)` |
| Output validation schema | `data/schemas.py::SCHEMA_REGISTRY` |
| Active reward components + weights | `reward_components` / `reward_weights` in task YAML |
| Token budgets | `max_new_tokens`, `max_completion_length`, `inference_max_seq_length` in task YAML |
| Every generated name | `core/naming.py::task_prefix()` |
| Eval components run | `evaluation/evaluator.py` (e.g. `violations_only` skips captioning + grounding) |

What is deliberately **shared** across tasks: the base model, `datasets/augmented` (SFT input),
`datasets/grpo_pool` (GRPO input), and the offline data prep that builds them. `build_grpo_pool.py` is
task-blind — one pool build serves every pipeline. Task-specific formatting is applied lazily at load time by
`build_sft_dataset(..., task=)` / `build_grpo_dataset_for_task(..., task=)`. **Do not add a per-task pool.**

### Naming and versioning

`core/naming.py` holds `TASK_PREFIXES` (`unified` → `unified`, `violations_only` → `vo`) and the two name
builders. For `--version v5`, tier `8b`:

| Artifact | unified | violations_only | future `object_only` |
|---|---|---|---|
| SFT variant | `unified-sft-8b-v5` | `vo-sft-8b-v5` | `oo-sft-8b-v5` |
| Merged KL base | `merged-unified-sft-8b-v5` | `merged-vo-sft-8b-v5` | `merged-oo-sft-8b-v5` |
| GRPO variant | `unified-grpo-8b-v5` | `vo-grpo-8b-v5` | `oo-grpo-8b-v5` |
| Baseline results dir | `baseline_8b_v5` *(legacy, see below)* | `vo-baseline-8b-v5` | `oo-baseline-8b-v5` |

`merged_checkpoint_name()` must produce byte-identical strings in `submit_unified_pipeline.py`,
`submit_vo_pipeline.py`, and `run_inference.py` or GRPO silently trains against the wrong KL reference.
**Keep version tags in `v<digits>` form** — `run_inference.py` reverse-engineers the merged base with the regex
`-(v\d+)(?:_[^-]*)?$`, and a tag like `exp-a` breaks the lookup.

Paths: `$VLM_DATA_ROOT/checkpoints/qwen3vl-<tier>/<variant>/{final,best,checkpoint-N}` and
`$VLM_DATA_ROOT/results/inference/<run_name>/{predictions.jsonl, repair_applied/, evaluation_results/}`.

### Parallel-safety: the isolation guarantee

Multiple task pipelines may run concurrently on ARC at the same `--version` and tier. Nothing coordinates them at
runtime — isolation comes entirely from **every writable path being namespaced by `task_prefix(task)`**:

| Writable artifact | Namespaced by |
|---|---|
| Checkpoints, merged models | `<variant>` = `<prefix>-<phase>-<tier>-<version>` |
| Inference predictions, repairs, metrics | `results/inference/<run_name>/` |
| Oversample manifests | `oversample_manifest_<tier>_<variant>.json` |
| SLURM logs, job names | per-script (`sft_vo_%j` vs `sft_unified_%j`) |
| W&B runs | `qwen3-<tier>-<variant>-repaired` |

Everything the pipelines *share* — `datasets/{processed,augmented,grpo_pool}`, the HF model cache — is read-only
during training, so concurrent readers are safe. (The submitters pre-download models on the login node
specifically to avoid concurrent SLURM jobs racing on HF cache locks.)

**Two places that do not yet follow the rule.** Neither collides today, but both will if copy-pasted for a new task:

1. `hpc_baseline_unified.sh:59` emits `--run_name baseline_${TIER}_${VERSION}` — underscored, **no task prefix**.
   Every other pipeline uses `<prefix>-baseline-<tier>-<version>`. This legacy form is why `compare_results.py`
   and `plot_metrics.py` branch on `task == "unified"` instead of calling `task_prefix()` uniformly. A new task
   must **not** copy this; use the prefixed form.
2. `hpc_merge_sft_vo.sh` uses job name `vlm-merge-sft` and logs `merge_sft_%j` (unprefixed), and it **omits
   `--task`**, silently relying on `merge_sft_adapter.py`'s default of `violations_only`. A new task's merge
   script copy-pasted from this one would merge with the wrong task while looking correct. Always pass `--task`
   explicitly, as `hpc_merge_sft_unified.sh` does.

### Adding a new task pipeline

`core/naming.py`'s docstring claims adding a task is "ONE line here." That is true for *naming* only. The full
checklist:

1. `core/constants.py` — add to `VALID_TASKS`.
2. `core/naming.py` — uncomment/add the `TASK_PREFIXES` entry (`object_only` → `oo`, `caption_only` → `co`).
   Without it, `task_prefix()` falls back to the full task name, which still works but yields long folder names.
3. `configs/tasks/<task>.yaml` — `task_name`, `prompt_key`, `reward_components`, `reward_weights`, token budgets.
   Copy `violations_only.yaml`, **not** `unified.yaml` (whose `reward_weights` block is dead — see stale list).
4. `data/prompt_templates.py` — new prompt constant + `PROMPT_REGISTRY` entry.
5. `data/schemas.py` — new Pydantic output model + `SCHEMA_REGISTRY` entry.
6. `data/preprocessor.py:296-307` — branches in `build_target_json` and `build_gt_dict`.
7. `evaluation/evaluator.py` — gate which metric families run for the task.
8. `scripts/hpc_{baseline,sft,merge_sft,grpo}_<prefix>.sh` + `scripts/submit_<prefix>_pipeline.py` — these are
   copy-paste clones today, not parameterized. Give each a distinct `--job-name` and log path, and heed the two
   warnings above.

No new data prep is needed — the shared pool and augmented set already serve any task.

### Data flow

```
HF hub → datasets/raw → datasets/raw_cleaned → datasets/processed → datasets/augmented → SFT
                                                        └────────→ datasets/grpo_pool → GRPO
```

**The naming trap:** in `configs/base.yaml`, `processed_subdir` points at `datasets/augmented`, and the
non-augmented base is `raw_processed_subdir` → `datasets/processed`. So `data.loader.load_processed_dataset()`
returns the **augmented** set. Only `build_grpo_pool.py` reads the truly-processed one.

Augmentation (`data/augment_rare_classes.py`) is **pixel-only** — brightness/contrast, JPEG compression, gamma;
deliberately no spatial transforms, so bounding boxes and directional caption phrases stay valid. Per-rule
multiplicity is hardcoded in `main()` (rule_4 ×16, rule_2 ×12, rule_3 ×6) and **overrides the unused
`--num_augmentations` flag**. Augmented rows are identified *only* by an `_aug<N>` suffix on `image_id`; there is
no boolean column.

`build_grpo_pool.py` composes: the entire val split + every train violation image + enough random train safe
images to reach ~50/50. It reads pre-augmentation data because GRPO runs a single epoch, where near-duplicate
images would produce correlated reward groups instead of independent signal.

### Rewards and the output contract

`rewards/unified_reward.py::REWARD_COMPONENTS` is the canonical registry;
`get_reward_funcs_for_task(task)` filters and reweights it from the task YAML.

- `unified` — all 6 components at defaults (format .05, caption .15, grounding .25, violation_id .30,
  violation_grounding .15, reasoning .10).
- `violations_only` — 4 components: format .10, violation_id .40, violation_grounding .30, reasoning .20.

Every reward passes through `_strict_parse_for_task()`, which runs `parse_model_output` then Pydantic validation
and returns `None` on any exception. **`None` → `0.0` for that component.** The expected output is a *flat* JSON
object inside a ```` ```json ```` fence:

```json
{"caption":"...","rule_1_violation":{"bounding_box":[[x,y,x,y]],"reason":"..."},"rule_2_violation":null,
 "rule_3_violation":null,"rule_4_violation":null,"excavator":[[x,y,x,y]],"rebar":[],"worker_with_white_hard_hat":[]}
```

`caption` is required for `UnifiedOutput` (no default) — a missing caption zeroes **all six** rewards, including
grounding. `bounding_box` is a list *of* 4-float boxes. **Predicted boxes are scaled [0,1000]; ground truth stays
[0,1]** — rewards call `scale_1000_to_01` on predictions only. Getting this backwards silently zeroes every IoU
metric.

Prompts are defined **only** in `data/prompt_templates.py` — never hardcode one elsewhere.

## Invariants and known traps

**Never break these:**

1. **The GRPO dataset image column must be named `image` (singular).** TRL's rollout code looks for exactly that
   key; the prompt carries only a `{"type": "image"}` placeholder. Renaming it, or inlining the PIL object into
   the message dict, re-opens the model-is-blind bug. Verify with `scripts/verify_grpo_images.py`.
2. **GRPO must run against the merged SFT model** (`--base_model_override`). TRL computes its KL reference via
   `disable_adapter()`, so without the merge the reference is the raw pretrained base, not the SFT policy.
   `run_grpo.py` and `run_inference.py` both hard-refuse rather than run silently wrong;
   `--allow_unmerged_reference` is a smoke-test escape hatch only.
3. **The GRPO pool is mandatory** — no fallback. Run `python data/build_grpo_pool.py` or GRPO crashes with
   `FileNotFoundError`.
4. **Merged checkpoints need `tokenizer_name`** pointing at the original HF repo. Unsloth's processor
   auto-detection silently degrades to text-only (no `image_processor`) when loading a local `qwen3_vl` directory.
5. **Before raising `per_device_train_batch_size` or `steps_per_generation`,** run
   `scripts/test_processor_batch_collapse.py --num_unique_images K` for the resulting
   `unique_images_per_call = (per_device_bs × steps_per_generation) / num_generations`. The processor can collapse
   vision tokens for rows after the first when different images share one batched tokenize call. K=2 and K=4
   currently pass on real images. Current: `16 × 1 / 8 = 2` per call, `16 × 16 / 8 = 32` unique images per update.
6. **The `0.15` true-negative constant** appears in four reward files (`reward_violation_id.py:30`,
   `reward_grounding.py:26`, `reward_violation_grounding.py:35`, `reward_reasoning.py:47`), calibrated against the
   dataset's ~91% safe rate to defeat the "always predict safe" local minimum. Changing one without the others
   silently re-biases the objective.

**Debugging notes:**

- `_safe_reward` swallows exceptions and returns `0.0` at WARNING level — a broken reward is indistinguishable
  from a bad model. Grep SLURM stderr for `"Error in reward function"` before trusting a low score.
- OOM on GRPO is **image-size-scaled, not batch-scaled**: the failing allocation is a shape-dependent buffer tied
  to the image patch grid, so halving batch size doesn't help. The fix is `image_max_pixels` in `configs/grpo.yaml`
  (the override block in `grpo_trainer.py:107-113` is live; the key was removed from the config when training moved
  to H200 141GB). On an H100 93GB, re-add `image_max_pixels: 602112`.
- Watch `frac_reward_zero_std`: a recorded run showed 0.53, i.e. over half the unique images per update produced
  identical rewards across all 8 rollouts and contributed no gradient. `reward_format/std` is 0.0 post-SFT
  (saturated, zero gradient contribution) — that's expected, not a bug.
- Pre-warm the `all-MiniLM-L6-v2` cache (`SENTENCE_TRANSFORMERS_HOME`); the caption/reasoning rewards download it
  mid-training otherwise.

**Stale things — don't trust them:**

- `setup_project_structure.py` — dead one-time bootstrap describing an older layout. Gitignored yet tracked. Don't run it.
- `configs/tasks/unified.yaml`'s `reward_weights` uses an obsolete key namespace (`json_validity`,
  `rule_violation_accuracy`, …) that matches nothing in the registry. Unknown *component names* raise, but unknown
  *weight keys* fail **silently** — those five values are ignored entirely.
- `rewards/{json_validity,caption_quality,rule_violation_accuracy,grounding_iou}.py` are legacy and unwired;
  `rule_violation_accuracy` still returns 1.0 for both-empty, the exact hack the 0.15 constant replaced.
- `experiments/compare_results.py` reads a nested `metrics.json` shape that `run_evaluation.py` no longer writes;
  it produces a table of `None`s. Use `generate_comparison_csv.py`.
- `tests/test_grpo/test_grpo_config.py::test_grpo_overrides_model_registry_batch_size` asserts batch size 8; HEAD
  is 16. This test fails today.
- `preprocessing/structural_repair.py` re-declares `RuleViolation`/`UnifiedOutput` locally — keep in sync with
  `data/schemas.py` by hand.
- `evaluation_results/` and `evaluation_results_v2/` are different 8B runs with materially different numbers
  (VO SFT f1_micro 0.4883 vs 0.3537), both copied back from HPC. Nothing documents which is authoritative.
