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
  --adapter_path "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/vo-sft-8b-v5/best" \
  --output_path  "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-vo-sft-8b-v5"
python -m experiments.run_grpo --tier 8b --variant vo-grpo-8b-v5 --task violations_only \
  --base_model_override "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-vo-sft-8b-v5"
python -m experiments.run_inference --tier 8b --variant vo-sft-8b-v5 --checkpoint best --task violations_only
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
python -m experiments.generate_comparison_csv --version v4   # VO-only, per-tier CSVs
python -m experiments.compare_results --tier 8b --task unified --version v5   # any task, 3-row table
python -m experiments.plot_metrics_vo --version v4
python -m experiments.extract_qualitative                    # triumph/failure examples -> markdown
python analyze_metrics.py
```

These read SFT results from `<variant>_best` and GRPO from `<variant>_final`, matching what the pipeline
now writes.

## Architecture

### Config layering

`core/config.py::load_config(task, training_kind)` merges, **last wins**:

```
base.yaml → model_registry.yaml → {sft,grpo}.yaml → tasks/<task>.yaml
```

`merge_configs` is a shallow merge, nested dicts one level deep. Task YAML is last on purpose (e.g.
`violations_only.yaml`'s `max_completion_length: 1024` beats `grpo.yaml`'s `1000`). `grpo.yaml`'s top-level
`per_device_train_batch_size` beats `model_registry.yaml`'s per-tier nested one.

Both `run_sft` and `run_grpo` now go through `load_config()`, so this precedence is real for both. (`run_sft`
previously called `load_training_config("sft")`, reading `configs/sft.yaml` alone — a task YAML could never
override an SFT hyperparameter, silently.)

`run_grpo` additionally mutates `sft_cfg` in place (`max_seq_length`, `load_in_4bit`, gradient checkpointing,
pixel bounds) — **reading `configs/sft.yaml` will not tell you what GRPO actually loaded.** Read the
`run_manifest.json` written into the checkpoint dir.

**Current training shape.** SFT: 2 epochs over 8198 augmented rows at effective batch 32 = **512 steps**, eval
every 25 steps, early stopping at patience 4. GRPO: 2 epochs over the 1732-row pool at 32 unique images per
update = **108 steps**, `save_steps: 20`. If the first SLURM log line does not say `Total steps = 512` /
`Total steps = 108`, the config did not merge as expected.

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

**One place that still does not follow the rule.** It does not collide today, but it will if copy-pasted:

- `hpc_baseline_unified.sh:59` emits `--run_name baseline_${TIER}_${VERSION}` — underscored, **no task prefix**.
  Every other pipeline uses `<prefix>-baseline-<tier>-<version>`. This legacy form is why `compare_results.py`
  and `plot_metrics.py` branch on `task == "unified"` instead of calling `task_prefix()` uniformly. A new task
  must **not** copy this; use the prefixed form.

(`hpc_merge_sft_vo.sh`'s unprefixed job name/logs and missing `--task` were fixed — it now passes
`--task violations_only` explicitly and logs to `merge_sft_vo_%j`. Always pass `--task` in a new merge script;
`merge_sft_adapter.py` still defaults to `violations_only`, so an omission looks correct while merging the
wrong task.)

All 8 `hpc_*.sh` now start with `set -eo pipefail` and a guarded `cd`. Without that, a failed training step
still ran inference/repair/eval and the job could exit 0, so the `afterok` dependency would launch the next
stage against a missing adapter. Deliberately not `set -u` — several lines legitimately expand possibly-unset
vars (`PYTHONPATH`, `SLURM_JOB_ID`).

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
5. **`steps_per_generation` must be passed explicitly to `GRPOConfig`.** If omitted, TRL defaults it to
   `gradient_accumulation_steps`, silently overriding whatever the config says. Every GRPO run before this fix
   ran at `steps_per_generation=16` regardless of the `1` in the YAML.

   Two different quantities, easy to confuse:
   - `K = unique_images_per_call = (per_device_bs × steps_per_generation) / num_generations` — how many
     *genuinely different* images share one batched tokenize call. **This is the collapse-risk number**, and
     the `--num_unique_images` argument of `scripts/test_processor_batch_collapse.py`.
   - `unique_images_per_update = (per_device_bs × gradient_accumulation_steps) / num_generations` — the
     effective RL batch. `steps_per_generation` does **not** affect it (the factors cancel).

   Current: `16 × 4 / 8 = 8` (K, in `256/64 = 4` calls per step) and `16 × 16 / 8 = 32` unique images per
   update. **K = 2, 4, 8, 16, 32 have all been verified NO COLLAPSE on real dataset images** (2026-08-31), so
   the processor no longer constrains this — generation-time KV cache does. Re-run the script for any new K,
   and raise `steps_per_generation` only with headroom evidence from `GPUMemoryLoggingCallback`: spg=16 means
   256 live sequences each carrying ~4256 vision patches, a memory profile never actually exercised (the old
   accidental spg=16 runs were prompt-only, so they are not evidence it fits).
6. **The `0.15` true-negative constant** appears in four reward files (`reward_violation_id.py:30`,
   `reward_grounding.py:26`, `reward_violation_grounding.py:35`, `reward_reasoning.py:47`). Note its stated
   rationale ("balance the EV against the 91% imbalance") is arithmetically void — the constant cancels out of
   the always-safe-vs-honest expected-value comparison at every class balance. More importantly, TRL's
   `scale_rewards` defaults to `'group'`, so advantages are z-scored within each rollout group: for a safe image
   whose 8 rollouts take only two values, the constant is **affine-invariant and changing it does nothing**.
   Retuning safe-vs-violation balance belongs in the pool composition (currently 50/50), not here. If you do
   change it, all four sites must move together — but `reward_grounding.py:26` is per-*class*, not per-image,
   and is inactive under `violations_only`.
7. **`best/` and `final/` are now different checkpoints.** `configs/sft.yaml` sets
   `load_best_model_at_end: false`, so `final/` is the literal end-of-training state (debugging) and `best/` is
   the lowest-`eval_loss` state, written eagerly by `SaveBestModelCallback`. **The merge → GRPO handoff consumes
   `best/`**, and the post-SFT eval runs `--checkpoint best` so the reported SFT numbers describe the checkpoint
   GRPO actually trains from. GRPO itself has no `best/` (no eval dataset), so `hpc_grpo_*.sh` stays on
   `--checkpoint final`. Keep `best_model_threshold: 0.0` — it is an *absolute* delta, and the old `0.005`
   froze `best/` hundreds of steps early on a ~0.055 loss.

**Violation semantics — one predicate, everywhere.** `rewards/reward_utils.py::_is_violation_present` is the
single source of truth for "is this rule violated?", used by the GRPO rewards *and* by
`evaluation/metrics_{violations,reasoning}.py`, so training and evaluation can never disagree.

**`null` is the only safe signal.** The prompt says *"If NOT violated, output null"*, so emitting a violation
object at all is an assertion of violation — even `{"reason": "", "bounding_box": []}`. A contentless
assertion still earns nothing downstream (empty boxes → IoU 0.0; empty reason → ~0), so the model gets
identification credit only. The one exception is a bare `{}`, which carries no keys and no assertion;
`structural_repair.py:1007` normalizes it to `null` first.

This matters because `preprocessing/structural_repair.py` manufactures these shapes: `:959` rewrites a bare
`true` into `{"reason":"","bounding_box":[]}`, and `:975` turns a bare reason string into a box-less
violation. Reading the first as *Safe* would invert the model's own answer — crediting an unsubstantiated
alarm as "correctly identified a safe site" — and left a reward-hacking surface where empty violation objects
collected the true-negative reward on safe images.

`rule_0` is a pseudo-rule for the safe class: **rule_0 TP = correctly said safe** (the true negative of
violation detection), **FN = false alarm on a safe image**, **FP = missed a real violation**. So
`violation_identification_recall_rule_0` = 1 − false-alarm rate, and is the over-flagging guardrail.
Parse/schema failures reach the metrics as `None`, are **never** credited rule_0 TP (they used to be, via
`{}`, silently inflating that guardrail on an ~88%-safe test set), and surface as
`violation_prediction_failure_{count,rate}`.

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
- `preprocessing/structural_repair.py` re-declares `RuleViolation`/`UnifiedOutput` locally — keep in sync with
  `data/schemas.py` by hand.
- `docs/Metrics.md` documents a metric namespace that no longer exists (`grounding_iou_all_macro_*`,
  `_excl`, `grounding_iou_total_macro`). Grepping `evaluation/` for those returns nothing — the live families
  are `grounding_{mask,greedy}_iou_{all,exist}_*`. Its analysis of `rule_0` semantics is still sound, and its
  tn0/tn1 rationale still applies to **object** grounding — but the **violation**-grounding `_tn1` keys have
  been removed (see below); only `_tn0` remains there.
- **All pre-existing GRPO metrics are void.** Those runs trained prompt-only — images never reached the model
  (fixed in `b8f2470`). `evaluation_results/` and `evaluation_results_v2/` also disagree on the same-named 8B
  run (VO SFT f1_micro 0.4883 vs 0.3537) with nothing documenting which is authoritative. Baseline and SFT
  numbers there are usable; do not draw any GRPO conclusion from them.
