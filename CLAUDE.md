# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project fine-tuning Qwen3-VL (2B/4B/8B via Unsloth) for construction safety inspection on the
`LouisChen15/ConstructionSite` dataset (7009 train / 3004 test). Training is two-phase:
**LoRA SFT → merge adapter → GRPO** on the merged model.

**This repo runs a family of parallel pipelines, not one pipeline.** Each *task* is a full, independent
baseline→SFT→merge→GRPO→eval pipeline over the same images and the same base model, differing only in what the
model is asked to output. All four are live:

| Task | Prefix | Output | Wire format | Capabilities |
|---|---|---|---|---|
| `unified` | `unified` | caption + 3 object classes + 4 rule violations | fenced JSON | caption, objects, violations |
| `violations_only` | `vo` | 4 rule violations only | fenced JSON | violations |
| `object_only` | `oo` | 3 object classes only, boxes `[0,1000]` | fenced JSON | objects |
| `caption_only` | `co` | one scene description | **bare prose** (no JSON, no fence) | caption |

Three of the four emit **one flat JSON object** per image. `caption_only` is the exception: the caption *is* the
entire output, so wrapping it in JSON would add a formatting confound to the exact quantity being measured. Its
completion is bare prose, parsed by `evaluation/output_parser.py::parse_output_for_task`, which wraps it into
`{"caption": ...}` so every downstream layer stays dict-shaped.

Any number of these may run **concurrently on ARC** at the same `--version` and the same tier without colliding.
That isolation is a designed property, not an accident — see [Parallel-safety](#parallel-safety-the-isolation-guarantee),
and it is asserted by `tests/test_core/test_name_isolation.py`.

**`core/tasks.py` is the single place a task is registered.** One frozen `TaskSpec` per task carries its name,
prefix, *capability set* (`caption` / `objects` / `violations`) and wire format; `VALID_TASKS`, `TASK_PREFIXES`
and every metric/repair gate in the repo derive from it. No conditional anywhere should compare a task name to a
literal string — ask a capability question (`task_has(task, CAP_OBJECTS)`) instead. The old style was a set of
binary negations (`if task != "violations_only":`), each of which silently handed a brand-new task the *unified*
behaviour.

## Environment

Two very different environments; know which one you're in.

| | Local (Windows dev box) | HPC (ARC / SLURM) |
|---|---|---|
| Python | 3.10.11, `.\venv\Scripts\Activate.ps1` | `module load gcc/13.3.0 python/3.12.5`, `source $HOME/envs/vlm_grpo/bin/activate` |
| Data root | `./vlm_data_root` | `VLM_DATA_ROOT=/home/$USER/vlm-finetuning-project1` |
| GPU / unsloth | none | **H200**, `--partition=gpu-h100 --gres=gpu:h200:1` |

**Runs locally:** the test suite, `preprocessing/structural_repair.py`, all plotting/analysis scripts, anything
importing only `core/`, `data/schemas.py`, `rewards/`. **HPC only:** `experiments/run_{sft,grpo,inference}.py`
(`run_sft.py` imports `unsloth` at line 5, before transformers — it cannot even be imported on Windows),
`merge_sft_adapter.py`, `augment_rare_classes.py`, all `scripts/hpc_*.sh`. The four `submit_*_pipeline.py`
wrappers and `scripts/submit_pipeline.py` run locally as a dry run (no `sbatch` → they print the exact commands).

`requirements.txt` has loose lower bounds and is **not** the authoritative version set — the working pins live in
`scripts/setup_arc.sh` (`transformers==5.4.0`, `trl==0.23.0`, `datasets==4.3.0`, `unsloth_zoo==2026.8.12`).

There is no linter, formatter, Makefile, or CI. `.gitattributes` forces LF on `*.sh`/`*.py`/`*.yaml` to prevent
SLURM CRLF errors — don't defeat it from Windows.

## Commands

### Tests

```powershell
python -m pytest tests/ -v                                       # all (~501 tests, no GPU needed)
python -m pytest tests/test_core -v                               # task registry + name-isolation proof
python -m pytest tests/test_rewards/test_unified_reward.py -v     # single file
python -m pytest tests/test_evaluation/test_output_parser.py::test_strip_fences -v   # single test
python -m pytest tests/ -k "violation and not grounding" -v
python -m pytest tests/ -k "_oo or _co" -v                        # just the two new pipelines
python -m pytest tests/test_core/test_blocker_fixes.py -v         # the six pre-flight blockers
python -m pytest tests/test_core/test_ledger_fixes.py -v          # BUG-08..29
```

**Reward-surface validator (no GPU, run before every submit).**

```powershell
python scripts/validate_rewards.py                       # all four tasks, probe + census
python scripts/validate_rewards.py --task object_only --probe
python scripts/validate_rewards.py --task object_only --probe --pool-stats   # on ARC
python scripts/validate_rewards.py --task object_only --sft-stats            # on ARC
```

`scripts/validate_rewards.py` scores synthetic honest and degenerate policies against real
ground truth and **fails** if any degenerate policy beats the honest one, if any object class's
break-even IoU exceeds 0.75, or if unconditional rule_1 assertion beats honest abstention on
pool expected value. It is the cheapest check in the repo and it is the one that catches
reward hacking before a GRPO job burns GPU hours. Its `--census` half tokenizes every SFT
target and fails if any would truncate at `max_seq_length`.

Use `python -m pytest`, not bare `pytest`: `tests/` has no `__init__.py` and `conftest.py` does no `sys.path`
insertion, so first-party imports (`from rewards...`) only resolve when CWD is on the path. On HPC the SBATCH
scripts export `PYTHONPATH`, so bare `pytest` works there. No pytest config file exists; no markers are registered.

### Full pipeline (HPC, from repo root on the login node)

```bash
python scripts/submit_pipeline.py --task unified         --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task object_only     --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task caption_only    --tiers 2b 4b 8b --version v1
```

`scripts/submit_pipeline.py` is the one submitter. The four per-task wrappers are equivalent 3-line shims kept
as stable entry points:

```bash
python scripts/submit_unified_pipeline.py --tiers 8b --version v1   # == --task unified
python scripts/submit_vo_pipeline.py      --tiers 8b --version v1   # == --task violations_only
python scripts/submit_oo_pipeline.py      --tiers 8b --version v1   # == --task object_only
python scripts/submit_co_pipeline.py      --tiers 8b --version v1   # == --task caption_only
```

Fire as many as you like back to back, same `--version`, same tiers — they are namespace-isolated end to end
(see [Parallel-safety](#parallel-safety-the-isolation-guarantee)).

`--version` is **required**, must match `v<digits>`, and is the single source of truth for every generated name.
The submitter pre-downloads the model on the login node (`--skip-preload` to opt out when the HF cache is
already warm), then submits 4 jobs per tier with `afterok` dependencies: `baseline` (independent) ‖
`sft → merge → grpo`.

There are only **four** phase scripts, all task-parameterized — `scripts/hpc_{baseline,sft,merge_sft,grpo}.sh`,
taking the task as their first positional argument. They replaced 8 per-task clones. Because `#SBATCH`
directives cannot read arguments, the submitter passes `--job-name`, `--output`, `--error`, `--mem` and `--time`
on the `sbatch` command line, all of which **beat** the in-file directives. So each task still gets its own job
names (`vlm-sft-oo`) and log files (`sft_oo_%j.out`) — the same names as before.

On Windows these submitters degrade gracefully — `sbatch` is missing, so they print the exact commands with
`DUMMY_JOB_ID`. Useful as a dry run, and the fastest way to eyeball that two tasks' paths do not overlap.

### Individual stages

```bash
python -m experiments.run_sft --tier 8b --variant oo-sft-8b-v1 --task object_only
python scripts/merge_sft_adapter.py --tier 8b --task object_only \
  --adapter_path "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/oo-sft-8b-v1/best" \
  --output_path  "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-oo-sft-8b-v1"
python -m experiments.run_grpo --tier 8b --variant oo-grpo-8b-v1 --task object_only \
  --base_model_override "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-oo-sft-8b-v1"
python -m experiments.run_inference --tier 8b --variant oo-sft-8b-v1 --checkpoint best --task object_only
python preprocessing/structural_repair.py --input "$PREDS/predictions.jsonl" \
  --output "$PREDS/repair_applied/predictions_repaired.jsonl" --task object_only
python -m experiments.run_evaluation --predictions_path "$PREDS/repair_applied/predictions_repaired.jsonl" \
  --skip_spice --task object_only
```

`--task` now carries `choices=VALID_TASKS` at every entry point, so a typo fails immediately instead of
surfacing later as `FileNotFoundError: configs/tasks/<typo>.yaml`. `--variant` is **required** on
`run_sft`/`run_grpo` (the old `unified-sft-v4` defaults silently produced stale, unversioned runs), and
`merge_sft_adapter.py --task` is **required** (it used to default to `violations_only`, which was right for
exactly one caller and silently wrong for every other).

`python scripts/preflight_grpo.py --tier 2b --task object_only` runs the sanity check before burning a GRPO job
— use it. It now also assembles the task's real reward functions, so a typo in a task YAML's
`reward_components` is caught here rather than mid-training, and it measures the text-only prompt length for
*that* task instead of the old hardcoded 233-token constant (which was measured for the unified/VO prompt).

**Inference has no auto-resume and no per-batch retry.** `run_inference_batched` runs the split start to
finish, truncates `predictions.jsonl` on every run (opened once in `"w"` mode), and lets a failing batch crash
the job. Both features existed for Colab and on SLURM combined into silent metric corruption: a caught batch
failure was written as `raw_output: ""`, resume re-ran exactly those images because it only counted non-empty
outputs as complete, and the append-mode write left two records for one `image_id` — inflating every metric's
denominator with spurious failures. Re-running a stage now re-does the whole split, which is what you want on a
cluster. `structural_total_samples_count` should always equal the split size (3004 for test); anything else
means something is wrong. SFT/GRPO checkpoint resume is unaffected and still enabled.

Inference → **structural repair → evaluation** is a fixed chain; never evaluate raw `predictions.jsonl`.
Evaluation needs a JVM for METEOR/CIDEr-D **only for tasks that score text** (`caption` or `violations`
capability); `object_only` needs no JRE. Pipelines always pass `--skip_spice`; pass `--skip_java_switch`
off-Linux.

### Data prep

```bash
python -m data.augment_rare_classes     # → datasets/augmented   (SFT input; sbatch scripts/augment_data.sh)
python data/build_grpo_pool.py          # → datasets/grpo_pool   (GRPO input; no args, no --version)
```

### Local analysis (run from repo root — these use relative `Path("evaluation_results")`)

```powershell
python -m experiments.compare_results --tier 8b --task object_only --version v1   # any task, 3-row table
python -m experiments.plot_metrics --tier 8b --task object_only --version v1      # capability-gated plots
python -m experiments.generate_comparison_csv --task violations_only --version v1 # per-tier CSVs
python -m experiments.plot_metrics_vo --task violations_only --version v1         # violation-only 3x3 suite
python -m experiments.extract_qualitative --task violations_only --tier 8b --version v1
```

**First, materialise the local layout.** The pipeline writes
`results/inference/<run>/evaluation_results/metrics.json` on ARC, but `plot_metrics.py`,
`plot_metrics_vo.py`, `generate_comparison_csv.py` and `extract_qualitative.py` all read a flat
`evaluation_results/<run>/metrics.json`. Nothing used to create that, so those four silently found
nothing. `scripts/fetch_results.py` is the missing step:

```powershell
python scripts/fetch_results.py --task object_only --version v1 --tiers 2b 4b 8b
python scripts/fetch_results.py --task object_only --version v1 --source ./arc_results --dry-run
```

Its docstring carries the `rsync` command for pulling the tree off the cluster. `compare_results.py`
reads the ARC layout directly and needs no fetch.

All four resolve result folders through `core/naming.py::results_dir_names(task, tier, version)` — one helper,
so no two of them can disagree about a folder name. SFT results come from `<variant>_best` and GRPO from
`<variant>_final`. `plot_metrics.py` draws only the plot families the task's capabilities support;
`plot_metrics_vo.py` is violation-specific and refuses a task without that capability.

## Architecture

### Config layering

`core/config.py::load_config(task, training_kind)` merges, **last wins**:

```
base.yaml → model_registry.yaml → {sft,grpo}.yaml → tasks/<task>.yaml
```

`merge_configs` is a shallow merge, nested dicts one level deep. Task YAML is last on purpose (e.g.
`object_only.yaml`'s `max_completion_length: 768` beats `grpo.yaml`'s `1000`). `grpo.yaml`'s top-level
`per_device_train_batch_size` beats `model_registry.yaml`'s per-tier nested one. **`configs/sft.yaml` is not in
the GRPO chain** — which is what made the third row below a ghost variable.

Both `run_sft` and `run_grpo` pass the merged dict to their trainer, so this precedence is real for both. It
was not always. Three keys used to say one thing while the runtime read another, with no log line and no
error; all three are fixed and pinned by `tests/test_core/test_blocker_fixes.py`.

| Config said | What actually ran | Now |
|---|---|---|
| SFT LR clamped `4b → 5e-5`, `8b → 2e-5`, with a log line confirming it | `1.0e-4` at **every** tier — `run_sft.py` built `sft_cfg` and never passed it, so the trainer re-read `sft.yaml` alone | Clamp **deleted**; flat `1.0e-4`. Task YAMLs can now override an SFT key at all |
| `image_max_pixels: 1204224` (a 1.2 MP cap) | Uncapped, up to 14.6 MP — `apply_pixel_bounds` wrote a key shape transformers rejects | Writes `{"shortest_edge","longest_edge"}`: a **key rename, not a unit conversion** — those keys hold pixel *areas*. A sqrt would cap area at 1097 px² |
| GRPO adapter shape from `sft.yaml` | From `model_loader.py`'s own literals — `lora` and all four `finetune_*` switches were absent from GRPO's merged config | `grpo.yaml` declares them explicitly. Identical values, so behaviour is unchanged by construction |

The third one broke nothing (the literals matched) but `finetune_vision_layers: false` — freezing the vision
tower for a vision-grounding RL phase — is a research decision that was being made by a Python default. A
parametrized test now asserts every key `model_loader.py` consumes is present in the merged config for **both**
training kinds and all four tasks.

**Why SFT's learning rate is flat across tiers.** A 5× LR spread confounds the scale comparison the three
tiers exist to make: an 8b regression would be unattributable. 512 steps cannot absorb it either — at 1e-4
eval loss still improved to ~step 250, so 2e-5 would stop mid-descent and be reported as the model's ceiling.
And per-tier tapering is a full-fine-tuning instinct; LoRA is far less LR-sensitive to scale (QLoRA tapered 2×
across a 9× parameter range) and 1e-4 is already stable at 8b here. A per-tier LR is still allowed, but it
belongs in `model_registry.yaml`'s tier block as declared configuration, never as a hidden override.
`lora.alpha: 16` at `r: 16` is left alone for the same comparability reason — it is the validated operating
point and the loss curve shows no underfitting. Raise `r` for capacity, as its own ablation.

**Why GRPO's learning rate is `2.0e-6`.** Cumulative LR ("mass") over 108 steps with warmup 0.05 + cosine,
against the 512-step SFT run at 1e-4 (mass 2.57e-2). The last column discounts by the recorded
`frac_reward_zero_std = 0.53` — over half the unique images per update produce identical rewards across all 8
rollouts, so their group-normalised advantages are exactly 0:

| peak LR | mass | vs SFT | gradient-weighted |
|---|---|---|---|
| `2.0e-7` (old) | 1.10e-5 | 1/2336 | 1/4970 |
| `1.0e-6` | 5.50e-5 | 1/467 | 1/994 |
| **`2.0e-6`** | **1.10e-4** | **1/234** | **1/497** |

**108 steps is the binding constraint** — a low LR is only conservative if there is runway to accumulate it.
Safe because three brakes are already tight: `max_grad_norm: 0.3` caps the per-step update regardless of LR,
`beta: 0.04` penalises KL drift from the merged reference, and `scale_rewards="group"` stops reward magnitude
inflating step size. Chosen on **failure asymmetry**: too high fails loudly (reward collapse, KL blowup, visibly
degenerate output); too low fails *silently*, as flat metrics indistinguishable from "GRPO does not help this
task" — the ambiguity that voided the pre-`b8f2470` runs. **Verify on the 2b smoke run before the other 11:**
`reward/mean` rising and `objective/kl` off zero → proceed; both flat across all 108 steps → 5e-6; reward
rising while output degenerates → 1e-6.

**Three length keys, three different jobs.** Easy to conflate, and one of them was a real bug (BUG-13):

| Key | Consumed by | Bounds |
|---|---|---|
| `max_seq_length` | `FastVisionModel.from_pretrained` — **both** SFT and GRPO | the model **load** window: a ceiling, not an allocation |
| `SFTConfig.max_length` | HF Trainer, SFT only | prompt **+** target as one sequence, vision tokens included |
| `max_prompt_length` / `max_completion_length` | `GRPOConfig`, GRPO only | the prompt (incl. vision tokens), and **one** rollout's output |

So SFT and GRPO both pass `max_seq_length` to the loader; they differ in what bounds the *training* sequence,
because SFT trains on one concatenated sequence while GRPO generates prompt and completion in separate phases.
`max_seq_length` was previously passed to `SFTTrainer` as a kwarg it does not accept, so the configured 2048
was dropped and the real ceiling was `max_length`'s default of **1024** — below the observed 1865 max.

Two things that regularly trip people up here:

- **`max_completion_length` is per rollout, not per group.** Each of the 8 rollouts gets the full budget; it is
  not divided among them and it is not a total. It is a ceiling on generation, not a preallocation — HF
  `generate` grows its KV cache dynamically, so raising it costs nothing until a rollout actually runs that
  long. Its real job is bounding the damage from a degenerate repeating generation.
- **Vision tokens count inside `max_prompt_length`.** The `{"type": "image"}` placeholder expands to ~1176–1270
  real tokens at the 1.2 MP cap, which is why the measured worst-case prompt is **1519** tokens (~233 text +
  ~1270 vision), not ~233. `scripts/validate_rewards.py --census` measures **text only** and says so — add the
  vision tokens before comparing any census number to a ceiling.

`run_grpo` additionally mutates `sft_cfg` in place (`max_seq_length`, `load_in_4bit`, gradient checkpointing,
pixel bounds) — **reading `configs/sft.yaml` will not tell you what GRPO actually loaded.** Read the
`run_manifest.json` written into the checkpoint dir.

**Current training shape.** SFT for `unified` / `violations_only`: 2 epochs over 8198 augmented rows at
effective batch 32 = **512 steps**, eval every 25 steps, early stopping at patience 4. SFT for `object_only` /
`caption_only`: the same, over the un-augmented split instead (`sft_dataset_subdir`), so ~**394 steps** — the job
logs the actual train/val sizes at startup. GRPO, all four tasks: 2 epochs over the 1732-row shared pool at 32
unique images per update = **108 steps**, `save_steps: 20`. If the first SLURM log line disagrees with the
expected step count, the config did not merge as expected.

**Live token budgets.** `max_new_tokens` is kept equal to `max_completion_length` per task, so inference can
never truncate an output GRPO trained the policy to produce. `max_prompt_length` is **shared** — it lives once
in `grpo.yaml` and is deliberately *not* pinned per task, because it caps the prompt, which is not a per-task
quantity; only the output length is.

Prompt sizes below are text + ~1270 vision tokens at the 1.2 MP cap. They grew when the prompts took on the
paper's rule wording and the reason-style guidance, which is what forced `max_prompt_length` up to 2304 and
`max_seq_length` to 3600.

| Task | completion | prompt (text + vision) | GRPO 2304 + completion | inference window | inference prompt cap |
|---|---|---|---|---|---|
| `unified` | 1024 | ~540 + 1270 = ~1810 | 3328 | 3200 | 2176 |
| `violations_only` | 1024 | ~390 + 1270 = ~1660 | 3328 | 3200 | 2176 |
| `object_only` | 768 | ~265 + 1270 = ~1535 | 3072 | 2688 | 1920 |
| `caption_only` | 768 | ~215 + 1270 = ~1485 | 3072 | 2944 | 2176 |

Worst case is 3328 against `max_seq_length: 3600`. `unified` carries the longest prompt of the four yet used to
have the second-*smallest* inference window (2816), so its prompt cap was 1792 against an ~1810-token prompt —
it would have silently truncated. Its window is now matched to `violations_only`.

A completion that truncates mid-JSON fails the parse, which zeroes **every** reward component — so a
truncation is indistinguishable from a terrible model. Confirm real target lengths with
`scripts/validate_rewards.py --census` (remembering it reports text only) before lowering any of these.

### Tasks — what `--task` actually controls

`--task` is threaded through every stage and is the axis that distinguishes one pipeline from another. It is
**validated** in three places, so a typo fails immediately rather than surfacing hours later: `choices=VALID_TASKS`
at every entry point, an explicit `validate_task` call inside each generic SLURM script before any work starts,
and `core/config.py::load_task_config`, which every stage funnels through and which now raises a `ValueError`
naming the registry instead of a `FileNotFoundError` on `configs/tasks/<typo>.yaml`.

| Dimension | Mechanism |
|---|---|
| Registration, prefix, capabilities, wire format | `core/tasks.py::TASK_REGISTRY` |
| Prompt | `prompt_key` in task YAML → `data/prompt_templates.py::PROMPT_REGISTRY` |
| Raw-completion parsing | `evaluation/output_parser.py::parse_output_for_task` (JSON vs bare prose) |
| SFT target | `data/preprocessor.py::build_target_json(raw, task)` — dispatch table, raises on unknown |
| GRPO/eval ground truth | `data/preprocessor.py::build_gt_dict(raw, task)` — dispatch table |
| Output validation schema | `data/schemas.py::SCHEMA_REGISTRY` |
| Active reward components + weights | `reward_components` / `reward_weights` in task YAML |
| Token budgets | `max_new_tokens`, `max_completion_length`, `inference_max_seq_length` in task YAML |
| SFT input dataset | `sft_dataset_subdir` in task YAML (absent ⇒ the shared default) |
| Every generated name | `core/naming.py` (`variant_name`, `merged_checkpoint_name`, `baseline_run_name`, `results_dir_names`, `slurm_job_name`, `slurm_log_stem`) |
| Eval metric families | `evaluation/evaluator.py`, gated on **capabilities** |
| Structural repair transforms | `preprocessing/structural_repair.py`, gated on **capabilities** |

**Capability gating.** Three capabilities map onto the output field groups, the reward components and the
metric families:

| Capability | Fields | Rewards | Eval metrics | Tasks |
|---|---|---|---|---|
| `caption` | `caption` | `reward_caption` | captioning (needs a JVM + images) | `unified`, `caption_only` |
| `objects` | `excavator`, `rebar`, `worker_with_white_hard_hat` | `reward_grounding` | grounding | `unified`, `object_only` |
| `violations` | `rule_1..4_violation` | `reward_violation_id`, `reward_violation_grounding`, `reward_reasoning` | violations + reasoning (needs a JVM + images) | `unified`, `violations_only` |

`reward_format` is active for every task, but means different things: schema-valid fenced JSON for the three
JSON tasks, *clean prose* (no fence, no JSON object, no `"caption":` label, non-blank) for `caption_only`.
Without that second meaning the format reward would be free — any non-empty string parses.

Java is required **only** when a task scores text, i.e. has `caption` or `violations`. `object_only` evaluates
with no JRE and no images — and, since BUG-17, does not even build the test-split image map.

**Caption metric semantics.** Blank predictions are **excluded** from the graders and reported as
`captioning_blank_prediction_rate` / `_count` / `captioning_scored_count`, mirroring how
`violation_prediction_failure_rate` already works. They used to be rewritten to the literal string
`"empty"` and scored, so a completely failed generation earned a real nonzero BERTScore and the
failure was invisible in the caption metrics. Read the pair together: `captioning_bertscore_f1` is
quality *given a caption was produced*, `captioning_blank_prediction_rate` is how often one wasn't.

**Two CLIPScores.** `captioning_clipscore` is the standard number, kept for historical
comparability. It silently truncates text at **77 tokens** — roughly 55 words against a 48.5-word
average reference, which is why it was flat baseline-vs-SFT (0.7781 → 0.7732) while every
content-overlap metric moved sharply.

`captioning_long_clipscore` reads the **whole** caption by chunking it into ≤75-content-token
windows, scoring each against the image with the *same* encoder and the *same* `2.5·max(0, cos)`
formula, and averaging. Because the model and formula are identical, the two numbers sit on one scale
and the gap between them is precisely the part standard CLIPScore never read — verified: a caption
that fits scores identically to 1e-4, while a 132-word caption reads 0.3687 standard vs 0.4150
chunked. Companion keys `long_clipscore_truncated_caption_count` and
`long_clipscore_avg_chunks_per_caption` tell you how often the 77-token limit was actually binding;
if avg_chunks ≈ 1.0 the two scores should agree and truncation was never an issue on that data.

Its limitation, stated plainly: chunk-mean cannot see coherence *across* windows. It answers "is
every part of this caption supported by the image?", not "is this the best whole-caption
description".

**Why not a long-context CLIP variant.** `jinaai/jina-clip-v1` was tried first and is incompatible
with `transformers==5.4.0`: its remote `eva_model.py` calls `.item()` on a tensor built under
transformers' meta-device init, raising `Tensor.item() cannot be called on meta tensors`. No
`from_pretrained` flag disables that init, and `low_cpu_mem_usage=False` does not help. Chunking needs
**no new dependency, no extra download, and no third-party remote code**, so it cannot break that way.
`VLM_DISABLE_LONG_CLIP=1` turns it off; the test suite sets that.

What is deliberately **shared** across tasks: the base model, `datasets/grpo_pool` (GRPO input, for every task),
and the offline data prep that builds it. `build_grpo_pool.py` is task-blind — one pool build serves every
pipeline. Task-specific formatting is applied lazily at load time by `build_sft_dataset(..., task=)` /
`build_grpo_dataset_for_task(..., task=)`. **Do not add a per-task pool.**

**SFT input is the one thing that is not fully shared.** `unified` and `violations_only` read the default
(`base.yaml`'s `processed_subdir` → `datasets/augmented`). `object_only` and `caption_only` set
`sft_dataset_subdir: datasets/processed` in their task YAML, because the augmentation duplicates images by rare
*violation* rule (rule_4 ×16, rule_2 ×12, rule_3 ×6) and those duplicates carry identical boxes and identical
captions — no class rebalancing for those tasks, only overfitting pressure. Augmentation only touches the
**train** split, so val and test are byte-identical between the two roots and all four tasks are still evaluated
on exactly the same test images. For the same reason `run_sft.py` skips rare-rule **oversampling** for tasks
without the `violations` capability — it is defined purely by which violation rules a sample trips. Expect ~394
SFT steps for `oo`/`co` versus 512 for `unified`/`vo`; the first log line reports the actual train/val sizes.

**Stratified sampling is a separate question from oversampling, and applies more widely.** Oversampling changes
*how often* a row is seen; the sampler changes only *when* — every index still appears exactly once per epoch.
`data/oversampling.py::build_rare_mask_for_task` picks the axis from the task's **capabilities**:

| Task | rare axis | why |
|---|---|---|
| `unified`, `violations_only` | rules 2/3/4 | unchanged from the legacy `build_rare_mask` |
| `object_only` | `rebar` or `worker_with_white_hard_hat` | the two hard classes; excavator is excluded — at 2415 train occurrences vs 846 and 680 it would mark most images rare and degenerate to a plain shuffle |
| `caption_only` | `None` → plain shuffle | there is no rare caption |

`unified` deliberately stratifies on **violations**, not objects: its violation components carry 0.55 of its
reward weight against 0.25 for grounding, and stratifying two axes at once over-constrains the ordering with no
clear winner.

**Why `object_only` gets it at all.** For a class present in a fraction *p* of images, the chance a batch of 32
contains none of it is `(1-p)^32` — 80% at *p* = 0.7% (un-augmented rule_4), 8% at 8%, 1.7% at 12%. Those
starved steps contribute no gradient for that class. The magnitude for objects is far smaller than for
violations, but stratifying when it is *not* needed costs nothing, while not stratifying when it *is* leaves
batches with no signal for a class — so the asymmetry decides it. Measured on a synthetic split at realistic
prevalence the sampler cut rare-rows-per-batch spread from **sd 2.21 to sd 0.54** with the mean unchanged.

Measure the real incidence before drawing conclusions — the paper cannot answer it, because Table 4 counts box
*occurrences* (rebar 846) rather than images containing at least one:

```bash
python scripts/validate_rewards.py --sft-stats --task object_only    # needs the dataset; run on ARC
```

**Honest limitation:** the mask is a single boolean, so it spreads "contains a rare class" evenly rather than
guaranteeing each class appears in every batch. At 8% prevalence `worker_with_white_hard_hat` would still be
absent from ~8% of batches. A per-class guarantee needs a genuinely multi-label sampler, which buys ~8% more
exposure of one class for real added complexity — and the class-imbalance problem it would address is already
handled in the reward by `grounding_tn_constant`.

Per-epoch reshuffling works because `models/sft_trainer.py` forwards HF's `set_epoch` onto the sampler
(`dataloader.set_epoch = sampler.set_epoch`). `get_train_dataloader()` is called once before the epoch loop, so
without that forward the sampler's epoch would stay pinned and **every epoch would replay byte-identical
order** — invisible at 1 epoch, silently wasting the second. Both are pinned by
`tests/test_data/test_rare_mask_for_task.py`.

### Naming and versioning

`core/naming.py` builds every name from the prefixes in `core/tasks.py`. `TASK_PREFIXES` is *derived* — do not
edit it, add a `TaskSpec`. For `--version v1`, tier `8b`:

| Artifact | unified | violations_only | object_only | caption_only |
|---|---|---|---|---|
| SFT variant | `unified-sft-8b-v1` | `vo-sft-8b-v1` | `oo-sft-8b-v1` | `co-sft-8b-v1` |
| Merged KL base | `merged-unified-sft-8b-v1` | `merged-vo-sft-8b-v1` | `merged-oo-sft-8b-v1` | `merged-co-sft-8b-v1` |
| GRPO variant | `unified-grpo-8b-v1` | `vo-grpo-8b-v1` | `oo-grpo-8b-v1` | `co-grpo-8b-v1` |
| Baseline results dir | `unified-baseline-8b-v1` | `vo-baseline-8b-v1` | `oo-baseline-8b-v1` | `co-baseline-8b-v1` |
| SLURM job / log stem | `vlm-sft-unified` / `sft_unified` | `vlm-sft-vo` / `sft_vo` | `vlm-sft-oo` / `sft_oo` | `vlm-sft-co` / `sft_co` |

Baseline naming is now **uniform**. `hpc_baseline_unified.sh` used to emit the legacy unprefixed
`baseline_<tier>_<version>` — the one writable path in the repo not namespaced by task, and the reason
`compare_results.py` and `plot_metrics.py` each carried an `if task == "unified"` branch. Both branches are
gone; every lookup goes through `results_dir_names()`. Any pre-existing `results/inference/baseline_<tier>_<v>/`
folder on ARC will no longer be found — versioning restarted at `v1` for exactly this reason.

`merged_checkpoint_name()` must produce byte-identical strings in `submit_pipeline.py` and `run_inference.py`
or GRPO silently trains against the wrong KL reference; both now call the same helper.
**Keep version tags in `v<digits>` form** — `run_inference.py` reverse-engineers the merged base with the regex
`-(v\d+)(?:_[^-]*)?$`, and a tag like `exp-a` breaks the lookup. `submit_pipeline.py` refuses any other form
up front, and `tests/test_core/test_name_isolation.py` asserts the round-trip for every task.

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
| SLURM logs, job names | `slurm_log_stem()` / `slurm_job_name()` → `sft_oo_%j`, `vlm-sft-oo`, … |
| W&B eval runs | `qwen3-<tier>-<variant>-repaired` |
| W&B training groups | `sft-<task>` / `grpo-<task>` |
| Comparison CSVs / plot dirs | `csv_comparisons_<prefix>_<version>`, `plots_<prefix>_<version>` |

Everything the pipelines *share* — `datasets/{processed,augmented,grpo_pool}`, the HF model cache — is read-only
during training, so concurrent readers are safe. (The submitter pre-downloads models on the login node
specifically to avoid concurrent SLURM jobs racing on HF cache locks.)

**Every exception has been closed.** The legacy unprefixed unified baseline run name is gone (see
[Naming](#naming-and-versioning)), and `models/sft_trainer.py`'s W&B group is now `sft-<task>` rather than a
bare `sft`. `tests/test_core/test_name_isolation.py` enumerates every writable name for all four tasks × three
tiers × three versions and asserts the set has no duplicates — so a future task cannot reintroduce a collision
without a red test.

All four `hpc_*.sh` start with `set -eo pipefail` and a guarded `cd`. Without that, a failed training step
still ran inference/repair/eval and the job could exit 0, so the `afterok` dependency would launch the next
stage against a missing adapter. Deliberately not `set -u` — several lines legitimately expand possibly-unset
vars (`PYTHONPATH`, `SLURM_JOB_ID`). `hpc_merge_sft.sh` additionally refuses to run if no adapter exists at
`<sft_variant>/best`, and `hpc_grpo.sh` refuses if the merged KL base is missing.

### Adding a new task pipeline

Six steps, none of them orchestration:

1. `core/tasks.py` — one `TaskSpec` in `TASK_REGISTRY`: name, unique prefix, capability set, wire format.
   `VALID_TASKS` and `TASK_PREFIXES` derive from it; every naming, gating and validation site follows.
2. `configs/tasks/<task>.yaml` — `task_name`, `prompt_key`, `reward_components`, `reward_weights`, token budgets,
   and optionally `sft_dataset_subdir`. Copy `violations_only.yaml` or `object_only.yaml`. Every key in
   `reward_weights` must also appear in `reward_components` (`get_reward_funcs_for_task` now raises otherwise —
   unknown weight keys used to be ignored silently).
3. `data/prompt_templates.py` — new prompt constant + `PROMPT_REGISTRY` entry.
4. `data/schemas.py` — new Pydantic output model + `SCHEMA_REGISTRY` entry. Its fields must match the declared
   capabilities exactly; `tests/test_core/test_task_registry.py` asserts this, and the repair layer derives its
   canonical-key allow-set from `model_fields`.
5. `data/preprocessor.py` — a target builder in `_TARGET_BUILDERS` and a ground-truth builder in `_GT_BUILDERS`.
   Boxes: targets scale to `[0,1000]`, ground truth stays `[0,1]`.
6. `tests/` — mirror the `_oo` / `_co` suites (`test_preprocessor_*`, `test_reward_*`, `test_evaluator_*`) plus a
   fixture factory in `conftest.py`.

**Nothing to add in:** the SLURM scripts, the submitter, the evaluator, structural repair, the reward assembly,
the comparison tables, or the plots — all of those read the registry. There is no new data prep either: the
shared GRPO pool serves any task.

If the new task's wire format is not fenced JSON, also give
`evaluation/output_parser.py::parse_output_for_task` / `serialize_output_for_task` a branch for it, and a
`reward_format` meaning (see `caption_only` for the worked example).

### Data flow

```
HF hub → datasets/raw → datasets/raw_cleaned → datasets/processed ──→ datasets/augmented → SFT (unified, vo)
                                                        │
                                                        ├──────────────────────────────→ SFT (oo, co)
                                                        └────────→ datasets/grpo_pool ──→ GRPO (all four)
```

**The naming trap:** in `configs/base.yaml`, `processed_subdir` points at `datasets/augmented`, and the
non-augmented base is `raw_processed_subdir` → `datasets/processed`. So a bare
`data.loader.load_processed_dataset()` returns the **augmented** set.
`load_processed_dataset(subdir=...)` takes an override, which is how `object_only` and `caption_only` read the
un-augmented split via their `sft_dataset_subdir` key (see [Tasks](#tasks--what---task-actually-controls));
`build_grpo_pool.py` reads the truly-processed one directly.

Note that **inference/eval always uses the default root for every task**, so all four pipelines are scored on
byte-identical test images. That is safe because augmentation only rewrites the `train` split.

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
`get_reward_funcs_for_task(task)` filters and reweights it from the task YAML. **That is the only
path.** A second "composite reward" mode used to exist as a fallback for TRL versions without
`reward_weights` support; it is deleted. It ignored the task's `reward_components` entirely and
always scored all six components at *unified* weights, so any non-unified task falling into it would
have trained silently against the wrong objective — and `trl==0.23.0` supports `reward_weights`
natively, so it was unreachable dead weight wrapped around a live footgun.

**The repetition penalty now actually fires.** `REPETITION_PENALTY_FACTOR` lived only inside that
deleted composite path, so in production the live reward path had no repetition check at all.
`_apply_repetition_penalty` applies it per component, which is identical to applying it to the total
because TRL sums linearly (`Σ wₖ·(f·rₖ) = f·Σ wₖ·rₖ`). Trigger: more than 5 occurrences of one
identical box tuple, pooled across every box field the task owns. Tunable per task via
`repetition_penalty_factor` (default `0.5`; `1.0` disables). `caption_only` parses to a caption with
no boxes, so it can never fire there.

- `unified` — all 6 components at defaults (format .05, caption .15, grounding .25, violation_id .30,
  violation_grounding .15, reasoning .10). Its YAML deliberately declares **no** `reward_components`, which is
  what selects that full-registry fallback.
- `violations_only` — 4 components: format .10, violation_id .40, violation_grounding .30, reasoning .20.
- `object_only` — 2 components: format .10, grounding .90.
- `caption_only` — 2 components: format .10, caption .90.

Every reward passes through `_strict_parse_for_task()`, which runs `parse_output_for_task` (the task's wire
format) then Pydantic validation, and returns `None` on any exception. **`None` → `0.0` for that component.**

The expected output for the three JSON tasks is a *flat* JSON object inside a ```` ```json ```` fence — the
unified shape, minus whatever the task does not own:

```json
{"caption":"...","rule_1_violation":{"bounding_box":[[x,y,x,y]],"reason":"..."},"rule_2_violation":null,
 "rule_3_violation":null,"rule_4_violation":null,"excavator":[[x,y,x,y]],"rebar":[],"worker_with_white_hard_hat":[]}
```

`caption_only` instead emits bare prose, with no fence and no keys.

Required-field asymmetries, each deliberate:

- `UnifiedOutput.caption` is required — a missing caption zeroes **all six** rewards, including grounding.
- `ObjectOnlyOutput`'s three class keys are all **required**, unlike `UnifiedOutput`'s object fields which
  default to `[]`. Under `unified`, the required `caption` anchors the strict schema gate; `object_only` has no
  such anchor, so with everything defaulting the schema would accept `{}` and even `{"excavators": [...]}`. That
  would make schema adherence uninformative, make `reward_format` free, and — worst — let the strict gate accept
  an aliased key so structural repair never renames it, silently scoring a real detection as "nothing detected".
- `CaptionOnlyOutput.caption` rejects blank/whitespace, since it is the entire output.
- `ViolationsOnlyOutput`'s four fields all default to `None`, so `{}` validates. That is pre-existing behaviour,
  and consistent with `null` meaning "not violated".

`bounding_box` is a list *of* 4-float boxes. **Predicted boxes are scaled [0,1000]; ground truth stays [0,1]** —
rewards call `scale_1000_to_01` on predictions only. Getting this backwards silently zeroes every IoU metric.

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
6. **The true-negative constants are now per-task configuration, not literals.** Two knobs, both read from
   `configs/tasks/<task>.yaml` via `rewards/reward_utils.py::reward_constant` (lru-cached):

   - `grounding_tn_constant` — credit for correctly calling an object class ABSENT. Scalar or per-class
     mapping. Read by `reward_grounding.py`.
   - `violation_tn_constant` — credit for correctly calling an image SAFE. Read by `reward_violation_id.py`,
     `reward_violation_grounding.py` and `reward_reasoning.py`, so all three violation sites move together by
     construction (the old hand-coordination hazard is gone).

   Both default to the historical `0.15` when a task YAML says nothing.

   **Why they were retuned.** A true-negative constant sets the break-even detection quality for a class:
   emitting boxes for class *k* is positive-EV only when `E[IoU_k] > c·(1−p_k)/p_k`, where `p_k` is that
   class's prevalence in the GRPO pool. At a flat `c = 0.15` and the measured prevalences (excavator 0.361,
   rebar 0.088, worker_with_white_hard_hat 0.115) the break-evens were **0.27 / 1.55 / 1.15** — two of them
   above 1.0, i.e. *unreachable*. Suppressing those two classes was strictly dominant regardless of how good
   the detector became. All three JSON tasks now carry frequency-aware values solving for a 0.5 break-even.

   Likewise `violation_tn_constant: 0.15` made always-asserting rule_1 (EV ≈ 0.391, since rule_1 covers 39%
   of the pool) beat honest abstention (EV 0.075) by 5×, for a policy that never looks at the image. It is
   now `0.85`, above the ≈0.78 crossover. The old rationale — "balance the EV against the 91% imbalance" —
   never described what GRPO sees: `build_grpo_pool.py` builds the pool **50/50**.

   **On affine invariance.** CLAUDE.md used to claim the constant was irrelevant under TRL's
   `scale_rewards='group'`. That holds only for groups taking **two** distinct values. Verified: at 2 values
   the advantages are invariant (0.15 → `[+0.5764, −1.7291]`, 0.50 → `[+0.5771, −1.7312]`, the delta being
   the `1e-4` epsilon); at **≥3** values it is not (best-rollout advantage `+1.2407` at 0.15 vs `+1.0863` at
   0.30). For `object_only`, where grounding is the *only* varying component and spans `{0, c, IoU}³/3`,
   the constant is the whole objective's shape.

   `scripts/validate_rewards.py` fails the build if any class's break-even exceeds 0.75 or if reflexive
   flagging beats abstention. Run it after touching either knob.
7. **`best/` and `final/` are now different checkpoints.** `configs/sft.yaml` sets
   `load_best_model_at_end: false`, so `final/` is the literal end-of-training state (debugging) and `best/` is
   the lowest-`eval_loss` state, written eagerly by `SaveBestModelCallback`. **The merge → GRPO handoff consumes
   `best/`**, and the post-SFT eval runs `--checkpoint best` so the reported SFT numbers describe the checkpoint
   GRPO actually trains from. GRPO itself has no `best/` (no eval dataset), so `hpc_grpo.sh` stays on
   `--checkpoint final`. Keep `best_model_threshold: 0.0` — it is an *absolute* delta, and the old `0.005`
   froze `best/` hundreds of steps early on a ~0.055 loss.

**Violation semantics — two predicates, deliberately different.**
`rewards/reward_utils.py::_is_violation_present` remains the single source of truth for "is this rule
violated?", used by the GRPO rewards *and* by `evaluation/metrics_{violations,reasoning}.py`, so training and
evaluation can never disagree about presence.

`_is_substantive_violation` is the second predicate: present **and** carrying a non-empty `reason` or at least
one bounding box. `reward_violation_id` uses **both**, because a contentless
`{"reason":"","bounding_box":[]}` is an assertion with no content and those two facts pull opposite ways:

- as an *assertion* it must still count as a prediction, so flagging a safe image is penalised as a false
  positive — dropping it would let a false alarm collect true-negative credit;
- as *contentless* it must not earn true-positive credit, because it names no location and gives no reason.

So presence drives precision (the FP denominator) and substance drives recall (the TP numerator). A
contentless assertion therefore scores as a **miss** on a real violation and a **false alarm** on a safe
image — never as a hit. It previously scored a perfect F₂ = 1.0 on the most heavily weighted component of
`violations_only` (0.40) while contributing nothing to grounding or reasoning, both of which are
TP-conditioned and so never penalised it. Governed by `require_violation_substance` (default `true`).

**`null` is the only safe signal.** The prompt says *"If NOT violated, output null"*, so emitting a violation
object at all is an assertion of violation — even `{"reason": "", "bounding_box": []}`. A contentless
assertion earns nothing anywhere: `reward_violation_id` now requires substance for TP credit (see above),
and grounding/reasoning were always TP-conditioned. It still counts as a *prediction*, so flagging a safe
image is penalised. The one exception is a bare `{}`, which carries no keys and no assertion;
`normalize_violation_value` normalizes it to `null` first.

This matters because `preprocessing/structural_repair.py` manufactures these shapes: it rewrites a bare
`true` into `{"reason":"","bounding_box":[]}`, and turns a bare reason string into a box-less
violation. Reading the first as *Safe* would invert the model's own answer — crediting an unsubstantiated
alarm as "correctly identified a safe site" — and left a reward-hacking surface where empty violation objects
collected the true-negative reward on safe images.

All of that now runs **only for tasks with the `violations` capability**. It used to run unconditionally *and*
assign rather than test, so every repaired record of every task gained four phantom `rule_N_violation: null`
keys. Likewise the object-box repairs run only under `objects` (they were previously gated behind
`task == "unified"`, which is why a recoverable `object_only` box would have landed in `still_broken`), and the
caption list-join only under `caption`. `caption_only` takes an entirely separate plain-text repair path that
unwraps a stray fence or `{"caption": ...}` object and writes the repaired `raw_output` back as bare prose.

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
- **Deleted** in the ledger pass, listed because older notes reference them: `scripts/test_grpo.sh` and
  `scripts/test_eval_grpo.sh` (task-blind smoke scripts that defaulted every stage to `unified`),
  `analyze_metrics.py` (hardcoded `vo_{phase}_{tier}` against a layout nothing produces), and
  `scripts/test_{no_unsloth_tokenize,batch_vs_explicit_template,trl_collator}.py` (spent forensics from
  the model-is-blind investigation). `scripts/test_processor_batch_collapse.py` survives — it is still
  the reusable no-GPU collapse check.
- `rewards/{json_validity,caption_quality,rule_violation_accuracy,grounding_iou}.py` are legacy and unwired;
  `rule_violation_accuracy` still returns 1.0 for both-empty, the exact hack the 0.15 constant replaced.
- `experiments/run_dual_evaluation.py` returns a **nested** metrics shape
  (`{structural_metrics, strict_metrics, valid_metrics}`) that no analysis script reads any more, and no
  `hpc_*.sh` invokes it. Its capability gating is kept in sync with `evaluation/evaluator.py`, but treat the file
  as semi-stale.
- `rewards/reward_utils.py::_strict_parse`/`_strict_parse_cached` and
  `evaluation/output_parser.py::validate_unified_output` are the pre-task legacy parse path, hardcoded to
  `UnifiedOutput`. Use `_strict_parse_for_task` / `parse_output_for_task` / `validate_output_for_task`.
- `data/preprocessor.py` still carries the task-blind `raw_sample_to_conversation`,
  `build_unified_sft_dataset`, `to_grpo_prompt` and `build_grpo_dataset` alongside the `_for_task` versions the
  pipeline actually uses.

- `docs/Metrics.md` documents a metric namespace that no longer exists (`grounding_iou_all_macro_*`,
  `_excl`, `grounding_iou_total_macro`). Grepping `evaluation/` for those returns nothing — the live families
  are `grounding_{mask,greedy}_iou_{all,exist}_*`. Its analysis of `rule_0` semantics is still sound, and its
  tn0/tn1 rationale still applies to **object** grounding — but the **violation**-grounding `_tn1` keys have
  been removed (see below); only `_tn0` remains there.
- **All pre-existing GRPO metrics are void.** Those runs trained prompt-only — images never reached the model
  (fixed in `b8f2470`). `evaluation_results/` and `evaluation_results_v2/` also disagree on the same-named 8B
  run (VO SFT f1_micro 0.4883 vs 0.3537) with nothing documenting which is authoritative. Baseline and SFT
  numbers there are usable; do not draw any GRPO conclusion from them.
