# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project fine-tuning Qwen3-VL (2B/4B/8B via Unsloth) for construction safety inspection on the
`LouisChen15/ConstructionSite` dataset (6308 train / 701 val / 3004 test, un-augmented). Training is two-phase:
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
That isolation is a designed property — see [Parallel-safety](#parallel-safety-the-isolation-guarantee) —
asserted by `tests/test_core/test_name_isolation.py`.

**`core/tasks.py` is the single place a task is registered.** One frozen `TaskSpec` per task carries its name,
prefix, *capability set* (`caption` / `objects` / `violations`) and wire format; `VALID_TASKS`, `TASK_PREFIXES`
and every metric/repair gate in the repo derive from it. No conditional anywhere should compare a task name to a
literal string — ask a capability question (`task_has(task, CAP_OBJECTS)`) instead.

**Note:** `output_format` used to also appear as a key in every `configs/tasks/*.yaml` (e.g.
`"fenced_minimized_json"`) — removed 2026-09-05. Nothing ever read it, and its vocabulary disagreed with the
real source of truth, `core/tasks.py::TaskSpec.output_format` (`"fenced_json"`/`"plain_text"`). Don't re-add a
YAML-level `output_format` without reconciling the two vocabularies.

## Environment

Two very different environments; know which one you're in.

| | Local (Windows dev box) | HPC (ARC / SLURM) |
|---|---|---|
| Python | 3.10.11, `.\venv\Scripts\Activate.ps1` | `module load gcc/13.3.0 python/3.12.5`, `source $HOME/envs/vlm_grpo/bin/activate` |
| Data root | `./vlm_data_root` | `VLM_DATA_ROOT=/home/$USER/vlm-finetuning-project1` |
| GPU / unsloth | none | varies by stage — see [GPU policy](#gpu-policy) below |

**Runs locally:** the test suite, `preprocessing/structural_repair.py`, all plotting/analysis scripts, anything
importing only `core/`, `data/schemas.py`, `rewards/`. **HPC only:** `experiments/run_{sft,grpo,inference}.py`
(`run_sft.py` imports `unsloth` at line 5, before transformers — it cannot even be imported on Windows),
`merge_sft_adapter.py`, `augment_rare_classes.py`, all `scripts/hpc_*.sh`. The four `submit_*_pipeline.py`
wrappers and `scripts/submit_pipeline.py` run locally as a dry run (no `sbatch` → they print the exact commands).

`requirements.txt` has loose lower bounds and is **not** the authoritative version set — the working pins live in
`scripts/setup_arc.sh` (`transformers==5.4.0`, `trl==0.23.0`, `datasets==4.3.0`, `unsloth_zoo==2026.8.12`).
**A local dev venv may have a different TRL version installed** — always re-verify a TRL-default claim against
the ARC-pinned `0.23.0`, not whatever is on Windows.

There is no linter, formatter, Makefile, or CI. `.gitattributes` forces LF on `*.sh`/`*.py`/`*.yaml` to prevent
SLURM CRLF errors — don't defeat it from Windows.

## Commands

### Tests

```powershell
python -m pytest tests/ -v                                       # all (~507 tests, no GPU needed)
python -m pytest tests/test_core -v                               # task registry + name-isolation proof
python -m pytest tests/test_core/test_blocker_fixes.py -v         # the pre-flight blockers, B1-B8
python -m pytest tests/ -k "_oo or _co" -v                        # just the two newer pipelines
```

**Reward-surface validator (no GPU, run before every submit).**

```powershell
python scripts/validate_rewards.py                       # all four tasks, probe + census
python scripts/validate_rewards.py --task object_only --probe --pool-stats   # on ARC
python scripts/validate_rewards.py --sft-stats --task object_only            # on ARC
```

Scores synthetic honest and degenerate policies against real ground truth and **fails** if any degenerate
policy beats the honest one, if any object class's break-even IoU exceeds 0.75, or if unconditional rule_1
assertion beats honest abstention on pool expected value. `--census` tokenizes every SFT target and fails if
any would truncate at `max_seq_length`, deriving the vision-token ceiling analytically rather than sampling one
image (see [Three length keys](#three-length-keys-three-different-jobs)).

Use `python -m pytest`, not bare `pytest`: `tests/` has no `__init__.py` and `conftest.py` does no `sys.path`
insertion, so first-party imports only resolve when CWD is on the path. On HPC the SBATCH scripts export
`PYTHONPATH`, so bare `pytest` works there.

### Full pipeline (HPC, from repo root on the login node)

```bash
python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task unified          --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task object_only      --tiers 2b 4b 8b --version v1
python scripts/submit_pipeline.py --task caption_only     --tiers 2b 4b 8b --version v1
```

One submitter for every task. The four per-task wrappers (`submit_{unified,vo,oo,co}_pipeline.py`) are
equivalent 3-line shims. `--version` is **required**, must match `v<digits>`, and is the single source of truth
for every generated name. `--tiers` currently accepts **any string** (no `choices=`) — a typo like `--tiers 8B`
is accepted, submits four jobs, and only fails on the compute node after the queue wait; double-check spelling
by eye.

**Task/tier selection matrix** — what to pass for each shape of run:

| You want | Command |
|---|---|
| One task, all three tiers | `python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b 8b --version v1` |
| One task, one tier (2b smoke) | `python scripts/submit_pipeline.py --task violations_only --tiers 2b --version v1` |
| One task, two tiers | `python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b --version v1` |
| All four tasks, one tier | run the command above once per task, same `--tiers 2b --version v1` |
| All four tasks, all tiers (full grid) | run it four times, once per task, `--tiers 2b 4b 8b --version v1` each |

Every combination is safe to fire concurrently at the same `--version` — see
[Parallel-safety](#parallel-safety-the-isolation-guarantee). The submitter pre-downloads the model on the login
node (`--skip-preload` to opt out), then submits 4 jobs per tier with `afterok` dependencies: `baseline`
(independent) ‖ `sft → merge → grpo`.

**Known gap in the preload step (verify before a large submission):** `preload_model()` in
`scripts/submit_pipeline.py` calls `subprocess.run(["hf", "download", model_name])` with **no `HF_HOME`
override** — it inherits the login shell's environment, which has no `HF_HOME` set anywhere (`setup_arc.sh`'s
`export HF_HOME` only lives inside that one-time setup script's own process). Every phase script separately sets
`HF_HOME=$HOME/scratch/hf_cache`. So the preload may download into `~/.cache/huggingface` while every job looks
in `~/scratch/hf_cache` — defeating the preload's purpose (the cache-lock race it exists to prevent still
happens) and burning home-directory quota. **Check `ls ~/.cache/huggingface/hub` vs `ls ~/scratch/hf_cache/hub`
on the login node after a preload** before trusting it worked.

There are only **four** phase scripts, all task-parameterized — `scripts/hpc_{baseline,sft,merge_sft,grpo}.sh`,
taking the task as their first positional argument. Because `#SBATCH` directives cannot read arguments, the
submitter passes `--job-name`, `--output`, `--error`, `--mem`, `--time` and (for GRPO only, and only as an
escape hatch) `--gres` on the `sbatch` command line, all of which **beat** the in-file directives.

On Windows these submitters degrade gracefully — `sbatch` is missing, so they print the exact commands with
`DUMMY_JOB_ID`. Useful as a dry run, and the fastest way to eyeball that two tasks' paths do not overlap.

**W&B runs fully offline.** All four `hpc_*.sh` export `WANDB_MODE=offline`, so **no run appears in the web
dashboard while a job is running or after it finishes**, including the go/no-go checks below that say "watch
`reward/mean` on W&B." Either read the values off the SLURM `.out` log directly (GRPO logs every 5 steps, SFT
every step) or run `wandb sync $HOME/scratch/wandb/offline-run-*` from the login node afterward.

### GPU policy

The `gpu-h100` partition holds **both** card types — four H100 nodes against **one** H200 node (`egh2`, 2
GPUs) — so the GRES *type*, not the partition, selects the hardware.

| stage | GRES | why |
|---|---|---|
| `hpc_baseline.sh` | `gpu:h100:1` | indifferent to the card; H100s are plentiful |
| `hpc_sft.sh` | `gpu:h100:1` | same |
| `hpc_merge_sft.sh` | `gpu:h100:1` | same |
| `hpc_grpo.sh` | `gpu:h100:1` | **as of 2026-09-06** — moved off H200, see decision below |

`submit_pipeline.py --gres gpu:h100:1` is an **escape hatch**: it forces *every* stage onto one card type for a
debug run (e.g. onto the H200 for a comparison test). Without it no `--gres` reaches the `sbatch` line and each
script's own directive governs. `tests/test_core/test_blocker_fixes.py` pins the per-stage policy.

**GRPO applies the same 1.2 MP image cap as SFT and inference — it is not uncapped.**
`models/grpo_trainer.py::run_grpo` loads two configs (`cfg` from the grpo chain, `sft_cfg` from the sft chain)
and passes **`sft_cfg`** — which unconditionally carries `image_min_pixels: 200704` /
`image_max_pixels: 1204224` from `configs/sft.yaml` — into `load_model_for_training(sft_cfg=sft_cfg, ...)` →
`apply_pixel_bounds`. `grpo.yaml`'s own `if "image_max_pixels" in cfg:` override block is currently dead code
(no task YAML or `grpo.yaml` sets that key), but it doesn't need to fire — the cap is already there via
`sft_cfg`. Earlier versions of this doc and `scripts/hpc_grpo.sh`'s comment claimed GRPO runs uncapped up to
14.6 MP; that was true only **before** the pixel-bounds key-rename fix, and `models/model_loader.py`'s own
comment says as much: *"This previously wrote `{min_pixels, max_pixels}`... the cap was never applied — which
is the most likely cause of the recorded 92.97/93.12 GiB OOM."*

**Decision (2026-09-06): move GRPO off H200 onto H100, same as every other stage.** Supersedes the 2026-09-05
decision to keep GRPO on H200. The 92.97/93.12 GiB OOM that originally justified H200 was measured **before**
the pixel-bounds key-rename fix above, under images that were silently uncapped — evidence that predates the
fix making the cap real. Once real GRPO jobs started running under the corrected, capped code, the real vo-2b
run (job `47873931`) measured only **~37-38 GB peak** via repeated `nvidia-smi` checks — comfortably inside a
single H100's ~80-93 GB, let alone the H200's 141 GB. Combined with H100 being far more plentiful (10 cards
across 4 nodes vs H200's 2 cards on 1 node, so much shorter queue waits), the H200 pin no longer has a
justification. `configs/grpo.yaml`'s `per_device_train_batch_size: 16` was originally sized assuming H200
headroom and is kept at 16, not reverted to its old fallback of 8, on the strength of the 2b measurement above
— **but 4b/8b have not been measured under this GRES on H100.** Watch memory (`nvidia-smi`, or the
`GPUMemoryLoggingCallback` W&B panel) on the first 4b/8b GRPO run under this policy; if it OOMs, drop
`per_device_train_batch_size` back to 8 in `configs/grpo.yaml` (previously verified safe).

**Operational note: changing a phase script's `--gres` only affects jobs submitted *after* the change.**
`sbatch` reads `#SBATCH` directives once, at submission time, and bakes the resulting resource request into
the job's own record in the scheduler — it never re-reads the script file when the job actually starts
running. A GRPO job already queued or running under the old `hpc_grpo.sh` keeps requesting H200 regardless of
a later edit to the script; only a fresh `sbatch` (a fresh `submit_pipeline.py` call) picks up the new GRES.
Verify a specific job's actual request with `scontrol show job <jobid> | grep -i gres`.

**Walltime: `gpu-h100`'s real `MaxTime` is `1-00:00:00` (24h), confirmed via `scontrol show partition gpu-h100`
on ARC 2026-09-06.** `--time` is a partition property, not a GRES-type one — it binds identically no matter
which card GRES requests. GRPO's walltime was `48:00:00` and has been corrected to `24:00:00` in both
`TIME_CONFIG` (`scripts/submit_pipeline.py`) and `hpc_grpo.sh`'s own `#SBATCH --time=` directive. The old value
was silently unsubmittable: `sbatch` rejects an over-limit `--time` at submission, not at runtime, so baseline/
sft/merge would have queued fine while every GRPO job — the last stage in the chain — simply never got
scheduled, looking nothing like a training failure. Pinned by
`tests/test_core/test_blocker_fixes.py::test_b13_grpo_walltime_does_not_exceed_partition_max_time`.

**24h has no confirmed headroom for the larger tiers.** The only recorded GRPO walltime (5h12m/epoch) is from a
prompt-only run — images never reached the model — so it says nothing about what real image-conditioned
generation costs at 4b/8b. If a GRPO job is killed by the wall, it is not lost: `models/grpo_trainer.py`
auto-resumes from the last checkpoint (`save_steps: 20`, so at most ~20 steps of progress at risk) — the
response to a timeout is to **re-submit the identical GRPO job** (same variant name), which continues rather
than restarts. Watch the 2b run's actual `train_runtime` before assuming 4b/8b fit.

### Individual stages

```bash
python -m experiments.run_sft --tier 8b --variant oo-sft-8b-v1 --task object_only
python scripts/merge_sft_adapter.py --tier 8b --task object_only \
  --adapter_path "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/oo-sft-8b-v1/final" \
  --output_path  "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-oo-sft-8b-v1"
python -m experiments.run_grpo --tier 8b --variant oo-grpo-8b-v1 --task object_only \
  --base_model_override "$VLM_DATA_ROOT/checkpoints/qwen3vl-8b/merged-oo-sft-8b-v1"
python -m experiments.run_inference --tier 8b --variant oo-sft-8b-v1 --checkpoint final --task object_only
python preprocessing/structural_repair.py --input "$PREDS/predictions.jsonl" \
  --output "$PREDS/repair_applied/predictions_repaired.jsonl" --task object_only
python -m experiments.run_evaluation --predictions_path "$PREDS/repair_applied/predictions_repaired.jsonl" \
  --skip_spice --task object_only
# violation tasks: add --use_llm_judge (the phase scripts always pass it) -- see below
```

`--task` and `--tier` are both `required=True` (`choices=VALID_TASKS` on `--task`) at every direct entry
point — `run_sft.py`, `run_grpo.py`, `run_inference.py`, `run_evaluation.py`,
`scripts/preflight_grpo.py`, and `merge_sft_adapter.py`. Neither flag has a silent default any more (the old
`active_tier`/`"unified"` fallbacks are gone); a hand-run invocation that forgets either now errors
immediately instead of training/evaluating/comparing against the wrong config. The orchestrated `hpc_*.sh`
path was never affected either way — it always passed both explicitly.

`python scripts/preflight_grpo.py --tier 2b --task object_only` runs a sanity check before burning a GRPO job.
It assembles the task's real reward functions (catching a `reward_components` typo early) and measures the
real text-only prompt length for that task — but it validates against the **SFT-augmented split**
(`load_processed_dataset()`, the default), not `datasets/grpo_pool` itself. A missing/malformed pool is not
caught here; it surfaces later as `run_grpo.py`'s `FileNotFoundError` (no silent fallback — intentional).

**Inference has no auto-resume and no per-batch retry.** `run_inference_batched` runs the split start to
finish, truncates `predictions.jsonl` on every run (opened once in `"w"` mode), and lets a failing batch crash
the job. Re-running a stage now re-does the whole split, which is what you want on a cluster.
`structural_total_samples_count` should always equal the split size (3004 for test).

Inference → **structural repair → evaluation** is a fixed chain; never evaluate raw `predictions.jsonl`.
Evaluation needs a JVM for METEOR/CIDEr-D only for tasks that score text (`caption` or `violations`
capability); `object_only` needs no JRE. Pipelines always pass `--skip_spice`; pass `--skip_java_switch`
off-Linux.

### Data prep

Both derive from `datasets/processed` and neither reads the other, so they can run in parallel.
**Built and verified on ARC 2026-09-02.**

```bash
sbatch scripts/augment_data.sh          # → datasets/augmented  (SFT input for unified/vo; CPU-only, ~30 min)
python data/build_grpo_pool.py          # → datasets/grpo_pool   (GRPO input, all four; no args, ~2 min)
python scripts/dataset_report.py        # → datasets/stats/dataset_report.json  (full inventory)
```

Each writes a manifest next to its output: `datasets/augmented/augment_manifest.json` and
`datasets/grpo_pool/build_manifest.json`. **`scripts/augment_data.sh` has no `set -eo pipefail`, no guarded
`cd`, and its last line unconditionally prints "completed successfully"** — a failed `cd` or a crashed
`data.augment_rare_classes` call still exits 0. Since `datasets/augmented` is the SFT input for `unified` and
`violations_only`, check the actual row counts below after running it, don't just trust the exit code.

**Measured dataset facts** — no longer estimates, `build_manifest.json`'s real numbers:

| | rows | rule_1 | rule_2 | rule_3 | rule_4 | safe |
|---|---|---|---|---|---|---|
| train (un-augmented) | **6308** | 609 (9.7%) | 53 (0.8%) | 98 (1.6%) | 42 (0.7%) | 5528 (87.6%) |
| val | **701** | 68 | 6 | 11 | 4 | 615 |
| **GRPO pool** | **1732** | 677 (**39.1%**) | 59 (3.4%) | 109 (6.3%) | 46 (2.7%) | 866 (**50.0%**) |

- **The pool is exactly 50/50** violation/safe — what `violation_tn_constant` (0.30 as of 2026-09-10, was
  0.85) is solved against, not the ~88%-safe test distribution. All 701 val rows + all 780 train violation images + 251 randomly-drawn train
  safe images (val already supplies 615 of the ~866 needed).
- **rule_1 covers 39.1% of the pool** — the figure the reflexive-flagging EV calculation uses. Measured.
- **rule_4 sits at 0.67% of un-augmented train**, so a batch of 32 contains none of it 80.8% of the time —
  precisely why augmentation exists. rule_2 is starved in 76.3% of batches, rule_3 in 60.6%.
- **50.87% of the GRPO pool (881 of 1732) contains none of the three `object_only` target classes.** For those
  images the honest answer and the degenerate "always empty" answer are the *same string* — see
  [Rewards](#rewards-and-the-output-contract).

Step counts (`dataloader_drop_last: true`, 2 epochs, effective batch 32): `8198 // 32 × 2 =` **512** for
`unified`/`vo` SFT, `6308 // 32 × 2 =` **394** for `oo`/`co` SFT, and `1732 // 32 × 2 =` **108** for GRPO (all
four tasks). `dataloader_drop_last` is now explicit in `configs/grpo.yaml` (it used to be absent from the
GRPO chain, defaulting to `TrainingArguments`' own `False`), so this floor-division arithmetic is enforced,
not assumed.

### LLM-as-a-judge reasoning evaluation

`evaluation/metrics_llm_judge.py` replicates the dataset paper's Stage-3 reasoning evaluation, so our reasoning
scores land in the **same units as the paper's Table 8** (0-6 per rule; LLaVA-13B 2.9, GPT4V 4.5, human upper
bound 5.2 on rule_1). It runs **alongside** `reasoning_text_similarity_*` — BERTScore/METEOR/CIDEr-D/CLIPScore
measure overlap with the reference sentence; the judge measures whether the explanation is about the right rule,
describes the same violation, and pinpoints the violator. Neither replaces the other.

**Enabled by `run_evaluation.py --use_llm_judge`**, which `hpc_{baseline,sft,grpo}.sh` always pass. Gated on the
`violations` capability, so `object_only`/`caption_only` skip it silently. Without the flag its keys are
**absent** from `metrics.json`, never zeros.

**Cache the judge on the login node before submitting** — compute nodes have no internet, and the repo is gated:

```bash
# 1. once, in a browser: accept the Llama 3 license at
#    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct  (same account as your token)
module load gcc/13.3.0 python/3.12.5 && source $HOME/envs/vlm_grpo/bin/activate
export HF_HOME="$HOME/scratch/hf_cache"            # MUST be the cache the jobs read
hf auth login                                        # token with read access to gated repos
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"   # ~16 GB
```

`--exclude "original/*"` is not optional in practice: the repo also ships `original/consolidated.00.pth`, a second
~16 GB copy of the weights in Meta's own format that transformers never reads. If the gate is refused, set
`configs/base.yaml::llm_judge.model_id` to `NousResearch/Meta-Llama-3-8B-Instruct` (same weights, ungated).

**If the model is not cached, the evaluation still succeeds.** `run_llm_judge` is fail-soft by default
(`llm_judge.fail_hard: false`): it logs an ERROR, writes `llm_judge_status.json` with `"status": "failed"`, and
omits its keys — so a cache problem cannot cost the 3004-image evaluation it runs inside, or block `afterok`
dependents. Check that file, not just the job exit code.

**What is replicated exactly, from the paper:** the three criteria and the 0/1/2 scale definition (verbatim in
`CRITERIA_DEFINITIONS` / `SCALE_DEFINITION`), Llama 3 8B Instruct, text-only, true positives only, per-rule
three-shot prompting, beam search with `num_beams: 5` and `seed: 20`, and a plain-text reply
(`Relevance: 2 marks` … `Overall: 6 marks`, as in the paper's figures — the parser also accepts the paper's own
`Relavance` misspelling).

**What the paper does not publish, and what we chose — know these before comparing to Table 8:**

- **The judge prompt.** `JUDGE_SYSTEM_PROMPT` is built from the paper's criteria and scale text word for word.
- **The rule text** shown to the judge is `data/prompt_templates.py::SAFETY_RULE_TEXTS` — the wording our models
  were prompted with. That dict is now the single source of truth for rule wording; `_SAFETY_RULES` is rebuilt
  from it, and `tests/test_evaluation/test_llm_judge.py` pins the training prompts by sha256 so the refactor
  provably changed nothing the models see.
- **Few-shot scores are only partly human.** The paper's examples were human-judged; we have no human judgments.
  `FEWSHOT_EXAMPLES` (3 per rule) uses the paper's **five published, human-scored worked examples verbatim**
  (Fig. 8 + appendix) as anchors and fills the other **seven** slots with train-split references plus
  hand-authored candidates — tagged `source="authored:train/<image_id>"`. Those seven scores are our
  calibration. Across all twelve, every mark 0/1/2 is demonstrated for every criterion.
- **No human-correlation validation.** The paper reports Spearman 0.83 / Pearson 0.91 against humans; without
  human labels we cannot. Every judged item goes to `llm_judge_details.json` (reference, candidate, raw reply,
  parsed marks) — **read a sample of it before trusting a mean.**
- **`batch_size: 1`**, not larger. Batched beam search pads prompts, and in bf16 that can flip a close beam; at 1
  a given (reference, candidate) pair scores identically in every run, which a v1-vs-v2 comparison depends on.

**Keys** (all prefixed `reasoning_llm_judge_`): `{relevance,equivalence,specificity}_rule_N` (0-2) and
`total_rule_N` (0-6, **the Table 8 column**), each with `_micro` and `_macro`; `scored_count_*`,
`unparsed_count_*`, `unparsed_rate_micro`, `empty_candidate_count_micro`, `clamped_count_micro`, and
`<metric>_macro_n_rules`. The macro follows the same rule as the text-similarity macros (below): mean over rules
that had a true positive. `llm_judge_status.json` records the model, decoding settings, `rubric_sha256` (so two
runs' judge scores can be checked for having used the same rubric), few-shot provenance and a histogram of totals.

**`scored_count_rule_N` always equals `reasoning_text_similarity_scored_count_rule_N`** — both families score the
list from `evaluation/metrics_reasoning.py::collect_tp_reason_pairs`, extracted for exactly that purpose and
verified against the pre-extraction code on 691 real TP pairs.

**Scoring edge cases:** an unparseable reply scores **0/0/0** and is counted — this *depresses* the means, so
treat `unparsed_rate_micro` above ~0.02 as a prompt/parser problem to investigate, not a model result. An empty
candidate reasoning scores 0/0/0 without calling the model. A mark outside 0-2 is clamped and counted. The
total is always the sum of the three criteria, never the reply's own `Overall` line.

**Cost (estimated, not yet measured on ARC):** ~16 GB bf16 + beam KV cache, on top of BERTScore/CLIP, well
inside an 80 GB H100; roughly 5-15 min per evaluation for the 200-500 TPs a run typically has. **Red flags:**
`unparsed_rate_micro > 0.02`; nearly every item at 6/6 (anchors too lenient, or the judge agreeing by default);
`total_rule_1` far outside 2-6 (check the few-shot blocks first).

### Local analysis

Two commands, not five. `compare_results.py`, `plot_metrics.py`, `plot_metrics_vo.py` and
`generate_comparison_csv.py` were deleted in `07592de`; their logic lives in the two below.

```bash
# 1. ON ARC, against the live results tree. No GPU, no SLURM job, pure file I/O.
#    Discovers every task/tier/phase/version present; no flags needed.
python -m experiments.build_results_index --out results_index/index.json

# 2. Anywhere, against that one file (scp back ~450 KB instead of many metrics.json).
python -m experiments.compare_all --index results_index/index.json --out results_index/
python -m experiments.compare_all --index results_index/index.json --out results_index/ --no-charts
python -m experiments.compare_all --index results_index/index.json --out results_index/ --bootstrap 0
```

```powershell
python -m experiments.extract_qualitative --task violations_only --tier 8b --version v1
python scripts/fetch_results.py --task object_only --version v1 --tiers 2b 4b 8b  # ARC layout -> flat local
```

**`compare_all.py` prints confidence intervals, and you should read them before reading any delta.**
`--bootstrap B` (default 2000, `0` disables) resamples images to produce a 95% CI per run plus a *paired*
significance test for every adjacent comparison -- baseline/sft/grpo within a tier, and tier-to-tier within a
phase -- writing `significance.csv` alongside. It needs nothing but `index.json`: the per-image outcomes ride
along as a ~4 KB base64 string per run (`violation_per_image_outcomes_b64`, written by
`evaluation/metrics_violations.py`). Runs evaluated before that key existed report as unavailable rather than
failing. Uses numpy when importable, pure Python otherwise.

This is not decoration. **rule_2/rule_3/rule_4 have 25/63/24 positives in the entire 3004-image test split**,
so a per-rule F1 there carries a 95% interval up to 0.33 wide and `violation_identification_f1_macro` averages
four such numbers. On the real vo v1 runs, baseline to SFT is +0.24 (p<0.001) while SFT to GRPO is +0.018
(p~0.08) at 4b and +0.012 (p~0.16) at 8b, and 4b to 8b is *negative* and non-significant in both phases.
Reading point estimates alone makes phase and tier rankings look like they flip run to run; they do not, they
are inside the noise.

The delta CSV's **win tally** (`SUMMARY … _wins`) skips keys that are not "higher is better":
`results_lib.py::is_non_comparable_key` excludes counts, thresholds, word lengths, and — since the Tier-0 keys
arrived — `_n_rules` divisors, `*_positive_rate` (the GT rate is constant, so every tie was "won" by BASELINE; a
higher predicted rate is over-flagging), `unparsed` and `failure` rates (higher is worse).

**Read `violation_pred_positive_rate` first, before any recall number.** It is the fraction of images the
model flagged at all, printed next to `violation_gt_positive_rate` (0.1368 on the test split) at the top of
the violation headline table. The zero-shot baselines flag 59-94% of images, so their recall of 0.85-0.90 is
bought entirely by over-flagging and is not a detection result. `compare_all.py` also prints a **support
counts** table per task: the GT/predicted/TP counts and `scored_count`s that are the denominators behind every
ratio above them.

**Why the old `METRIC_GROUPS` dict is gone.** `generate_comparison_csv.py` hand-listed which metric keys to
emit and covered only the structural + violation families, so running it for `object_only` or `caption_only`
silently produced a CSV that was ~95% `N/A` with no warning. `compare_all.py` cannot repeat that: the index is
long-format (one row per metric key per run) and `metric_family` is derived from the key's prefix, so a new
key added to any `evaluation/metrics_*.py` appears in the next build automatically, with nothing to maintain.

`scripts/fetch_results.py` turns the ARC layout
(`results/inference/<run>/evaluation_results/metrics.json`) into the flat local layout
(`evaluation_results/<run>/metrics.json`). The comparison toolset no longer needs it —
`build_results_index.py` reads either layout directly (`--layout auto`) — so it is now only for materializing
the per-record files (`predictions_with_eval.json`, `still_broken.json`, `change_manifest.json`) that
`extract_qualitative.py` digs through. Both resolve result folders through
`core/naming.py::results_dir_names(task, tier, version)`; `experiments/results_lib.py::parse_run_name` is its
exact inverse, and returns `None` (never raises) for a name that is not ours.

## Architecture

### Config layering

`core/config.py::load_config(task, training_kind)` merges, **last wins**:

```
base.yaml → model_registry.yaml → {sft,grpo}.yaml → tasks/<task>.yaml
```

`merge_configs` is a shallow merge, nested dicts one level deep. **`configs/sft.yaml` is not in the GRPO
chain** — `models/grpo_trainer.py::run_grpo` loads it as a *second, separate* dict (`sft_cfg`, alongside
`cfg` from the grpo chain) specifically because `load_model_for_training` needs an `sft_cfg`-shaped argument.
A short, explicit list of keys is copied from `cfg` onto `sft_cfg` before that call — this is the mechanism
that lets `grpo.yaml` override specific SFT-shaped values for the GRPO phase, and it has been the source of
two of this repo's ghost-variable bugs (both now fixed, both listed below).

**Ghost-variable table — config said one thing, runtime read another. Status is what it is right now:**

| Config said | What actually ran | Now |
|---|---|---|
| SFT LR clamped `4b → 5e-5`, `8b → 2e-5` | `1.0e-4` at every tier — `run_sft.py` never passed the merged config to the trainer | **Fixed.** Clamp deleted; flat `1.0e-4`; config now reaches the trainer |
| `image_max_pixels: 1204224` (1.2 MP cap) | Uncapped, up to 14.6 MP — `apply_pixel_bounds` wrote a key shape transformers rejects | **Fixed.** Writes `{"shortest_edge","longest_edge"}` — a key rename, those keys hold pixel *areas* |
| GRPO adapter shape (`lora`/`finetune_*`) from `grpo.yaml` | From `configs/sft.yaml`'s values instead — the copy-over block that moves keys from `cfg` to `sft_cfg` never included `lora`/`finetune_*`, so `load_model_for_training` (which reads exclusively off `sft_cfg`) never saw `grpo.yaml`'s declared block | **Fixed 2026-09-05.** Copy-over now includes `lora` and all four `finetune_*` keys. Was a no-op on *observed* behaviour only because the two files' values happened to be identical — editing `grpo.yaml`'s rank for a GRPO-side ablation would have silently done nothing |
| `run_sft_unified`'s W&B init logs `task_cfg` | `UnboundLocalError` on every single SFT run, every task, every tier, right after model load — `task_cfg` was referenced in the W&B config dict before being assigned (the load lived further down, in the git-metadata block) | **Fixed 2026-09-05.** Load moved above the W&B init |
| `repetition_penalty: 1.0` in every task YAML, `unified.yaml`'s comment reads *"locked-in production default per ablation"* (i.e., someone concluded the reward-side penalty should be off) | Was read from a **different key**, `repetition_penalty_factor`, which no YAML set — always fell back to its Python default **`0.5` (ACTIVE)**. Every GRPO run before this fix had trained with the penalty on, contrary to the checked-in intent | **Fixed 2026-09-05, per explicit decision: OFF.** `_apply_repetition_penalty` now reads the same `repetition_penalty` key every task YAML already sets; dead `REPETITION_PENALTY_FACTOR` constant deleted |
| `scale_rewards="group"`, treated by this doc as one of GRPO's three load-bearing safety brakes | Was never explicitly set in `grpo_config_kwargs` — true only because it happened to be TRL 0.23.0's own default | **Fixed 2026-09-05, pinned at `"group"`** — the brake now survives a future TRL version changing its own default |
| `configs/grpo.yaml`'s "108 steps" comment assumes floor-division (`dataloader_drop_last`) | Was absent from the entire GRPO config chain (SFT sets it; GRPO didn't) — `TrainingArguments` defaults it to `False` | **Fixed 2026-09-05.** `dataloader_drop_last: true` now explicit in `configs/grpo.yaml`, confirmed via the pinned `trl==0.23.0` source that `GRPOTrainer.get_train_dataloader` passes it straight to a standard `DataLoader` |
| `base.yaml`'s `seed: 42` | Already reached the merged GRPO config for free (the merge chain always starts with `load_base_config()`), but `models/grpo_trainer.py` never read `cfg["seed"]` back out into `GRPOConfig(seed=...)` | **Fixed 2026-09-05.** `seed=cfg.get("seed", 42)` now threaded through — no new `configs/grpo.yaml` key needed, the value was already there |
| `loss_type` for GRPO | Was never set — TRL 0.23.0 defaults to `"dapo"` (global-token-count loss normalization) | **Fixed 2026-09-05, pinned at `"dapo"`** — already TRL's own recommended default and already what was running; declaring it explicitly just stops a future TRL upgrade from silently changing it. `"dapo"` specifically eliminates the length-bias problem this repo would otherwise have, since completions range from a handful of tokens (`object_only`'s box lists) to ~1000 (`unified`'s full JSON) |
| `mask_truncated_completions` for GRPO | Was never set — defaulted `False`, so a rollout that hit `max_completion_length` still contributed full per-token loss despite having no real stopping decision | **Fixed 2026-09-05, turned ON** (a real behaviour change from the TRL default) — TRL's own docs cite the DAPO paper calling this "a good practice for training stability," and it directly addresses this repo's documented truncation risk (worst case 3328 tokens against `max_seq_length: 3600`, only 272 margin) |

**Why SFT's learning rate is flat across tiers.** A 5× LR spread would confound the tier-scale comparison the
three tiers exist to make. 512 steps cannot absorb it either — at 1e-4 eval loss was still improving to ~step
250, so 2e-5 would stop 8b mid-descent. LoRA is also far less LR-sensitive to scale than full fine-tuning
(QLoRA tapered only 2× across a 9× parameter range). A per-tier LR is still allowed — put it in
`model_registry.yaml`'s tier block as declared config, never as a hidden override.

**Why GRPO's learning rate is `1.0e-5` (raised from `2.0e-6`, 2026-09-10).** The earlier sizing argument
below was sound in form and wrong in outcome: it chose the LR so GRPO's cumulative learning "mass" would sit
~1/234 of SFT's, and the resulting runs then could not move the policy at all.

**Measured across all 108 logged steps of all three real vo v1 tiers:** KL to the merged SFT reference
averaged **0.0003-0.0005** and peaked at **0.001**, against `beta: 0.04`. The KL brake never engaged.
`grad_norm` sat at 0.1-0.4 against `max_grad_norm: 0.3`, so clipping rarely bound either. `cosine` decayed
the LR below 1.7e-7 by step ~90 and to exactly 0 at step 108, so the last fifth of every run learned nothing.
Between 41% and 48% of groups had zero reward variance and therefore contributed no gradient at all. Net
effect: SFT→GRPO was **statistically indistinguishable from no change** at 4b (+0.018 F1_micro, p=0.08) and
8b (+0.012, p=0.16). The policy was not mis-steered; it barely moved.

| peak LR | mass | vs SFT | gradient-weighted | outcome |
|---|---|---|---|---|
| `2.0e-7` (v0) | 1.10e-5 | 1/2336 | 1/4970 | inert |
| `2.0e-6` (v1) | 1.10e-4 | 1/234 | 1/497 | **measured: KL ≤ 0.001, effect not significant** |
| **`1.0e-5` (v2)** | **5.50e-4** | **1/47** | **1/99** | to be measured |

`lr_scheduler_type` also moved `cosine` → **`constant_with_warmup`**: there is no overfitting pressure to
anneal against over 2 epochs of a 1732-image pool with a KL anchor, so a flat LR after warmup keeps the whole
budget productive instead of spending its tail at ~0.

**Go/no-go on the 2b smoke run:** `kl` should rise into **0.01-0.10** and `reward/mean` should climb steadily.
If `kl > 0.5`, or reward rises while completions degenerate, drop to `5.0e-6`.

Still safe because three brakes are tight: `max_grad_norm: 0.3`, `beta: 0.04` (KL to the merged reference), and
`scale_rewards="group"` (normalises advantage so reward magnitude can't inflate step size). **A fourth fact,
newly confirmed:** with `num_iterations=1` and `steps_per_generation(4) ≤ gradient_accumulation_steps(16)`,
TRL's own PPO-style clipping is **structurally inert** here — the importance ratio is identically 1.0, so the
realized per-token loss is plain `-A_i + β·KL_i` (group-relative-advantage REINFORCE with a KL anchor), not
clipped-ratio PPO. This doesn't change the LR sizing argument, but it means `max_grad_norm`/`beta` really are
the *only* two brakes doing the clipping job, not one of three.

Verify on the 2b smoke run: `reward/mean` rising and `objective/kl` off zero → proceed; both flat across all
108 steps → try 5e-6; reward rising while output degenerates → drop to 1e-6.

### Three length keys, three different jobs

| Key | Consumed by | Bounds |
|---|---|---|
| `max_seq_length` | `FastVisionModel.from_pretrained` — **both** SFT and GRPO | the model **load** window: a ceiling, not an allocation |
| `SFTConfig.max_length` | HF Trainer, SFT only | prompt **+** target as one sequence, vision tokens included |
| `max_prompt_length` / `max_completion_length` | `GRPOConfig`, GRPO only | the prompt (incl. vision tokens), and **one** rollout's output |

**In SFT, one YAML key feeds both destinations.** `configs/sft.yaml`'s single `max_seq_length: 3072` is read
twice, for two different jobs: `model_loader.py` passes it to `FastVisionModel.from_pretrained` (the load
window), and `sft_trainer.py:227` passes the *same value* as `SFTConfig.max_length` (the training-sequence
ceiling). They can never drift apart in SFT because there is only one number. In GRPO the two are genuinely
separate keys (`grpo.yaml`'s own `max_seq_length` loads the model; `max_prompt_length`/`max_completion_length`
bound the RL sequence).

Two things that regularly trip people up:

- **`max_completion_length` is per rollout, not per group.** Each of the 8 rollouts gets the full budget; it
  is a generation ceiling, not a preallocation — HF `generate` grows its KV cache dynamically.
- **Vision tokens count inside `max_prompt_length`.** The `{"type": "image"}` placeholder expands to
  ~1176–1270 real tokens at the 1.2 MP cap (each output token covers `patch²×merge²` px, measured 1024 on ARC
  — patch 16, merge 2 — so `1204224 / 1024 = 1176` tokens). The measured worst-case prompt is **1519** tokens
  (~233 text + ~1270 vision), not ~233.

A completion that truncates mid-JSON fails the parse, which zeroes **every** reward component — so truncation
is indistinguishable from a terrible model, and it is also indistinguishable from a genuinely-wrong-but-valid
completion at the training signal level: every reward function returns exactly `0.0` for both "parse failed"
and "parsed fine but scored zero" (e.g. all-false-negative grounding). Confirm real target lengths with
`scripts/validate_rewards.py --census` (text only — it reports the vision ceiling separately) before lowering
any budget.

**Live token budgets** (`max_new_tokens` kept equal to `max_completion_length` per task, so inference can
never truncate an output GRPO trained the policy to produce):

| Task | completion | prompt (text + vision) | GRPO 2304 + completion | inference window |
|---|---|---|---|---|
| `unified` | 1024 | ~540 + 1270 = ~1810 | 3328 | 3200 |
| `violations_only` | 1024 | ~390 + 1270 = ~1660 | 3328 | 3200 |
| `object_only` | 768 | ~265 + 1270 = ~1535 | 3072 | 2688 |
| `caption_only` | 768 | ~215 + 1270 = ~1485 | 3072 | 2944 |

Worst case 3328 against `max_seq_length: 3600` — 272 tokens of margin.

### Tasks — what `--task` actually controls

| Dimension | Mechanism |
|---|---|
| Registration, prefix, capabilities, wire format | `core/tasks.py::TASK_REGISTRY` |
| Prompt | `prompt_key` in task YAML → `data/prompt_templates.py::PROMPT_REGISTRY` (the YAML key is descriptive text only — the registry is keyed by task *name*, not by this string; nothing enforces the two stay in sync) |
| Raw-completion parsing | `evaluation/output_parser.py::parse_output_for_task` (JSON vs bare prose) |
| SFT target / GRPO ground truth | `data/preprocessor.py::build_target_json` / `build_gt_dict` — dispatch tables, raise on unknown task |
| Output validation schema | `data/schemas.py::SCHEMA_REGISTRY` |
| Active reward components + weights | `reward_components` / `reward_weights` in task YAML |
| Token budgets | `max_new_tokens`, `max_completion_length`, `inference_max_seq_length` in task YAML |
| SFT input dataset | `sft_dataset_subdir` in task YAML (absent ⇒ the shared augmented default) |
| Every generated name | `core/naming.py` |
| Eval metric families | `evaluation/evaluator.py`, gated on **capabilities** |
| Structural repair transforms | `preprocessing/structural_repair.py`, gated on **capabilities** |

**Capability gating.**

| Capability | Fields | Rewards | Eval metrics | Tasks |
|---|---|---|---|---|
| `caption` | `caption` | `reward_caption` | captioning (needs a JVM + images) | `unified`, `caption_only` |
| `objects` | `excavator`, `rebar`, `worker_with_white_hard_hat` | `reward_grounding` | grounding | `unified`, `object_only` |
| `violations` | `rule_1..4_violation` | `reward_violation_id`, `reward_violation_grounding`, `reward_reasoning` | violations + reasoning (needs a JVM + images) | `unified`, `violations_only` |

`reward_format` is active for every task but means different things: schema-valid fenced JSON for the three
JSON tasks, *clean prose* (no fence, no JSON object, no `"caption":` label, non-blank) for `caption_only` — and
this second meaning is **softer**: `_parse_plain_caption` tolerantly unwraps a stray JSON-wrapped caption
before scoring content, so wrapping the correct caption in `{"caption": "..."}"` only costs the 0.10-weighted
format term (0.9 vs a possible 1.0), not the full reward the way a wrong wire format zeroes everything for the
other three tasks. Likely intentional generosity (recovery signal if the policy reverts to a JSON habit
mid-RL), but worth knowing when comparing `reward_format` curves across tasks — they aren't measuring
comparable failure modes.

Java is required only when a task scores text (`caption` or `violations`). `object_only` evaluates with no JRE
and no images.

**What is shared across tasks:** the base model, `datasets/grpo_pool` (task-blind, one build serves every
pipeline — **do not add a per-task pool**), and the data prep that builds it.

**SFT input is the one thing not fully shared.** `unified`/`violations_only` read `datasets/augmented`.
`object_only`/`caption_only` set `sft_dataset_subdir: datasets/processed`, because augmentation duplicates
images by rare *violation* rule and those duplicates carry identical boxes/captions — no rebalancing for
those two tasks, only overfitting pressure. `run_sft.py` correspondingly skips rare-rule oversampling and
routes the stratified sampler to a different axis for tasks without the `violations` capability:

| Task | rare axis for the sampler | why |
|---|---|---|
| `unified`, `violations_only` | rules 2/3/4 | |
| `object_only` | `rebar` or `worker_with_white_hard_hat` | excavator excluded — too common (2415 occurrences) to mark "rare" without degenerating to a plain shuffle |
| `caption_only` | `None` → plain shuffle | no rare caption axis |

**Measured, ARC 2026-09-02:** on the real `object_only` SFT split, per-class batch-starvation
(`(1-p)^32`, batch 32) is mild — excavator 34.5% prevalence → 0.0% of batches empty, rebar 12.1% → 1.6%,
`worker_with_white_hard_hat` 9.8% → 3.7%. The single boolean mask is sufficient; a multi-label sampler would
buy at most ~4% more exposure for real added complexity. Contrast the *un-augmented* violation axis (the
reason augmentation exists): rule_4 0.67% prevalence → 80.6% of batches empty, rule_2 → 76.3%, rule_3 → 60.7%.
After augmentation all three fall to ≈5–6%.

Oversampling (`oversample_rule24_multiplier`/`oversample_rule3_multiplier`) is currently configured to `1` for
both — **a no-op today**; augmentation alone carries the rebalancing weight. Raising either above 1 would
scale raw image RAM roughly linearly (HF `Image()` features decode a fresh PIL object per duplicate-index
access, they don't cache), on top of the RAM note below.

Per-epoch reshuffling works because the stratified sampler is installed via an `SFTTrainer` subclass
overriding `get_train_dataloader()`, which manually attaches `dataloader.set_epoch = sampler.set_epoch` — a
bare `DataLoader` has no such method, and `get_train_dataloader()` is called once before the epoch loop, so
without the forward every epoch would replay byte-identical order. Confirmed the sampler partitions rare/common
indices with no overlap and no omission, so every index still appears exactly once per epoch regardless.

### Naming and versioning

For `--version v1`, tier `8b`:

| Artifact | unified | violations_only | object_only | caption_only |
|---|---|---|---|---|
| SFT variant | `unified-sft-8b-v1` | `vo-sft-8b-v1` | `oo-sft-8b-v1` | `co-sft-8b-v1` |
| Merged KL base | `merged-unified-sft-8b-v1` | `merged-vo-sft-8b-v1` | `merged-oo-sft-8b-v1` | `merged-co-sft-8b-v1` |
| GRPO variant | `unified-grpo-8b-v1` | `vo-grpo-8b-v1` | `oo-grpo-8b-v1` | `co-grpo-8b-v1` |
| Baseline results dir | `unified-baseline-8b-v1` | `vo-baseline-8b-v1` | `oo-baseline-8b-v1` | `co-baseline-8b-v1` |
| SLURM job / log stem | `vlm-sft-unified` / `sft_unified` | `vlm-sft-vo` / `sft_vo` | `vlm-sft-oo` / `sft_oo` | `vlm-sft-co` / `sft_co` |

`merged_checkpoint_name()` produces byte-identical strings in `submit_pipeline.py`, `hpc_merge_sft.sh`,
`hpc_grpo.sh` and `run_inference.py`'s reverse-engineering regex — confirmed by direct round-trip test for all
4 tasks × 3 tiers × 3 version tags. Keep version tags in `v<digits>` form; `submit_pipeline.py` refuses
anything else up front.

Paths: `$VLM_DATA_ROOT/checkpoints/qwen3vl-<tier>/<variant>/{final,best,checkpoint-N}` and
`$VLM_DATA_ROOT/results/inference/<run_name>/{predictions.jsonl, repair_applied/, evaluation_results/}`.

### Parallel-safety: the isolation guarantee

Multiple task pipelines may run concurrently on ARC at the same `--version` and tier. Nothing coordinates them
at runtime — isolation comes entirely from **every writable path being namespaced by `task_prefix(task)`**:
checkpoints, predictions/repairs/metrics, oversample manifests, SLURM job names/log stems, W&B run/group names,
comparison CSVs and plot dirs. `tests/test_core/test_name_isolation.py` enumerates every writable name for all
four tasks × three tiers × three versions and asserts the set has no duplicates.

Everything the pipelines *share* (`datasets/{processed,augmented,grpo_pool}`, the HF model cache) is read-only
during training, so concurrent readers are safe.

All four `hpc_*.sh` start with `set -eo pipefail` and a guarded `cd` (confirmed correct in every phase script —
**except `scripts/augment_data.sh`**, which has neither). `hpc_merge_sft.sh` refuses to run if no adapter
exists at `<sft_variant>/best`; `hpc_grpo.sh` refuses if the merged KL base is missing. Both guards confirmed
to actually fire.

### Adding a new task pipeline

Six steps, none of them orchestration: (1) `core/tasks.py` — one `TaskSpec`. (2) `configs/tasks/<task>.yaml` —
`task_name`, `prompt_key`, `reward_components`, `reward_weights`, token budgets, optionally
`sft_dataset_subdir`. Every key in `reward_weights` must also appear in `reward_components`. (3)
`data/prompt_templates.py` — new prompt constant + registry entry. (4) `data/schemas.py` — new Pydantic model +
registry entry, fields matching declared capabilities exactly. (5) `data/preprocessor.py` — a target builder
and a ground-truth builder. Boxes: targets scale to `[0,1000]`, ground truth stays `[0,1]`. (6) `tests/` —
mirror the `_oo`/`_co` suites.

**Nothing to add in:** the SLURM scripts, the submitter, the evaluator, structural repair, the reward assembly,
the comparison tables, or the plots — all of those read the registry. No new data prep either: the shared GRPO
pool serves any task.

### Data flow

```
HF hub → datasets/raw → datasets/raw_cleaned → datasets/processed ──→ datasets/augmented → SFT (unified, vo)
                                                        │
                                                        ├──────────────────────────────→ SFT (oo, co)
                                                        └────────→ datasets/grpo_pool ──→ GRPO (all four)
```

**The naming trap:** `configs/base.yaml`'s `processed_subdir` points at `datasets/augmented`, and the
non-augmented base is `raw_processed_subdir` → `datasets/processed`. So a bare `load_processed_dataset()`
returns the **augmented** set. `object_only`/`caption_only` read the un-augmented split via their
`sft_dataset_subdir` key. Inference/eval always uses the default root for every task, so all four pipelines
score on byte-identical test images (augmentation only rewrites `train`).

Augmentation (`data/augment_rare_classes.py`) is **pixel-only** (brightness/contrast, JPEG, gamma) —
deliberately no spatial transforms, so boxes and directional caption phrases stay valid. Multiplicities
`RULE_MULTIPLIERS = {4: 16, 2: 12, 3: 6}` are module-level, override the unused `--num_augmentations` flag, and
precedence is rule_4 > rule_2 > rule_3 (an image tripping several rules is duplicated once, under its rarest
rule). Realised counts (`augment_manifest.json`, ARC 2026-09-02): before/after by rule —
609→871 / 53→689 / 98→696 / 42→714 / safe 5528 unchanged, total 6308→**8198**. Rule spread collapses
14.5:1 → 1.26:1.

`build_grpo_pool.py` composes: the entire val split + every train violation image + enough random train safe
images to reach ~50/50 — measured 701 + 780 + 251 = **1732**, landing at exactly 866/866. It reads
pre-augmentation data because GRPO runs few epochs, where near-duplicate images would produce correlated
reward groups instead of independent signal.

### Rewards and the output contract

`rewards/unified_reward.py::REWARD_COMPONENTS` is the canonical registry; `get_reward_funcs_for_task(task)`
filters and reweights it from the task YAML, building both the function list and the weight list in one loop
(confirmed order-safe — no dict-based reordering risk between funcs and weights).

- `unified` — all 6 components at defaults (format .05, caption .15, grounding .25, violation_id .30,
  violation_grounding .15, reasoning .10). Declares **no** `reward_components`, which is what selects the
  full-registry fallback.
- `violations_only` — 4 components: format **.05**, violation_id **.422**, violation_grounding **.317**,
  reasoning **.211** (2026-09-10; was .10/.40/.30/.20). `reward_format`'s std measured **exactly 0.0000 at
  every logged step** of both the 4b and 8b GRPO runs — post-SFT every rollout in a group is schema-valid, so
  the component contributes no advantage whatever weight it carries. Cut to .05 rather than removed: at 2b it
  still varies (std 0.04-0.13), and a format regression during RL must stay visible in the logged means. The
  freed .05 was redistributed **proportionally**, so each remaining component keeps its share of the
  non-format mass to within 0.3%.
- `object_only` — 2 components: format .10, grounding .90.
- `caption_only` — 2 components: format .10, caption .90.

Expected output for the three JSON tasks is a *flat* JSON object inside a ```` ```json ```` fence, key order
matching `preprocessor.build_target_json` exactly (caption → rules → objects for unified). `caption_only`
emits bare prose, no fence, no keys.

**`bounding_box` scaling: predicted boxes are `[0,1000]`, ground truth stays `[0,1]`** — rewards call
`scale_1000_to_01` on predictions only. Backwards, this silently zeroes every IoU metric (confirmed: a box
already in `[0,1]` scale, divided again by 1000, collapses to a point and scores identically to omitting the
box).

**Required-field asymmetry has a bigger real-world cost than it looks.** `UnifiedOutput`'s object fields
default to `[]`, so a key-name typo (e.g. `"excavators"` instead of `"excavator"`) is silently absorbed as
"nothing detected" — real detection scored as if absent, costing ~9% of that image's total reward.
`ObjectOnlyOutput`'s three class keys are all **required**, so the identical typo fails schema validation
outright — **100%** of that image's reward, loudly. Same mistake, 9% vs 100% cost, by design (an anchor field
that always exists vs. one that doesn't) — but the magnitude gap is worth knowing before reading a `unified`
reward curve, since a systematic key-naming regression there would be far harder to notice than the same bug
in `object_only`.

**A parse/schema failure and a genuinely-wrong-but-valid completion are indistinguishable in the reward
signal.** Every reward function returns exactly `0.0` for both "the JSON didn't parse" and "the JSON parsed
fine but every class was wrong" — the distinction exists in `evaluation/metrics_structural.py`'s parse-failure
counters, but GRPO training itself only ever sees the collapsed `0.0`. Given truncation is a documented live
risk, this is not hypothetical.

**How this becomes a gradient.** Per prompt (one image), 8 rollouts are scored, then
`A_i = (r_i - mean_8) / (std_8 + 1e-4)` — `scale_rewards="group"`, `std` with `ddof=1`. **If all 8 rollouts
score identically, the advantage is ~0 for all 8** (only the `1e-4` floor survives) — zero gradient,
regardless of the absolute score. This is exactly what `frac_reward_zero_std` measures (recorded 0.53 on a
prior run — over half the unique images per update produce a zero-std group). Any component that saturates
within a group (e.g. `reward_format` once SFT reliably emits valid JSON) contributes zero gradient for that
group *regardless of its configured weight* — a 0.90-weighted component that's saturated is exactly as inert
as a 0.05-weighted one. `object_only` has a structural version of this at scale: on the 50.87% of the pool
with no target object, the honest answer and the "always predict empty" answer are the *same completion*, so
those rollouts contribute no gradient no matter how the policy is doing — a throughput problem, not a bias
one (the images are still valid supervision for *when* to abstain, via SFT).

**The reward-hacking exploits this was originally built to close, and their current status:**

- **Reflexive rule_1 flagging vs honest abstention** (`unified`, `violations_only`): closed. At the historical
  flat `violation_tn_constant: 0.15`, always-asserting rule_1 beat honest abstention (~0.391 vs 0.075 EV) by
  5×. Now `0.85`; `scripts/validate_rewards.py --probe` passes.
- **Suppressing the two rare object classes** (`object_only`, `unified`): closed. At flat `c=0.15` the
  break-even IoUs for rebar/hard-hat were 1.55/1.15 — above 1.0, i.e. unreachable, making suppression strictly
  dominant. Per-class retuned constants (excavator 0.283, rebar 0.048, hard-hat 0.065) put all three
  break-evens at ≈0.48–0.53, confirmed against the real measured pool prevalences (excavator 0.3545, rebar
  0.0837, hard-hat 0.1189).
- **Contentless violation assertion** (`{"reason":"","bounding_box":[]}`): closed by requiring *substance*
  (non-empty reason or ≥1 box) for TP credit in `reward_violation_id`, while still counting the assertion as a
  false alarm on a safe image via presence alone. Scores as a **miss** on a real violation and a **false
  alarm** on a safe image — never a hit.
- **`object_only`'s "detect only the common class" hack:** closed, but not by the TN constant — a
  common-class-only policy that emits an excavator box on a truly-safe image scores *worse* (FP penalty) than
  honest abstention on exactly the images where hedging would help.
- **The remaining, not-closeable-by-a-constant issue** is the 50.87%-no-object throughput problem above —
  read the real `frac_reward_zero_std` off the 2b smoke run before concluding anything needs to change; the
  lever, if it becomes worth pulling, is GRPO pool composition (currently balanced on violations only, not
  object presence).

## Invariants and known traps

**Never break these:**

1. **The GRPO dataset image column must be named `image` (singular).** TRL's rollout code looks for exactly
   that key; the prompt carries only a `{"type": "image"}` placeholder, never an inlined PIL object. Confirmed
   directly in `data/preprocessor.py::to_grpo_prompt_for_task`. Verify with `scripts/verify_grpo_images.py`.
2. **GRPO must run against the merged SFT model** (`--base_model_override`). TRL computes its KL reference via
   `disable_adapter()` **only because `beta=0.04 != 0`** (TRL skips loading a reference model entirely at the
   default `beta=0.0`) — so without the merge, or if `beta` were ever zeroed, the reference silently becomes
   the raw pretrained base. `run_grpo.py`/`run_inference.py` hard-refuse without `--base_model_override`;
   `--allow_unmerged_reference` is a smoke-test escape hatch only.
3. **The GRPO pool is mandatory** — no fallback. `python data/build_grpo_pool.py` or GRPO crashes with
   `FileNotFoundError`.
4. **Merged checkpoints need `tokenizer_name`** pointing at the original HF repo. Unsloth's processor
   auto-detection silently degrades to text-only when loading a local `qwen3_vl` directory.
5. **`steps_per_generation` must be passed explicitly to `GRPOConfig`.** If omitted, TRL defaults it to
   `gradient_accumulation_steps`, silently changing the generation-batch shape.

   Two different quantities, easy to confuse: `K = (per_device_bs × steps_per_generation) / num_generations`
   — genuinely different images sharing one `generate()` call (the collapse-risk number); vs.
   `unique_images_per_update = (per_device_bs × gradient_accumulation_steps) / num_generations` — the
   effective RL batch (`steps_per_generation` cancels out of this one). Current: `16×4/8=8` (K) and
   `16×16/8=32` (unique images/update). K = 2/4/8/16/32 all verified NO COLLAPSE on real images. **Verified
   against real TRL 0.23.0 source**: `generation_batch_size` (not `gradient_accumulation_steps`) is what TRL's
   own divisibility check uses whenever `steps_per_generation` is set explicitly, as it is here — `64 % 8 == 0`,
   accepted cleanly.
6. **The true-negative constants are per-task configuration**, read via `rewards/reward_utils.py::reward_constant`
   (lru-cached): `grounding_tn_constant` (per-class, `reward_grounding.py`) and `violation_tn_constant` (all
   three violation reward sites move together by construction). See [Rewards](#rewards-and-the-output-contract)
   for the derivation and current values. A related, less-documented sibling constant: `violation_fbeta`
   (default `2.0`, recall-weighted F-beta, no task YAML currently overrides it). `repetition_penalty`
   (`rewards/reward_utils.py::reward_constant(task, "repetition_penalty", 1.0)`) is the fourth — currently
   `1.0` (disabled) for every task by explicit decision; see the ghost-variable table above.
7. **`final/` is the SFT handoff; `best/` is a diagnostic (changed 2026-09-10).** `load_best_model_at_end:
   false`, so `final/` is the literal end-of-training state and `best/` is the lowest-`eval_loss` state,
   written eagerly on every improvement. **The merge → GRPO handoff consumes `final/`, post-SFT eval runs
   `--checkpoint final`, and the results directory is `<variant>_final`.** Nothing reads `best/`; it is kept
   only so a run can be inspected for "what would eval_loss have chosen?".

   **Why the switch.** `eval_loss` cannot rank checkpoints on this task. 86.3% of images are safe, so 86.8% of
   the SFT target character mass is the fixed all-null skeleton, and after ~step 125 the loss sits in a 5-11%
   band that is pure noise — at 8b the step-to-step jitter (0.05532 → 0.06166, 11%) exceeds the total
   improvement still available (1.9%). `early_stopping_patience` was therefore counting noise, not a plateau,
   and stopped every v1 tier early: **2b at step 300 of 512, 4b and 8b at 375** — 27-41% of the budget
   discarded, with `best/` frozen at step 200/275/275. It is now `null` (disabled) and runs spend the full
   budget.

   The switch renames the SFT results directory, so `core/naming.py::results_dir_names` emits `_final` for both
   trained phases. `experiments/results_lib.py::parse_run_name` still **accepts the legacy `_best`** for `sft`
   only, so an index built over pre-switch results keeps working instead of reporting a whole phase as
   unparseable.

   GRPO never had a `best/` (no eval dataset) — `hpc_grpo.sh` was always on `--checkpoint final`.
   `save_total_limit: 3` only rotates numbered `checkpoint-N` directories; `best/`/`final/`/
   `persistent-checkpoint-N` are distinct names it cannot touch — confirmed by directory-name disjointness,
   not merely by absence of deleting code.

8. **LoRA rank is per-tier, and it is resolved in `models/model_loader.py`, not in `core/config.py`.**
   `configs/model_registry.yaml::lora_by_tier` is a **top-level** mapping (`2b: r=16, 4b: r=20, 8b: r=32`,
   with `alpha` tracking `r`). It cannot live under `models.<tier>`: `merge_configs` descends only one level
   into nested dicts, so anything there is never visible as a top-level key and would be silently inert.
   `resolve_lora_config(sft_cfg, tier)` layers it over the training config's own `lora:` block and is called
   from `load_model_for_training`, the one function every *fresh-adapter* path goes through — so SFT and GRPO
   cannot drift apart. The merge stage passes `adapter_path`, which reads rank from the adapter's own
   `adapter_config.json`, so it is unaffected.

   **Why per-tier.** LoRA parameters grow ~linearly with hidden size while base parameters grow
   quadratically, so a fixed `r=16` adapts a *smaller fraction* of a bigger model. Measured from the v1 SFT
   logs: 2b **1.10%** trained (23.7M/2.15B), 4b **0.88%** (39.3M/4.48B), 8b **0.58%** (51.3M/8.82B) — the 8B
   model was adapted at half the relative capacity of the 2B one, so "does scale help?" and "does adapter
   capacity help?" moved in opposite directions and could not be separated. Measured outcome: 4b→8b was
   negative and non-significant in both phases (SFT −0.001 p=0.53, GRPO −0.007 p=0.64). The v2 ranks re-level
   the adapted fraction to ~1.1% everywhere.

   **`alpha` must track `r`.** LoRA scales its update by `alpha/r`; holding `alpha` at 16 while raising `r`
   would shrink the effective update to 0.8× at 4b and 0.5× at 8b — reintroducing exactly the confound this
   removes. `resolve_lora_config` defaults `alpha` to `r` for that reason; an explicit `alpha` still wins.

**Violation semantics — two predicates, deliberately different.** `_is_violation_present` (presence) drives
precision — an assertion, even contentless, is still a prediction, so it's penalised as a false positive on a
safe image. `_is_substantive_violation` (presence **and** a non-empty reason or ≥1 box) drives recall — a
contentless assertion earns no true-positive credit. So a contentless assertion scores as a **miss** on a real
violation and a **false alarm** on a safe image — never a hit. Note: `reward_violation_grounding`'s TP set uses
plain presence, not substance, on both sides — the net score still lands at 0 for a contentless prediction
(empty-box IoU is 0), but via a different code path than its sibling `reward_violation_id`'s explicit
substance gate. The two currently agree by coincidence of that math, not by a shared predicate — a future
change to `require_violation_substance` would not automatically produce matching behaviour in both.

`null` is the only safe signal — even `{"reason": "", "bounding_box": []}` is an assertion. The one exception
is a bare `{}` (no keys, no assertion), normalized to `null` by `normalize_violation_value`.
`preprocessing/structural_repair.py` manufactures these shapes (rewrites bare `true` into a contentless
violation object, a bare reason string into a box-less one) — all of it capability-gated (`violations` only;
object-box repairs gated on `objects`; caption list-join on `caption`). `caption_only` takes a separate
plain-text repair path that unwraps a stray fence/`{"caption": ...}` and writes bare prose back.

**A LIST value for `rule_N_violation` is merged, not dropped.** Two shapes occur in real output and both
survive: a list of violation objects (one per instance -- the prompt itself asks for this) has its boxes
unioned and its reasons joined after order-preserving de-duplication; a list of bare boxes
(`[[0,0,1000,999]]`, the wrapping object omitted) becomes `{"bounding_box": [...], "reason": ""}`, inventing
nothing. Only a list with no usable box *and* no usable reason still collapses to `null`. This branch used to
`return None` unconditionally, which does not drop a repair -- it **inverts the model's answer** into "this
rule was not violated". Measured on `vo-baseline-2b-v1` it fired on 1260 of 3004 records (42%) and destroyed
182 true positives, dragging that run's recall from 0.887 to 0.469; because it fired on ~0% of the SFT/GRPO
runs it also made one column of the comparison table silently non-comparable with the rest. Logged as
`violation_list_merged`; `violation_list_dropped` now means only the genuinely-empty case.

`rule_0` is the pseudo-rule for the safe class: TP = correctly said safe, FN = false alarm on safe, FP = missed
a real violation. `violation_identification_recall_rule_0` = 1 − false-alarm rate. Parse/schema failures are
never credited rule_0 TP; they surface as `violation_prediction_failure_{count,rate}`.

**Macro averaging: TP-conditioned families skip unmeasured rules; identification families do not.** Reasoning
(`evaluation/metrics_reasoning.py`) and violation grounding (`metrics_violations.py`) can only be scored on a
rule the model *correctly identified* at least once -- with zero true positives there is no reasoning text and
no predicted box, so there is no measurement. Both macros therefore average only over rules that have data,
and publish the divisor as `<metric>_macro_n_rules`, which `compare_all.py` renders inline as `0.7817 (n=2)`.
They used to substitute a placeholder `0.0` and still divide by 4, which reported `vo-2b-sft` -- zero rule_3
and zero rule_4 detections -- as `bertscore_f1_macro = 0.391` when the two rules it *could* be scored on
averaged 0.782, and as `mask_iou_macro_tn0 = 0.208` against a real 0.415. That run read as a catastrophic
regression below its own untrained baseline on both families; it was an averaging artifact.
**Do not extend this to `violation_identification_*_macro`**: an identification metric has a well-defined
denominator even at zero true positives (rule_3 has 63 ground-truth positives whether or not the model finds
any), so an F1 of `0.0` there is a real, earned zero and must keep its full weight in the mean.

Every reasoning score is conditioned on the model's *own* true positives, so two runs are literally scored on
different image populations -- on the real vo runs `4b/grpo`'s rule_3 reasoning averages over 22 images and
`8b/grpo`'s over 8, and 8b scores higher partly because it only had to explain its 8 easiest cases. The
`scored_count_*` keys are always emitted (including as `0`) and printed in the support-counts table for
exactly this reason. Per-rule CIDEr-D is computed at every sample size and is deliberately **not** gated:
read it against its `scored_count`, because it is a corpus-level TF-IDF metric and a value derived from n=2
references carries no information.

**Debugging notes:**

- `_safe_reward` swallows exceptions and returns `0.0` at WARNING level — a broken reward is indistinguishable
  from a bad model. Grep SLURM stderr for `"Error in reward function"` before trusting a low score.
- Watch `frac_reward_zero_std` (W&B, or the offline log — see [W&B runs fully offline](#commands) above): a
  prior run showed 0.53. `reward_format/std` at 0.0 post-SFT is expected (saturated), not a bug.
- `object_only`'s ~0.5 `frac_reward_zero_std` floor is structural (see the throughput note above), not a sign
  GRPO is broken for that task.
- Pre-warm the `all-MiniLM-L6-v2` cache (`SENTENCE_TRANSFORMERS_HOME`); the caption/reasoning rewards download
  it mid-training otherwise.
- SFT/baseline's `--mem=150G` is load-bearing, not padding: `build_sft_dataset` fully materializes every
  decoded PIL image of the entire split (up to 8198 rows) into one Python list before training starts, with no
  streaming path.
- The merge stage writes no manifest of its own (unlike SFT's `run_config.json` and GRPO/inference's
  `run_manifest.json`) — no on-disk record of which adapter/task/git-commit produced a given merged checkpoint.

**Stale things — don't trust them:**

- `setup_project_structure.py` — dead one-time bootstrap. Gitignored yet tracked. Don't run it.
- `rewards/{json_validity,caption_quality,rule_violation_accuracy,grounding_iou}.py` — legacy, unwired.
- `rewards/reward_utils.py::_strict_parse`/`_strict_parse_cached` and
  `evaluation/output_parser.py::validate_unified_output` — pre-task legacy path hardcoded to `UnifiedOutput`.
  Use the `_for_task` versions.
- `data/preprocessor.py`'s task-blind `raw_sample_to_conversation`, `build_unified_sft_dataset`,
  `to_grpo_prompt`, `build_grpo_dataset` — superseded by the `_for_task` versions the pipeline actually uses.
- `data/dataset_cache.py` — unreferenced dead code.
- `experiments/run_dual_evaluation.py` — returns a nested metrics shape nothing reads any more; semi-stale.
- `evaluation/report_generator.py`, `evaluation/error_analyzer.py` — not called from any live pipeline path.
- `docs/Metrics.md` — documents a metric namespace that no longer exists (live families are
  `grounding_{mask,greedy}_iou_{all,exist}_*`).
- **All pre-existing GRPO metrics (from before this repo's `b8f2470`) are void** — those runs trained
  prompt-only, images never reached the model. Baseline and SFT numbers from that era are usable; draw no GRPO
  conclusion from them.
