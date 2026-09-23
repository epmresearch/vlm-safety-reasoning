# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the **engineering reference**: why the code is shaped the way it is, what is already decided, and what
will silently break if you change it. For anything else, see the documentation map below.

---

## Read this first — project status (2026-09-23)

**Nothing is running on the cluster.** The last SLURM job finished 2026-09-15T20:37 (`48521188`, the 8B GRPO
job of the v2 `violations_only` run). Nothing needs babysitting; `squeue -u $USER` is empty.

> ### ⚠️ The working tree is dirty, and it matters
>
> Last commit is `3809e34` ("final" — the MOCS review UI). **26 modified + 10 new files are
> uncommitted**: the whole `violations_think` arm, plus changes to shared code every task runs
> through (`core/tasks.py`, `data/preprocessor.py`, `evaluation/evaluator.py`,
> `experiments/results_lib.py`, `compare_all.py`, `submit_pipeline.py`, `validate_rewards.py`, all
> four `hpc_*.sh`). Any job submitted now runs *on top of* those changes and records
> `git_is_dirty: true` — the same problem as all nine v2 manifests (`README_v2.md` §13 P0-4).
>
> Always run `git status --short && git log --oneline -3` before assuming anything about the tree.

**Session context and the next job live in [`HANDOFF.md`](HANDOFF.md)** — what the last session did,
what is settled, and the audit brief that precedes the next submission.

| Task | v1 | v2 | notes |
|---|---|---|---|
| `violations_only` (`vo`) | 9/9 runs, **superseded** | **9/9 runs — the current result** | the only task with evaluated results |
| `unified` | SFT started at all three tiers (`datasets/stats/oversample_manifest_*_unified-sft-*-v1.json` exist); **no indexed inference/eval results** | not started | |
| `object_only` (`oo`) | never run | never run | code complete, tested, never submitted |
| `caption_only` (`co`) | never run | never run | code complete, tested. Two 4-job `co-*-v1` chains were submitted **by accident** on 2026-09-14 (`48501131`–`48501137`) — see [the pytest-on-ARC hazard](#tests); all cancelled, every artifact deleted and verified gone |
| `violations_think` (`vt`) | — | — | **new 2026-09-21**, never run. Code complete, tested and independently audited; blocked on `datasets/augmented_v3` + `datasets/grpo_pool_v3`, which do not exist yet. See [`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md) |

**Next up: `object_only` and `caption_only` at 2b/4b/8b, `--version v1`** — 24 jobs, the first end-to-end
runs either has ever had. Neither needs the MOCS harvest (MOCS carries none of the three object classes and
its captions are teacher output, not ground truth), so they are data-complete today. Audit brief in
[`HANDOFF.md`](HANDOFF.md) §6.

**The v2 `violations_only` run is complete, analysed, and reported.** All numbers, confidence intervals,
paired significance tests, per-rule breakdowns, the comparison against the dataset paper's Tables 7 and 8, the
root-cause analysis and the costed v3 plan live in **[`README_v2.md`](README_v2.md)** — that file is the single
source of truth for results. Do not re-derive results here; link to it.

**The six load-bearing v2 findings** an agent needs before touching anything (full evidence in `README_v2.md`):

1. **SFT is the big win and it is significant everywhere** (+0.21 to +0.30 F1 micro, *p* < 0.001 at all tiers).
   GRPO then adds +0.07 to +0.09 F1 micro, *p* < 0.001 at all tiers — recall +17 to +22 points with **no
   significant precision change**. In v1 the same GRPO comparison was *p* = 0.08 / 0.16 (noise).
2. **Macro-F1 is flat from SFT to GRPO (*p* = 0.33–0.52), and that is the reward working as specified, not a
   bug.** `violation_tn_constant: 0.30` implies a break-even confidence `p* = 0.298`, and GRPO's measured
   marginal precision — 0.370/0.370/0.388 on the flags it added, 0.077–0.250 on the flags it removed — sits
   either side of that line at every tier. One global threshold is ~2.5× too high for the rare rules.
   **Do not "fix" this by changing the LR, the step count, or the model size.**
3. **The GRPO pool is 14.7 : 1 rule_1 : rule_4** (677 vs 46 images) while SFT's augmented set is 1.26 : 1. GRPO
   both devalues *and* barely sees the rare rules.
4. **8B is worse than 4B after fine-tuning** (−0.033 F1, ns) although the 8B *baseline* is the best baseline.
   The v1 LoRA-capacity hypothesis is **dead** — v2 re-levelled ranks to 1.10/1.10/1.16% trained and the
   deficit survived. It is overfitting: 8B's best `eval_loss` is at step **125 of 512** and drifts +11.2%.
5. **`structural_json_validity_rate` is measured after structural repair.** It reads 0.981 for `vo-baseline-2b`
   whose real raw compliance is **20.11%**. Quote `repair_stats.csv::status:valid_raw:pct` instead. Repair is
   load-bearing for exactly that one run (F1 0.0155 → 0.1529); for the other eight it moves F1 by ≤ 0.008.
6. **The remaining error is ranking, not perception.** An oracle keeping only the correct flags from SFT ∪ GRPO
   scores F1 0.83–0.86 against the 0.60 realised, and no cheap post-hoc confidence proxy works (box count, box
   area, reason length, rules-per-image, template frequency — every AUC 0.44–0.59). The model exposes no
   confidence. Creating one is the highest-value v3 change.

**What comes next.** The prioritised, costed plan is [`README_v2.md` §13](README_v2.md#13-v3-plan-costed-and-prioritised).
In order, and none of it is started:

- **P0-1** evaluate the SFT `best/` and `persistent-checkpoint-{100..500}/` checkpoints (already on disk — see
  [invariant 7](#invariants-and-known-traps)) to test the 8B overfit hypothesis. No retraining.
- **P0-2** evaluate `merged-vo-sft-<tier>-v2` with no adapter, to separate RL from the 4-bit merge round trip.
- **P0-3** emit `structural_*_raw` from the pre-repair file.
- **P0-4** re-run `validate_rewards.py` and the test suite, then commit (all nine v2 manifests say
  `git_is_dirty: true`).
- **P1/P2** config changes (`max_grad_norm`, `num_generations`, pool rebalance, per-rule reward weighting) and
  then the confidence work. Rationale for each is in §13 — read it before proposing an alternative.

---

## Documentation map

Four documents, four jobs, no overlap, plus the figures one of them embeds. Update the right one — if you find
yourself copying a paragraph between two of these, it is in the wrong file.

| File | Owns | In git? |
|---|---|---|
| [`README.md`](README.md) | what the project is, why four pipelines, a short results summary, getting started | yes |
| [`README_v2.md`](README_v2.md) | **all results**: v2 tables, CIs, significance, paper comparison, root-cause analysis, the v3 plan | yes |
| **`CLAUDE.md`** (this file) | architecture, config layering, invariants, decisions already made, traps | yes |
| [`OPERATIONS.md`](OPERATIONS.md) | the ARC/SLURM runbook: setup, submitting, monitoring, failure recovery, artifact cleanup, result extraction | yes |
| `figures_v2/` | the 15 figures `README_v2.md` embeds | yes |

Three **temporary** documents also exist. Each says so at the top and each is meant to be deleted once its
work has landed — do not let them become a fifth and sixth source of truth:

| File | Owns | Delete when |
|---|---|---|
| [`HANDOFF.md`](HANDOFF.md) | session state, what is settled, the current audit brief | the oo/co runs land |
| [`PLAN_V3_THINK.md`](PLAN_V3_THINK.md) | the `violations_think` design | the arm has run |
| [`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md) | what was built for it, and what was deliberately skipped | the arm has run |

**`docs/` is git-ignored in its entirety** (`.gitignore:23`). It holds a large local-only working archive —
`docs/audit/`, `docs/Diagnosis/`, `docs/Jobs/` (including the full v1→v2 conversation transcripts),
`docs/logs/`. Nothing there is authoritative and none of it reaches a clone. **Never put a doc a future agent
needs in `docs/`** — it will not exist for them. Root-level `*.md` and `figures_v2/` are tracked.

---

## Overview

Research project fine-tuning Qwen3-VL (2B/4B/8B via Unsloth) for construction safety inspection on the
`LouisChen15/ConstructionSite` dataset (6308 train / 701 val / 3004 test, un-augmented). Training is two-phase:
**LoRA SFT → merge adapter → GRPO** on the merged model.

**This repo runs a family of parallel pipelines, not one pipeline.** Each *task* is a full, independent
baseline→SFT→merge→GRPO→eval pipeline over the same images and the same base model, differing only in what the
model is asked to output. All five are live:

| Task | Prefix | Output | Wire format | Capabilities |
|---|---|---|---|---|
| `unified` | `unified` | caption + 3 object classes + 4 rule violations | fenced JSON | caption, objects, violations |
| `violations_only` | `vo` | 4 rule violations only | fenced JSON | violations |
| `violations_think` | `vt` | a `<think>` block, then the **byte-identical** `violations_only` JSON | fenced JSON (with a preamble) | violations |
| `object_only` | `oo` | 3 object classes only, boxes `[0,1000]` | fenced JSON | objects |
| `caption_only` | `co` | one scene description | **bare prose** (no JSON, no fence) | caption |

**`violations_think` is `violations_only` plus a reasoning preamble, and nothing else.** Same Instruct
weights, same schema OBJECT, same ground-truth builder, same four reward components at the same weights;
its target builder delegates the JSON half to `_build_violations_only_target_json` verbatim. It is a
separate task purely so `vo-*-v2` stays byte-reproducible. Not started as of 2026-09-21 — see
[`PLAN_V3_THINK.md`](PLAN_V3_THINK.md) and [`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md).

Four of the five emit **one flat JSON object** per image. `caption_only` is the exception: the caption *is* the
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

## Where things live

### In the repo (all paths relative to the repo root)

| Path | What |
|---|---|
| `core/` | `tasks.py` (the task registry — the single place a task is registered), `naming.py` (every generated name), `config.py` (the merge chain), `constants.py` (`RULES`), `think_format.py` (the `<think>`-block wire format + its row validator), `callbacks.py`, `run_manifest.py`, `logging.py`, `io.py`, `wandb_utils.py` |
| `configs/` | `base.yaml` → `model_registry.yaml` → `{sft,grpo}.yaml` → `tasks/<task>.yaml`, merged last-wins |
| `data/` | `preprocessor.py` (SFT targets + GT dicts + GRPO prompts), `prompt_templates.py`, `schemas.py`, `loader.py`, `samplers.py`, `oversampling.py`, `box_utils.py`, `augment_rare_classes.py`, `build_grpo_pool.py` |
| `models/` | `model_loader.py` (loading + LoRA resolution + pixel bounds), `sft_trainer.py`, `grpo_trainer.py`, `inference.py` |
| `rewards/` | `unified_reward.py` (the registry + assembler), `reward_{format,caption,grounding,violation_id,violation_grounding,reasoning}.py`, `reward_utils.py` (predicates + `reward_constant`) |
| `evaluation/` | `evaluator.py` (orchestrator), `metrics_{structural,violations,reasoning,grounding,captioning,llm_judge,think}.py`, `output_parser.py` |
| `preprocessing/structural_repair.py` | the repair stage between inference and evaluation |
| `experiments/` | entry points `run_{sft,grpo,inference,evaluation}.py`; the analysis toolset `build_results_index.py` + `compare_all.py` + `results_lib.py` + `results_charts.py`; `extract_qualitative.py` |
| `scripts/` | `submit_pipeline.py` + five `submit_*_pipeline.py` shims; four `hpc_{baseline,sft,merge_sft,grpo}.sh` phase scripts; `merge_sft_adapter.py`, `validate_rewards.py`, `validate_think_dataset.py`, `preflight_grpo.py`, `fetch_results.py`, `dataset_report.py`, `setup_arc.sh`, `augment_data.sh` |
| `tests/` | 798 tests (797 pass, 1 skipped). `test_core/test_blocker_fixes.py` is the pre-flight/regression suite (B1–B14 plus the `test_v2_*` block that pins every v2 decision and `test_v3_*`, which pins that the violations_think arm is purely additive) |
| `results_index/` | **git-ignored.** Local analysis workspace: `v2_dump/` (the downloaded v2 run dump), `analysis_v2/` (the local `compare_all` output: CSVs + `significance.csv` + 120 charts), `results_folder_vo/` + `logs_folder_err_out_vo/` + `all_vo/` (v1 archive), `index.json` |
| `figures_v2/` | the 15 report figures, tracked so `README_v2.md` renders on GitHub |
| `docs/` | **git-ignored** local working archive — see the documentation map above |
| `vlm_data_root/` | the local mirror of the HPC data root; git-ignored |

### On ARC (`VLM_DATA_ROOT=$HOME/vlm-finetuning-project1`)

```
$VLM_DATA_ROOT/
├── datasets/
│   ├── raw/ raw_cleaned/ processed/        # 6308 / 701 / 3004 train/val/test
│   ├── augmented/                          # 8198 rows + augment_manifest.json   (SFT: unified, vo)
│   ├── grpo_pool/                          # 1732 rows + build_manifest.json     (GRPO: all four tasks)
│   └── stats/                              # dataset_report.json, oversample_manifest_<tier>_<variant>.json
├── checkpoints/qwen3vl-<tier>/
│   ├── <variant>/{final,best,checkpoint-N,persistent-checkpoint-N}
│   │                                       # + run_config.json, run_manifest.json, training_state.json
│   └── merged-<task>-sft-<tier>-<ver>/     # the 16-bit merge; GRPO's KL reference
├── results/inference/<run_name>/
│   ├── predictions.jsonl                   # RAW model output
│   ├── run_manifest.json                   # inference settings + prompts
│   ├── repair_applied/                     # predictions_repaired.jsonl, repair_report.json,
│   │                                       # change_manifest.json, still_broken.json
│   └── evaluation_results/                 # metrics.json, eval_manifest.json, predictions_with_eval.json,
│                                           # parsed_predictions.json, json_parse_failures.json,
│                                           # schema_validation_failures.json,
│                                           # llm_judge_status.json, llm_judge_details.json
├── logs/                                   # <log_stem>_<jobid>.{out,err} + the Python-side run_*.txt
└── (HF cache is separate: $HOME/scratch/hf_cache, set by every phase script)
```

`<run_name>` comes from `core/naming.py::results_dir_names`: `<prefix>-baseline-<tier>-<ver>` for the baseline,
`<prefix>-{sft,grpo}-<tier>-<ver>_final` for the two trained phases. `results_lib.py::parse_run_name` is the
exact inverse and still accepts the legacy `_best` suffix for `sft` only.

**Getting results off ARC** is an operational procedure — see [`OPERATIONS.md`](OPERATIONS.md).

## Commands

### Tests

```powershell
python -m pytest tests/ -v                                       # all 798, no GPU needed (~50 s)
python -m pytest tests/test_core -v                               # task registry + name-isolation proof
python -m pytest tests/test_core/test_blocker_fixes.py -v         # blockers B1-B14 + the test_v2_* decisions
python -m pytest tests/ -k "_oo or _co" -v                        # just the two newer pipelines
```

> ### ⛔ NEVER run the full test suite on ARC
>
> `tests/test_core/test_blocker_fixes.py::test_submitter_can_override_gres_for_every_stage` runs the **real**
> submitter twice through `subprocess`, with `--task caption_only --tiers 2b --version v1 --skip-preload`.
> `submit_pipeline.py::submit_job` shells out to `sbatch` and only falls back to `DUMMY_JOB_ID` on
> `FileNotFoundError` — so on a login node, where `sbatch` exists, **the test submits 8 real GPU jobs**. This
> happened on 2026-09-14 (`48501131`–`48501137`, cancelled and cleaned; see [`OPERATIONS.md`](OPERATIONS.md)).
>
> On ARC, always: `pytest tests/ -k "not submitter_can_override_gres"`. Locally it is harmless — Windows has no
> `sbatch`. The proper fix (make the test unable to reach a real `sbatch`, e.g. by pointing `PATH` at a stub or
> asserting on a dry-run flag) is **not done yet**.

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

> This section describes **what the submitter does**. The step-by-step procedure for actually running a version
> on ARC — environment setup, caching the judge, pre-flight checks, monitoring, holding/releasing queued jobs,
> recovering from `NODE_FAIL` / `DependencyNeverSatisfied` / walltime kills, deleting the artifacts of a failed
> run, and packing results for download — is [`OPERATIONS.md`](OPERATIONS.md).

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
headroom and is kept at 16, not reverted to its old fallback of 8, on the strength of the 2b measurement above.
**Settled by v2: all three tiers ran GRPO to completion on a single H100 with no OOM** (`48501343`,
`48521188`, `48501351`; zero `CUDA out of memory` in any log). The `per_device_train_batch_size: 16` question
is closed — leave it at 16.

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

**24h is now confirmed sufficient, measured on the real v2 runs** (each figure is the whole job: training +
inference over 3004 images + repair + evaluation + LLM judge):

| tier | baseline | SFT | merge | GRPO |
|---|---|---|---|---|
| 2b | 1:19:02 | 1:09:08 | 0:01:02 | **5:23:31** |
| 4b | 1:18:36 | 1:45:55 | 0:03:30 | **9:21:57** |
| 8b | 0:47:37 | 2:01:50 | 0:01:48 | **11:26:53** |

8B GRPO is the binding case at ~11.5 h against the 24 h wall — comfortable, but a change that raises
`num_generations` or `max_completion_length` eats that margin directly. If a GRPO job *is* killed by the wall
it is not lost: `models/grpo_trainer.py` auto-resumes from the last checkpoint (`save_steps: 20`, so at most
~20 steps at risk) — the response to a timeout is to **re-submit the identical GRPO job** (same variant name),
which continues rather than restarts.

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

**Cost — measured on the nine v2 runs, not estimated.** `147–288 s per evaluation` (193–387 judged items),
`status: ok` and **0 unparsed replies in all nine**, identical `rubric_sha256 28297153fe9d`. It fits inside an
80 GB H100 alongside BERTScore + CLIP with no memory trouble. The earlier "5–15 min" estimate was ~4× too
pessimistic.

**Red flags, and what v2 actually tripped:** `unparsed_rate_micro > 0.02` — **not tripped** (0 everywhere);
`total_rule_1` far outside 2–6 — **not tripped** (3.23–4.64); *nearly every item at 6/6* — **tripped, and it
is real.** 60% of 4B-SFT's judged items score a perfect 6/6, and `total_rule_4 = 6.00` on four of the nine runs
because the model emits essentially one templated sentence (82 distinct sentences serve 230 judged items) on
n = 4–12 items. **`reasoning_llm_judge_total_rule_4 = 6.0` is a template artifact, not a perfect reasoning
score** — see [`README_v2.md` §11.3](README_v2.md#113-the-reasoning-scores-reward-style-not-only-substance).
`relevance` is additionally near-saturated by construction (1.68–1.95 of 2), because our JSON format tells the
judge which rule the sentence belongs to; the paper's free-form models had to select the rule in prose.

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
four such numbers. On the v1 runs, baseline→SFT was +0.24 (p<0.001) while SFT→GRPO was +0.018 (p≈0.08) at 4b
and +0.012 (p≈0.16) at 8b. **On v2, SFT→GRPO is +0.091/+0.072/+0.074 at p<0.001 for all three tiers** — the
same point-estimate direction, now out of the noise. 4b→8b remains negative and non-significant in both trained
phases at both versions. Reading point estimates alone makes phase and tier rankings look like they flip run to
run; they do not, they are inside the noise until a paired test says otherwise.

**Two things the toolset does *not* do**, both worked around by hand for the v2 report:

- **It never pairs across `--version`.** `compare_all.py::print_significance` only compares runs inside one
  `(task, tier, version)`. A v1↔v2 comparison has to be built separately, and v1's `metrics.json` predates
  `violation_per_image_outcomes_b64`, so v1's outcome vectors must be **reconstructed** from
  `repair_applied/predictions_repaired.jsonl` with `parse_output_for_task` + `validate_output_for_task` +
  `build_gt_dict` + `_is_violation_present` (verified by reproducing all nine v2 numbers exactly by the same
  route). Adding a `--baseline-version` flag would be a genuine improvement.
- **It only bootstraps `f1_micro` and `f1_macro`** (`results_lib.py::BOOTSTRAP_METRICS`). Precision, recall,
  F2, per-rule and image-level intervals were computed outside the toolset for `README_v2.md`.

**`experiments/results_charts.py` used `cm.get_cmap(name, lut)` at three call sites** — deprecated in
matplotlib 3.7 and **removed in 3.9**, which is what ARC and any recent local env have, so chart generation
raised `AttributeError` and the corresponding test failed. Fixed 2026-09-16 with a version-safe `_cmap()`
helper (`matplotlib.colormaps[name].resampled(n)`, falling back to `cm.get_cmap` below 3.5). 120 charts now
generate; keep new chart code off the removed API.

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
| `dataset.grpo_pool_subdir` in `base.yaml` | Read by `data/loader.py::load_grpo_pool` from `load_base_config()` **only**, never from the merged task config — so a task YAML setting it was silently ignored. The key existed, looked overridable, and was not: a `violations_think` run would have trained on the v2 pool while its `run_manifest.json` claimed the v3 one. | **Fixed 2026-09-21.** `load_grpo_pool(subdir=None)` mirrors `load_processed_dataset`'s long-standing pattern; `grpo_trainer.py` passes `cfg.get("grpo_pool_subdir")`, and the fully resolved value is written to `run_manifest.json` as `resolved_grpo_pool_subdir`. Precedence: CLI → task YAML → `base.yaml` → literal default, so passing nothing reproduces the old path exactly |
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
| `2.0e-6` (v1) | 1.10e-4 | 1/234 | 1/497 | **measured: KL ≤ 0.001, effect not significant** (*p* = 0.08 / 0.16) |
| **`1.0e-5` (v2)** | **5.50e-4** | **1/47** | **1/99** | **measured: works.** KL 0.003–0.005, reward +0.06, F1 micro +0.07 to +0.09 at *p* < 0.001, every tier |

`lr_scheduler_type` also moved `cosine` → **`constant_with_warmup`**: there is no overfitting pressure to
anneal against over 2 epochs of a 1732-image pool with a KL anchor, so a flat LR after warmup keeps the whole
budget productive instead of spending its tail at ~0.

Still safe because the brakes are tight: `max_grad_norm: 0.3`, `beta: 0.04` (KL to the merged reference), and
`scale_rewards="group"` (normalises advantage so reward magnitude can't inflate step size). **One structural
fact:** with `num_iterations=1` and `steps_per_generation(4) ≤ gradient_accumulation_steps(16)`, TRL's own
PPO-style clipping is **inert** here — the importance ratio is identically 1.0, so the realized per-token loss
is plain `-A_i + β·KL_i` (group-relative-advantage REINFORCE with a KL anchor), not clipped-ratio PPO.

**What v2 actually measured, and what it means for the next change.** The old go/no-go rule here ("`kl` should
rise into 0.01–0.10; if flat, try 5e-6") is **superseded** — it would have fired a false alarm. Across all
three v2 GRPO runs:

| | 2b | 4b | 8b |
|---|---|---|---|
| `kl` mean / max | 0.0039 / 0.0064 | 0.0029 / 0.0047 | 0.0167 / 0.2078 |
| `reward` first → last (max) | 0.382 → 0.445 (0.490 @ 40) | 0.426 → 0.469 (0.499 @ 70) | 0.381 → 0.482 (0.523 @ 40) |
| `grad_norm` mean (logged steps **above** the 0.3 clip) | 0.408 (**17/21**) | 0.375 (**19/21**) | 0.423 (**14/21**) |
| `frac_reward_zero_std` mean | 0.405 | 0.455 | 0.466 |
| `reward_std` first → last | 0.110 → 0.081 | 0.116 → 0.053 | 0.134 → 0.056 |

Read that table in this order:

1. **KL stayed an order of magnitude below the predicted band and the run still worked.** KL is not the
   go/no-go signal on this task; `reward/mean` climbing and the downstream paired F1 test are.
2. **`beta` is not a brake at all.** At KL ≈ 0.004 and `beta: 0.04` the penalty contributes 1.6e-4 to the loss.
3. **`max_grad_norm: 0.3` is the only brake still doing anything, and it binds on 67–90% of steps.** It is set
   3.3× tighter than SFT's `max_grad_norm: 1.0` for no measured reason. **This — not the LR — is the next
   knob** (`README_v2.md` §13, P1-1). Raising the LR again while the clip binds mostly buys a flatter,
   more-normalised step, not a bigger one.
4. **The run stalls at roughly step 40–70 of 108.** `reward_std` halves while 41–47% of groups already have
   zero reward variance (and therefore zero gradient). The last third of every GRPO job buys almost nothing;
   the lever is `num_generations` or pool curation, not more steps.
5. **8B shows two transients nobody else does:** `kl = 0.2078` at the first logged step (then 0.0006), and
   `grad_norm = 2.34` at step 90 followed by `kl = 0.041` at step 95. Both recover; no lasting damage.
6. **Cosmetic NaN.** The 8B run logs `"kl": NaN` at exactly the two steps where `completions/clipped_ratio > 0`:
   with `mask_truncated_completions: true`, a fully-masked truncated sequence makes the logged per-token KL
   mean a 0/0. The loss stayed finite and training was unaffected — but if an entire generation batch were ever
   truncated, the loss itself would go NaN. A guard is on the v3 list (P1-7).

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
| Prompt | `prompt_key` in task YAML → `data/prompt_templates.py::PROMPT_REGISTRY`. **That key is load-bearing**: `get_prompt_for_task` reads `task_cfg["prompt_key"]` and looks the string up in the registry, so a typo is a `ValueError` at first use. (Earlier text here called it "descriptive text only" and claimed the registry was keyed by task *name* — wrong, corrected 2026-09-21.) |
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

| Artifact | unified | violations_only | violations_think | object_only | caption_only |
|---|---|---|---|---|---|
| SFT variant | `unified-sft-8b-v1` | `vo-sft-8b-v1` | `vt-sft-8b-v1` | `oo-sft-8b-v1` | `co-sft-8b-v1` |
| Merged KL base | `merged-unified-sft-8b-v1` | `merged-vo-sft-8b-v1` | `merged-vt-sft-8b-v1` | `merged-oo-sft-8b-v1` | `merged-co-sft-8b-v1` |
| GRPO variant | `unified-grpo-8b-v1` | `vo-grpo-8b-v1` | `vt-grpo-8b-v1` | `oo-grpo-8b-v1` | `co-grpo-8b-v1` |
| Baseline results dir | `unified-baseline-8b-v1` | `vo-baseline-8b-v1` | `vt-baseline-8b-v1` | `oo-baseline-8b-v1` | `co-baseline-8b-v1` |
| SLURM job / log stem | `vlm-sft-unified` / `sft_unified` | `vlm-sft-vo` / `sft_vo` | `vlm-sft-vt` / `sft_vt` | `vlm-sft-oo` / `sft_oo` | `vlm-sft-co` / `sft_co` |

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
**except `scripts/augment_data.sh`**, which has neither). `hpc_merge_sft.sh` refuses to run unless
`<sft_variant>/final/` contains an `adapter_config.json` or `adapter_model.safetensors`
(`scripts/hpc_merge_sft.sh:102-115` — **`final/`, not `best/`**; changed with the 2026-09-10 handoff switch,
and earlier text in this file saying `best/` was stale). `hpc_grpo.sh` refuses if the merged KL base is missing
(`:132-141`). Both guards confirmed to actually fire.

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
  5×. Raised to `0.85` for v1, then lowered to **`0.30`** for v2 in both task YAMLs;
  `scripts/validate_rewards.py --probe` passes at 0.30. **`0.30` is the current value — earlier text saying
  "now 0.85" was stale.**

  **What that constant actually sets is the model's operating point, and v2 measured it exactly.** The
  constant does not decide *whether* honest behaviour wins (the policy probe tests that separately, under the
  full weighted reward); it decides *where* the assert/abstain line sits:

  ```
  p* = c·(w_id + w_gnd + w_rsn) / [ c·(w_id + w_gnd + w_rsn) + w_id + w_gnd·E[IoU] + w_rsn·E[reason] ]
     = 0.30·0.95 / (0.30·0.95 + 0.422 + 0.317·0.45 + 0.211·0.50)  =  0.285 / 0.955  =  0.298
  ```

  (`ACHIEVABLE_TP_GROUNDING = 0.45`, `ACHIEVABLE_TP_REASONING = 0.50` in `scripts/validate_rewards.py`, both
  measured off real runs; the format weight cancels since both branches emit valid JSON.) v1 sat at `c = 0.85`
  → `p* = 0.546`, i.e. the model had to be more than half sure before speaking, which contradicts
  `violation_fbeta = 2.0`. The validator now asserts `p*` stays inside `VIOLATION_BREAKEVEN_BAND = (0.20, 0.50)`.

  **GRPO then converged to `p*` to within 0.07 at all three tiers** — marginal precision 0.370/0.370/0.388 on
  the flags it added, 0.077–0.250 on the flags it removed. That is the single most useful thing v2 established:
  the RL machinery is sound and the reward's *specification* is what sets behaviour. It is also why macro-F1 is
  flat — one global `p*` is ~2.5× too high for rules whose achievable precision is 0.15–0.19. **Changing
  `violation_tn_constant` alone cannot fix macro**, because a single constant cannot give four rules four
  thresholds; the per-rule options are in [`README_v2.md` §13](README_v2.md#13-v3-plan-costed-and-prioritised)
  (P1-4). Re-run `scripts/validate_rewards.py --probe` after touching any of this.
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

## Settled decisions — do not re-litigate

Each of these was argued, measured, and closed. Reopening one needs new evidence, not a fresh opinion. The
"why" is elsewhere in this file or in `README_v2.md`; this table exists so an agent does not spend a session
re-deriving a settled choice.

| Decision | Settled | Where the argument is |
|---|---|---|
| **Four independent task pipelines**, not one multi-task model | — | [Overview](#overview) |
| **One shared, task-blind GRPO pool.** Do not add a per-task pool | — | [Tasks](#tasks--what---task-actually-controls) |
| SFT learning rate **flat `1.0e-4` across tiers** (a per-tier LR is allowed only as declared config in `model_registry.yaml`) | — | [Why SFT's LR is flat](#config-layering) |
| **`final/` is the SFT handoff; early stopping OFF; `best/` is diagnostic only** | 2026-09-10 | invariant 7 |
| **`violation_tn_constant: 0.30`** (both violation tasks) and **`violation_fbeta: 2.0`** | 2026-09-10 | [Rewards](#rewards-and-the-output-contract) — includes the `p*` derivation |
| **GRPO LR `1.0e-5` + `constant_with_warmup`** | 2026-09-10 | [Config layering](#config-layering); v2 confirmed it works |
| **Per-tier LoRA `r/α = 16/20/32`, α tracking r** | 2026-09-10 | invariant 8; re-levels the adapted fraction to ~1.1% |
| **GRPO on `gpu:h100:1`, not H200**; `per_device_train_batch_size: 16` | 2026-09-06 / confirmed by v2 | [GPU policy](#gpu-policy) |
| **`repetition_penalty: 1.0` — the reward-side penalty is OFF** | 2026-09-05 | ghost-variable table |
| **`scale_rewards="group"`, `loss_type="dapo"`, `mask_truncated_completions=true`, `dataloader_drop_last=true`, `seed` threaded** — all pinned explicitly rather than inherited from TRL defaults | 2026-09-05 | ghost-variable table |
| **`RULE_MULTIPLIERS = {4: 16, 2: 12, 3: 6}` stays as-is**; `oversample_*_multiplier` stays at 1 | explicitly kept for v2 | [Data flow](#data-flow); pinned by `test_v2_augmentation_multipliers_unchanged` |
| **`reward_format` demoted to 0.05 for `violations_only`, not removed** (it saturates post-SFT but must stay visible) | 2026-09-10 | [Rewards](#rewards-and-the-output-contract) |
| **LLM judge is inline-only (`--use_llm_judge`), `batch_size: 1`, 3 few-shot per rule with the paper's 5 anchors first** | 2026-09-10 | [LLM-as-a-judge](#llm-as-a-judge-reasoning-evaluation) |
| **TP-conditioned macros skip unmeasured rules and publish `_macro_n_rules`; identification macros do NOT** | — | [Macro averaging](#invariants-and-known-traps) |
| **Per-rule CIDEr-D is never gated** (read it against `scored_count`); **CLIPScore stays out of the headline** but remains in the CSVs | — | this file + `README_v2.md` §11.7 |
| **W&B runs fully offline**; read curves from the SLURM `.out` or `wandb sync` afterwards | — | [Commands](#commands) |
| **The comparison toolset is terminal-first**: tables + CSVs + PNG charts, no web dashboard. Charts are on by default with `--no-charts` as the opt-out | — | [Local analysis](#local-analysis) |

**Working conventions with the maintainer** (these are process, not code):

- **Never commit or push unless explicitly asked.** Report what changed and let them run git.
- **Never edit the training or inference pipeline to make a tooling/analysis job easier.** Analysis reads the
  results tree; it does not reshape it.
- **Never run the full test suite on ARC** — see [the hazard](#tests).
- Prefer adding to the results toolset over writing another one-off plotting script; four of those were deleted
  in `07592de` for exactly that reason.

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

   **Mid-training checkpoints exist and are not rotated away.** `core/callbacks.py::PersistentCheckpointCallback`
   is wired into **both** trainers at `persistent_freq=100` (`models/sft_trainer.py:334`,
   `models/grpo_trainer.py:323`); it `copytree`s `checkpoint-N` to `persistent-checkpoint-N` whenever
   `N % 100 == 0`. With SFT's `save_steps: 50` over 512 steps that yields
   `persistent-checkpoint-{100,200,300,400,500}` per SFT variant, plus `best/` (2b step 275, 4b 375, **8b 125**)
   and `final/`. So the whole SFT budget sweep that would settle the 8B overfitting question
   ([`README_v2.md` §13](README_v2.md#13-v3-plan-costed-and-prioritised), P0-1) needs **no retraining at all** —
   only `run_inference --checkpoint <name>` + repair + eval against checkpoints already on disk. Confirm they
   are there before planning around them; nothing has verified this on the v2 tree yet.

   **One trap if you do that sweep.** `run_inference.py` names its results directory `<variant>_<checkpoint>`,
   so evaluating `persistent-checkpoint-300` writes `vo-sft-4b-v2_persistent-checkpoint-300` — and
   `results_lib.py::_PHASE_SUFFIXES` only accepts `{"final", "best"}` for `sft`, so
   `build_results_index.py` treats that directory as **not ours** and silently drops it into `skipped_names`.
   The naming scheme has no slot for a checkpoint id, and `--run_name` cannot invent one without breaking the
   `v<digits>` version parse. Either read those runs' `metrics.json` directly (fine for a 3–5 point sweep,
   which only needs `violation_identification_f1_{micro,macro}`), or extend `_PHASE_SUFFIXES` to accept
   `checkpoint-N` / `persistent-checkpoint-N` for `sft` first. Do not discover this after burning five
   inference jobs.

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

**Every metric in `metrics.json` is measured on the REPAIRED file, including the structural family.** The chain
is fixed (inference → repair → evaluation) and `run_evaluation.py` is pointed at
`repair_applied/predictions_repaired.jsonl`, so `structural_json_validity_rate` and
`structural_schema_adherence_rate` describe *post-repair* output, not what the model emitted. For
`vo-baseline-2b-v2` that is the difference between a reported **0.981** and a real raw compliance of
**20.11%**: 2,400 of 3,004 raw outputs fail to parse (degenerate repetition loops that hit the 1,024-token
cap), and the repair rescues 78.0% of records, manufacturing 378 TPs and 3,519 FPs and moving F1 micro from
0.0155 to 0.1529. For the other eight v2 runs the repair moves F1 by ≤ 0.008, and at the 8B baseline it
*hurts* (−0.0078).

Consequences, all of them load-bearing when writing anything up:

- **The honest raw number is `repair_stats.csv::status:valid_raw:pct`** (equivalently
  `repair_applied/repair_report.json`). Raw validity 20.11% → 99.93% is the single biggest thing SFT does, and
  the post-repair metric hides it. Emitting `structural_*_raw` keys from the pre-repair file is v3 P0-3.
- **Report `vo-baseline-2b` as "20.1% valid raw; 0.153 after repair"**, never as a detection result.
- A run's raw-vs-repaired gap is recoverable at any time by scoring `predictions.jsonl` and
  `repair_applied/predictions_repaired.jsonl` with the same parser — both files are kept.
- Measured across all nine v2 runs: **zero contentless assertions** (`{"reason":"","bounding_box":[]}`), so the
  presence-vs-substance asymmetry below has no effect on any current number.

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

*Dead code (present on disk, nothing calls it):*

- `setup_project_structure.py` — dead one-time bootstrap. On disk, git-ignored, **no longer tracked**. Don't run it.
- `rewards/{json_validity,caption_quality,rule_violation_accuracy,grounding_iou}.py` — legacy, unwired.
- `rewards/reward_utils.py::_strict_parse`/`_strict_parse_cached` and
  `evaluation/output_parser.py::validate_unified_output` — pre-task legacy path hardcoded to `UnifiedOutput`.
  Use the `_for_task` versions.
- `data/preprocessor.py`'s task-blind `raw_sample_to_conversation`, `build_unified_sft_dataset`,
  `to_grpo_prompt`, `build_grpo_dataset` — superseded by the `_for_task` versions the pipeline actually uses.
- `data/dataset_cache.py` — unreferenced dead code.
- `experiments/run_dual_evaluation.py` — returns a nested metrics shape nothing reads any more; semi-stale.
- `experiments/plot_8b_v2_comparison.py` — a one-off seaborn script that reads a git-ignored
  `evaluation_results_v2/` directory. Superseded by `compare_all.py` + `results_charts.py`. Note its "v2" means
  an old archive, **not** the current `--version v2`.
- `evaluation/report_generator.py`, `evaluation/error_analyzer.py` — not called from any live pipeline path.
- `scripts/{test_sampling,test_processor_batch_collapse}.py`, `scripts/push_adapter_to_hub.py`,
  `scripts/setup_drive_structure.py` — one-off utilities, not part of any pipeline.

*Stale documents (tracked, but superseded — do not quote numbers from them):*

- `baseline_vs_sft_report.md` — a `unified`/2B baseline→SFT report from before v1. Different task and tier from
  anything current.
- `evaluation_results_archive/`, `evaluation_results_archive_v2/` — pre-v1 metrics + plots. The `_v2` in that
  folder name is **not** `--version v2`; it is an old archive generation. `README.md` used to embed a plot from
  it; that section now points at `README_v2.md`.
- `ab.md` — the one-off prompt used to commission the `object_only`/`caption_only` pipelines. Historical.
- `docs/Metrics.md` — **no longer exists**; earlier text here referenced it. Nothing in `docs/` is tracked.

*Void measurements:*

- **All GRPO metrics from before this repo's `b8f2470` are void** — those runs trained prompt-only, images
  never reached the model. Baseline and SFT numbers from that era are usable; draw no GRPO conclusion from them.
- **All v1 `violations_only` numbers are superseded by v2** and several are artifacts rather than model
  behaviour: the v1 2B baseline was wrecked by the repair list-drop bug (recall 0.469 vs a real 0.890), every
  v1 SFT stopped early and handed off `best/`, and v1's TP-conditioned macros divided by 4 instead of by the
  measured rules. `README_v2.md` §9 attributes each v2 fix to its measured effect. Keep v1 only as the
  before-picture.
