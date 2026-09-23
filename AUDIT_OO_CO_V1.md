# `object_only` + `caption_only`, v1, 2b/4b/8b — audit result and runbook

**Temporary.** Delete once the oo/co runs have landed and their outcome is recorded in `README_v2.md`.

**Audited 2026-09-23** at `a44603a`, clean tree. Full adversarial pass over config resolution,
ghost variables, data/artifact flow, naming and save/load, rewards, evaluation, prompts/targets/repair,
and runtime failure modes. This file keeps only what is **still live** — findings that remain open, the
numbers that took real work to establish, and how to read the results. Everything verified-correct is
compressed into §6 so nobody re-audits it.

## ✅ VERDICT: GO

No blocker in the pipeline code. Configs resolve correctly and every key traced is genuinely read;
naming and save/load round-trip exactly; nothing collides; rewards are honest-optimal for both tasks;
evaluation gates correctly; prompt ≡ target ≡ schema ≡ parser with ≥2× token headroom.
Test suite: **1008 passed, 1 skipped**, local only (never on ARC — one test shells out to a real `sbatch`).

Residual risk is **interpretive, not operational**: three findings (F1, F2, D1) can make a number mean
something other than what it looks like. All three are handled by §4.

---

## 1. Pre-flight — RUN AND PASSED on ARC, 2026-09-23

| Check | Result |
|---|---|
| No surviving `oo-*`/`co-*`/`merged-{oo,co}-*` artifacts from the cancelled 2026-09-14 `co` chain | ✅ zero output, **with a positive control** (`ls .../vo-*` listed 10+ dirs). `$VLM_DATA_ROOT` was **empty** on the first attempt — without the positive control this would have been a false all-clear |
| Java at `$HOME/scratch/jdk-21.0.2/bin/java` (`caption_only` only) | ✅ present |
| `roberta-large` + `clip-vit-base-patch32` in `$HOME/scratch/hf_cache/hub` | ✅ both |
| `all-MiniLM-L6-v2` resolves under `HF_HUB_OFFLINE=1` | ✅ (needs the venv active; it fails with `ImportError` on the bare login shell — not a cache problem) |
| **Token census, `object_only`** — never run before | ✅ **PASS**, worst case **1846 < 3072** (prompt 280 + vision 1176 + target max 670, over 7009 train+val rows) |
| **Token census, `caption_only`** — never run before | ✅ **PASS**, worst case **1477 < 3072** (prompt 148 + vision 1176 + target max 301) |
| Qwen3-VL 2B/4B/8B cached | ✅ all three ⇒ `--skip-preload` is correct |
| Repo at `a44603a`, clean | ✅ |

Disk: home quota 130.6 GB available vs ~90 GB of new data (62 GB of it the six 16-bit merged
checkpoints). Fits; accepted.

---

## 2. Commands

```bash
# 0. LOCAL: commit + push the log-naming change (§7) first, or ARC runs with the old log names.
# 1. ARC login node:
cd $HOME/vlm-safety-reasoning
git pull                                               # must include the naming commit
module load gcc/13.3.0 python/3.12.5
source $HOME/envs/vlm_grpo/bin/activate
export VLM_DATA_ROOT="$HOME/vlm-finetuning-project1"    # NOT set by the login shell

python scripts/submit_pipeline.py --task object_only  --tiers 2b 4b 8b --version v1 --skip-preload
python scripts/submit_pipeline.py --task caption_only --tiers 2b 4b 8b --version v1 --skip-preload
```

`--skip-preload` is right **because pre-flight confirmed the base models are cached** — it also
sidesteps F4. It is not a blanket recommendation; without a warm cache the preload exists to stop
concurrent jobs racing on HF cache locks.

Eyeball the `Scheduling ... for Qwen3-VL-<T>` header lines for a tier typo before walking away:
`--tiers` has no `choices=` (F9), so `8B` is accepted and only fails after the queue wait.

**24 jobs.** Per tier: `baseline` ‖ `sft` start immediately, `merge` waits `afterok:<sft>`,
`grpo` waits `afterok:<merge>`.

### Monitoring

```bash
squeue -u $USER -o "%.10i %.24j %.9P %.8T %.10M %.6D %R"   # %.24j — default NAME is 8 chars, truncates
tail -f $VLM_DATA_ROOT/logs/sft_oo_2b_v1_<jobid>.out       # <phase>_<prefix>_<tier>_<version>_<jobid>

# Swallowed reward exceptions. .out, NOT .err — core/logging.py's only sink is sys.stdout.
grep -c "Error in reward function\|Error in batch reward function" \
     $VLM_DATA_ROOT/logs/grpo_{oo,co}_*_v1_*.out           # must be 0 everywhere
```

W&B is **offline** in all four phase scripts, so nothing appears in the dashboard. 30 offline runs
(5 per task/tier: baseline eval + SFT train/eval + GRPO train/eval; merge produces none), all names
unique and all already carrying tier + version. `wandb sync offline-run-*` would also re-sync the v2
`vo` runs — filter by mtime.

---

## 3. Live findings

### 🔴 Still-open code risk (cleared for *this* submission, will recur)

**R1 — GRPO auto-resume is unconditional and un-suppressible.**
`models/grpo_trainer.py:261-266`; `experiments/run_grpo.py` has no `--no-resume`; `hpc_sft.sh` never
passes SFT's. GRPO's resume is not gated on `training_state.json` and, unlike SFT (`sft_trainer.py:179-184`),
has **no model-identity check**. Any surviving `checkpoint-N` is resumed — for GRPO, on top of a
*freshly rebuilt, different* merged base — logging "Resuming from checkpoint" and exiting 0.
**Cleared for this run** (pre-flight §1). **It returns the moment you re-run any version**: always
confirm the checkpoint dirs are gone first, with a positive control on `$VLM_DATA_ROOT`.
*Proper fix:* `--no-resume` on `run_grpo.py` + a `--fresh` flag on the submitter, and give
`grpo_trainer.py` SFT's `hf_path` identity check.

### 🟠 Should-fix — all deferred until after the runs; none justifies delaying

| # | Finding | file:line | Failure scenario |
|---|---|---|---|
| **F1** | A **mid-prose ``` fence truncates a `caption_only` output** to whatever sits between the backticks. Verified: `Two workers stand ```beside``` an excavator and more text after.` → caption becomes **`beside`**, status `fixed_valid`, logged only as `caption_fence_stripped` | `evaluation/output_parser.py:19` | A zero-shot `co-baseline-*` falls into markdown habits mid-paragraph; its captioning metrics collapse on those records with no visible cause. Same shape as the violation list-drop precedent — a repair that *replaces* the answer. *Fix:* accept the fenced content only when it is essentially the whole completion (`len(match) ≥ 0.7 × len(stripped)`). *Detector:* `repair_report.json::change_type_summary.caption_fence_stripped` |
| **F2** | A **`[0,1]`-scale object prediction scores identically to predicting nothing**, silently. Verified on an excavator-present image: correct 0-1000 → `reward_grounding 0.3703`; already-normalized → **0.0377**; all-empty → **0.0377**. Zero repair entries, `valid_raw` status | `data/box_utils.py:73` + `rewards/reward_grounding.py:22` | `oo-baseline-*` emits normalized coords → `grounding_presence_recall ≈ 0` → written up as "the baseline cannot detect objects" when it detected them perfectly in the wrong unit. No prior run to catch it. *Fix:* `grounding_suspected_normalized_scale_count` in `metrics_grounding.py`. *Or:* §4 item 1 |
| **F3** | **`structural_*` carries zero information for `caption_only`** — always ≈1.0. Verified: clean prose, fenced output and `{"caption": …}` **all** give validity = adherence = 1.0. Compounded twice: eval reads the *repaired* file, and `is_clean_prose` (the real contract, what `reward_format` scores) is consulted by no metric | `evaluation/metrics_structural.py:36-41` | A `co-sft-8b-v1` emitting JSON on 100% of images reports perfect structural scores. *This is the `co` instance of P0-3.* *Fix:* emit `structural_clean_prose_rate` from the pre-repair file. *Until then:* quote `repair_stats.csv::valid_raw_%` |
| **F4** | **Preload fills the wrong cache** — `subprocess.run(["hf","download",...])` with no `env=`, inheriting a login shell that has no `HF_HOME`; every job reads `~/scratch/hf_cache` | `scripts/submit_pipeline.py:154` | ~35 GB into home quota **and** the cache-lock race it exists to prevent still happens. *Sidestepped this run by `--skip-preload`.* *Fix:* pass `env={**os.environ, "HF_HOME": ...}` |
| **F5** | **Reward failures are near-undiagnosable.** Both handlers log `Error in reward function compute_reward:` — `functools.wraps` captured the *inner* name, so no line names the component | `rewards/reward_utils.py:293,308` | `all-MiniLM-L6-v2` unresolvable → `_safe_batch_reward` returns `[0.0]*N` → **90% of `caption_only`'s objective silently zero for 108 steps**; `reward/mean` flatlines near 0.10, job exits 0. *Doc half fixed* (`CLAUDE.md`/`OPERATIONS.md` now say `.out`, both spellings). *Code fix outstanding:* pass the public name into the decorator |
| **F6** | **`worker_with_hard_hat` → `worker_with_white_hard_hat` repair alias is a semantic re-label**, logged only as `key_renamed`. The prompt is explicit that a yellow/red/blue/orange hat does not count | `preprocessing/structural_repair.py:529` | A model that answered the *broader* question has its answer re-labelled as the narrow one, manufacturing FPs for a rare class. Most likely on `oo-baseline-*`. *Quantify first:* count `key_renamed` on that field in the baseline `change_manifest.json` |
| **F7** | **No bootstrap CI or paired significance test exists for either task** — everything is keyed on `violation_per_image_outcomes_b64` | `experiments/results_lib.py:631` | Every oo/co number in the first report is a bare point estimate. Given `vo`'s history (SFT→GRPO looked like +0.018 *p*=0.08 in v1, +0.091 *p*<0.001 in v2) this is exactly what the CI machinery was built for. *Fix:* emit per-image outcome vectors for grounding and captioning, then generalise `BOOTSTRAP_METRICS` |
| **F8** | **Win tally counts two lower-is-better keys as wins.** Verified `is_non_comparable_key` → `False` for `grounding_mask_iou_all_macro_*_tn1`, `captioning_blank_prediction_rate`, `captioning_long_clipscore_avg_chunks_per_caption` | `experiments/results_lib.py:537-553` | A phase that predicts *less* wins on `_tn1`; a `co` phase with *more* blank captions scores a win in `delta_co_v1.csv`'s SUMMARY |
| **F9** | `--tiers` accepts any string (no `choices=`) | `scripts/submit_pipeline.py:173` | `--tiers 8B` submits 4 jobs that die on the compute node after the queue wait |
| **F10** | **Repair injects a missing `object_only` class key and logs no change**; `:838-840` drops a dict-valued class with no log entry at all | `preprocessing/structural_repair.py:1190-1192` | `structural_schema_adherence_rate` reads 0.0 raw and 1.0 post-repair, so "the model omits absent classes" becomes undetectable — the exact behaviour `ObjectOnlyOutput`'s required fields exist to surface. Training signal (100% reward loss) and reported metric disagree completely. Folds into P0-3 |
| **F11** | A run that dies after inference leaves a **stale `metrics.json`** the index reads as current | `scripts/hpc_*.sh` steps 2→3 | Re-running truncates `predictions.jsonl` but `metrics.json` is written last; an eval crash (OOM during BERTScore/CLIP over 3004 `co` captions) leaves old metrics beside new predictions. *Fix:* `rm -f "$EVAL_OUT_DIR/metrics.json"` before eval, or compare mtimes |

### ⚪ Cosmetic (one line each)

`load_model_for_inference` calls `load_config(training_kind="sft")` **without `task=`** (`model_loader.py:223`) — a task-level `image_*_pixels` would be ignored at inference; inert today ·
`grounding_tn_constant`'s `default: 0.15` would silently apply the old exploit-inducing flat value to any class later added to `GROUNDING_CLASSES` ·
`DEFAULT_POOL_PREVALENCE` (0.361/0.088/0.115, `validate_rewards.py:60-64`) and `object_only.yaml:29-30` are stale vs the measured 0.3545/0.0837/0.1189 (break-evens shift ≤0.028; nothing changes) ·
`grpo.yaml`'s LR comment still carries the superseded v1 sizing argument and a "kl should rise into 0.01-0.10" instruction v2 disproved ·
the merge stage writes no manifest, and inference `run_manifest.json` carries no git provenance ·
`reward_format` does not actually require the fence — unfenced bare JSON scores 1.0 (pre-existing, shared with `vo`, consistent with evaluation) ·
`is_clean_prose` misses an **unquoted** `caption:` label (verified: format 1.000, label scored as part of the caption) and false-positives on a prose caption starting with `[` ·
24 jobs × `--mail-type=BEGIN,END,FAIL` ≈ 72 emails ·
every oo/co eval logs a false *"SPICE will download ~2GB…"* warning ·
`compare_all.py:196`'s "(evaluated before that key existed)" is wrong — these tasks structurally cannot have that key ·
`eval_manifest.json` records `"use_llm_judge": true` while no `llm_judge_status.json` exists ·
the 18 eval W&B runs use a bare `wandb.init` with no group/entity, so on sync they may land in the default entity (pre-existing; v2 did the same) ·
that same `wandb.init` is wrapped in `except ImportError` only, so a non-ImportError failure kills the job *after* `metrics.json` is written (survived 9/9 in v2).

---

## 4. How to read the results — the three traps

1. **`oo-baseline-2b-v1`, before reading any number:**
   `head -3 $VLM_DATA_ROOT/results/inference/oo-baseline-2b-v1/predictions.jsonl`
   Coordinates must be in the **hundreds**, not fractions. Fractions ⇒ F2, and every baseline
   grounding number is meaningless rather than bad.
2. **For every `oo` run, cross-check `repair_stats.csv::status:valid_raw:pct` first.**
   `vo-baseline-2b-v2` had 20.11% raw validity while reporting `structural_json_validity_rate = 0.981`.
   A failed parse is credited as a *confident correct abstention* by the `*_tn1` grounding family
   (verified: 4/4 garbage → `mask_iou_all_macro_mean_tn1 = 0.667`). The two headline keys —
   `grounding_mask_iou_all_macro_mean_tn0` and `grounding_presence_f1_macro` — are honest (both 0.0 in
   the same test). **Never quote the `_tn1` family.**
3. **On `co` GRPO, watch caption diversity, not just `reward_caption/mean`.** A single
   image-independent centroid caption scores **0.526** vs honest-paraphrase **0.674** — 78% of honest
   with zero use of vision — and beats a correct-but-wrong-image caption (0.376). Rising mean reward +
   falling distinct-caption count = mode collapse; the fix is a diversity term in a future arm, not a
   weight change.

Also: **`caption_only` — quote `valid_raw_%`, not `structural_json_validity_rate`** (F3), and read
`captioning_scored_count` + `captioning_blank_prediction_rate` (in `master_wide.csv`, **not** the
terminal table) alongside `bertscore_f1`, or a high blank rate hides behind a good conditional score.
**Both tasks: no CIs, no significance tests** (F7) — treat every delta as a point estimate.

---

## 5. Measured facts worth keeping

These took real work to establish and exist in no other document.

### 5.1 What actually runs

| | `object_only` | `caption_only` |
|---|---|---|
| SFT input | `datasets/processed` (**6308** train / 701 val, un-augmented) | same |
| **SFT steps** | `6308 // 32 × 2 =` **394** (not `vo`'s 512) | same |
| Persistent checkpoints | **`{100,200,300}` only** — not `{100..500}` | same |
| Rare-sampler axis | rebar ∨ white-hard-hat; **1291/6308 = 20.47%** marked rare | **None** → plain shuffle |
| Oversample manifest | **not written** (gated on `CAP_VIOLATIONS`) | same |
| **GRPO steps** | `1732 // 32 × 2 =` **108** (shared pool) | same |
| Reward components | `reward_format` .10, `reward_grounding` .90 | `reward_format` .10, `reward_caption` .90 |
| Token budget (prompt / vision / target / cap) | 297 / 1176 / **390** / 3072 → **1863**, 1209 margin | 164 / 1176 / **195** / 3072 → **1535**, 1537 margin |
| `max_new_tokens`, `max_completion_length` | 768 (2.0× headroom) | 768 (3.9× headroom) |
| `inference_max_seq_length` → prompt cap | 2688 → 1920 | 2944 → 2176 |
| Eval needs JVM / images / network | **no / no / none at all** | **yes (hard fail) / yes / `roberta-large` + CLIP** |
| Metric keys emitted | **79** (5 structural + 74 grounding) | **23** (5 structural + 18 captioning) |
| LLM judge | **never imported** — double-gated at `evaluator.py:152,167` | same |

**LoRA is per-tier** (`lora_by_tier`, resolved in `model_loader.py` not `core/config.py`): r/α =
16/20/32, α tracking r, ⇒ ~1.10/1.10/1.16% trained. **Note these v1 runs use the v2 adapter-capacity
decision** — `lora_by_tier` is not versioned — so they are not comparable to `unified` v1 (flat r=16)
on that axis.

**SFT LR 1.0e-4 / cosine / warmup 0.10 / max_grad_norm 1.0; GRPO LR 1.0e-5 / constant_with_warmup /
warmup 0.05 / max_grad_norm 0.3 / beta 0.04 / num_generations 8 / steps_per_generation 4 /
grad_accum 16 / loss_type dapo / scale_rewards group / mask_truncated_completions true /
dataloader_drop_last true / seed 42 / temperature 0.9 / top_p 0.95** (the last two are **hardcoded** at
`grpo_trainer.py:307-308`, not config). The 1.2 MP image cap **does** apply to GRPO — it arrives via
`sft_cfg`, and `grpo.yaml`'s own override block correctly does not fire.

### 5.2 Dataset ground truth (measured on `data/processed`; matches the paper's Table 4 exactly)

| | rows | excavator | rebar | hard-hat | **no target object** |
|---|---|---|---|---|---|
| train (SFT) | 6308 | 0.3453 | 0.1214 | 0.0980 | 0.4959 |
| **GRPO pool** | 1732 | **0.3545** | **0.0837** | **0.1189** | **0.5087** |
| test | 3004 | 0.3595 | 0.1089 | 0.1045 | 0.4837 |

### 5.3 The three data realities that shape interpretation

**D1 — test captions come from a different length distribution than train.** Train median 54 words,
10th pct 36, **0.5% under 20 words**. Test median 44, 10th pct **14**, **17.2% under 20 words, 6.0%
under 10**. By `quality_of_info`: train "poor info" median 51 vs test "poor info" median **31**. SFT and
GRPO both train on train/val-derived data, so the model learns ~55-word paragraphs and meets a test set
where 1 in 6 references is under 20 words. `reward_caption`'s length brake shows the magnitude:
σ = max(0.6·len_gt, 5), so a 55-word prediction against a 5-word reference gets
`exp(−0.5·10²) ≈ 2e-22`. **Not a bug — a published property of the dataset.** Baseline→SFT→GRPO
comparisons stay valid; absolute scores are structurally depressed on that subset.
**Report captioning metrics stratified by reference length or `quality_of_info`** (already a column in
every prediction record's `sample` dict, so it is free post hoc).

**D2 — rebar will barely move under GRPO, and that is prevalence, not a tuning error.** Achievable
post-SFT IoU, from the archived `unified` runs in `evaluation_results_archive/` (the only measurement
of this anywhere — CLAUDE.md says baseline/SFT numbers from that era are usable, only GRPO is void):
excavator **0.829/0.835**, rebar **0.349/0.320**, hard-hat **0.550/0.641** (2b/8b,
`grounding_mask_iou_exist_macro_<cls>`, which is exactly `E[reward term | GT present]`).

Two framings of the break-even, and the second is the one that governs:

| class | c_k | break-even IoU `c(1−p)/p` | achievable | marginal threshold `c/(c+q)` | prior |
|---|---|---|---|---|---|
| excavator | 0.283 | 0.515 | 0.829 ✓ | **0.254** | 0.3545 |
| rebar | 0.048 | 0.526 | 0.349 ✗ | **0.121** | 0.0837 |
| hard-hat | 0.065 | 0.482 | 0.550 ✓ | **0.106** | 0.1189 |

The first column looks alarming for rebar and **is not the operative one**: a policy that emits rebar
whenever it is >12% confident is strictly better off than one that never emits, and honest beats
always-empty structurally (per class the gap is `p·q > 0`). Measured totals — honest 0.235/0.260/0.286
at E[IoU] 0.30/0.45/0.60 vs always-empty 0.185 everywhere. What the table *does* say: marginal reward
contribution is excavator **0.097**, hard-hat **0.021**, rebar **0.009**. **Excavator is worth ~10×
rebar to the objective.**

**D3 — `frac_reward_zero_std ≈ 0.5` for `object_only` is the floor, not a symptom.** 881/1732 pool
images (**50.87%**) contain none of the three classes; there honest ≡ always-empty ≡ common-class-only
are *the same string*, all 8 rollouts score 0.2188, std = 0, zero gradient. **Do not try to fix this
with `grounding_tn_constant`** — the lever is pool composition, and that would break the "one shared,
task-blind pool" decision. `caption_only` has no such floor.

---

## 6. Verified clean — do not re-audit

Config resolution (`base → model_registry → {sft,grpo} → tasks/<task>`), and **no new ghost variable
on the oo/co paths**: `sft_dataset_subdir`, `max_completion_length`, `max_new_tokens`,
`inference_max_seq_length`, `grounding_tn_constant`, `repetition_penalty` (key name byte-matches the
YAML; short-circuits at 1.0; the historical `repetition_penalty_factor` has no live read site),
`grpo_pool_subdir`, `lora_by_tier`, `seed`, `dataloader_drop_last`, `steps_per_generation` — all
traced to their consuming line ·
the `cfg → sft_cfg` copy-over in `grpo_trainer.py:89-132`, audited key by key ·
**name isolation**: 108 writable paths across 5 tasks × 3 tiers × 2 versions, **zero duplicates**;
nothing collides with `vo` v1/v2 or `unified` v1 ·
the **merged-checkpoint round trip** executed end to end (`merged-oo-sft-8b-v1`, `merged-co-sft-2b-v1`,
MATCH True) — and the reverse-engineering regex is dead anyway, since `hpc_grpo.sh:170` passes
`--base_model_override` explicitly ·
`final/` vs `best/`: merge guard checks `final/`, both inferences pass `--checkpoint final`,
`final/` is written in a `finally:` block and a failed snapshot now raises rather than exiting 0 ·
`parse_run_name` round-trips all 18 oo/co results-dir names ·
GRPO's KL reference (invariant 2), the `image` column singular (invariant 1), the mandatory pool
(invariant 3), `tokenizer_name` on merged loads (invariant 4), explicit `steps_per_generation`
(invariant 5) ·
**the GRPO pool carries what both tasks need** — no column projection in `build_grpo_pool.py`,
confirmed against the live ARC `dataset_report.json`: 1732 rows, object boxes **and** 1732 non-blank
captions ·
prompt ≡ target ≡ schema ≡ parser on every axis (fence, key names, **key order**, minimization, box
scale, ints, `[]`-for-absent), and the prompt is **hashed byte-identical** across SFT / GRPO /
inference for both tasks ·
box scaling applied to predictions **only**, once, in both the reward and the metric; of 9,060 real GT
boxes `clean_boxes` drops 3 (0.033%) ·
capability gating in the evaluator, structural repair and reward assembly ·
reward func/weight lists provably index-aligned; weights sum to 1.0; honest beats every degenerate
policy tested (all-empty, common-class-only, hallucinate-all, box flooding, full-image boxes, keyword
padding, keyword repetition, prompt echo, cross-image swap, scale inversion, key typo) ·
SLURM resources: `MEM_CONFIG`/`TIME_CONFIG` byte-identical to each script's own `#SBATCH`; 24:00:00 is
at, not over, `gpu-h100`'s real `MaxTime`; no `--gres` passed so each script's `gpu:h100:1` governs ·
`set -eo pipefail` + guarded `cd` in all four phase scripts; every guard task-blind and correct ·
no executable line in any `.sh` names a task, prefix, dataset path or row count ·
W&B: offline everywhere, 30 runs, **all names unique**, all already carrying tier + version.

**Walltime and memory should both be *easier* than `vo` v2**, which completed all three tiers on a
single H100 with zero OOM: SFT is 394 steps not 512, completions are 768 not 1024, and SFT
materialises 6308 rows not 8198. `co` is the slower of the two (prose ~85 tokens/completion vs `oo`'s
~35); 8b `co` GRPO estimated ~15–16 h against the 24 h wall, and a wall kill auto-resumes from
`save_steps: 20`.

---

## 7. Fixed by this audit

- **SLURM job names and log stems now carry tier and version** — `core/naming.py:92-131`.
  `vlm-sft-oo-2b-v1` / `sft_oo_2b_v1_<jobid>.out`, replacing 8 distinct stems for 24 jobs. All four
  args **required** (no silent default, matching the `--task`/`--tier` rule everywhere else); tier and
  version **appended**, so every existing glob still matches (`logs/grpo_vo_*.err` →
  `grpo_vo_8b_v2_*.err`). `test_name_isolation.py` grew 25 → 231 tests and no longer exempts job names
  and log stems from the tier/version-uniqueness check.
- **Stale docs corrected:** `CLAUDE.md` and `OPERATIONS.md` said to grep **stderr** for reward
  exceptions — `core/logging.py:17-22` has one sink, on stdout. Both now say `.out`, list both message
  spellings, and explain why every line reads `compute_reward`. Also corrected in `CLAUDE.md`: the
  token-budget table (`oo` ≈265 → **297**, `co` ≈215 → **164**), the log-filename pattern, and the
  naming table.

Other stale claims found and **not** yet corrected in their source docs: pool prevalence
0.361/0.088/0.115 in `object_only.yaml:29-30` and `validate_rewards.py:60-64`; "798 tests" in
`CLAUDE.md`; persistent checkpoints `{100..500}` (true for `vo`, not for oo/co); the unconditional
"Requires a JRE on PATH" in `run_evaluation.py:7-10`.

---

## 8. After the runs land

F1, F3, F5 (code half), F6–F11, plus two `validate_rewards.py` gaps: the object break-even check has a
**ceiling but no floor** (`BREAKEVEN_CEILING = 0.75`, one-sided — the violation path has a two-sided
band, so dropping `grounding_tn_constant` to 0.01 would make shotgunning boxes positive-EV and the
validator would still pass green), and **`caption_only`'s probe cannot detect the centroid hack**
because its "honest" policy is literally the reference caption. F3 and F10 both fold into
`README_v2.md` §13 **P0-3** — now more valuable than it looked, since for `caption_only` the
post-repair structural keys carry no information at all.

---

# Appendix — every value that reaches runtime

Reference, not narrative. "Source" is the file the winning value comes from after the
`base → model_registry → {sft,grpo} → tasks/<task>` merge; "Read at" is the line that consumes it.
Produced by executing `core.config.load_config(task=..., training_kind=...)` for real, then tracing
each key. **Identical for `object_only` and `caption_only` except where marked.**

## A.1 SFT

| Key | Value | Source | Read at |
|---|---|---|---|
| `learning_rate` | **1.0e-4**, all tiers | `sft.yaml:7` | `sft_trainer.py:207` |
| `lr_scheduler_type` | `cosine` | `sft.yaml:31` | `sft_trainer.py:211` |
| `warmup_ratio` | 0.10 | `sft.yaml:8` | `sft_trainer.py:208` |
| `weight_decay` | 0.01 | `sft.yaml:9` | `sft_trainer.py:209` |
| `optim` | `adamw_8bit` | `sft.yaml:30` | `sft_trainer.py:210` |
| `max_grad_norm` | **1.0** | `sft.yaml:32` | `sft_trainer.py:212` |
| `num_train_epochs` | 2 | `sft.yaml:2` | `sft_trainer.py:206` |
| `per_device_train_batch_size` | **32** | `model_registry.yaml:43/51/59` | `sft_trainer.py:203` via `get_batch_config` |
| `gradient_accumulation_steps` | **1** | `model_registry.yaml:46/54/62` | `sft_trainer.py:205` |
| **effective batch** | **32** | derived | — |
| `per_device_eval_batch_size` | 32 | `model_registry.yaml:44/52/60` | `sft_trainer.py:204` |
| `eval_accumulation_steps` | 1 | `model_registry.yaml:45/53/61` | `sft_trainer.py:216` |
| `dataloader_drop_last` | **true** | `sft.yaml:113` | `sft_trainer.py:227` |
| **computed steps** | `6308 // 32 × 2 =` **394** | derived | — |
| `logging_steps` | 1 | `sft.yaml:35` | `sft_trainer.py:213` |
| `save_steps` | 50 | `sft.yaml:36` | `sft_trainer.py:214` |
| `eval_steps` / `eval_strategy` | 25 / `steps` ⇒ **15 evals** | `sft.yaml:37-38` | `sft_trainer.py:215,217` |
| `save_total_limit` | 3 | `sft.yaml:39` | `sft_trainer.py:218` |
| `load_best_model_at_end` | **false** | `sft.yaml:52` | `sft_trainer.py:219` |
| `metric_for_best_model` / `greater_is_better` | `eval_loss` / false | `sft.yaml:53,57` | `sft_trainer.py:220-221` |
| `best_model_threshold` | 0.0 | `sft.yaml:58` | `sft_trainer.py:339` |
| `early_stopping_patience` | **null → OFF** | `sft.yaml:74` | `sft_trainer.py:345` (`if patience:`) |
| persistent-checkpoint freq | **100** (hardcoded) ⇒ `{100,200,300}` | — | `sft_trainer.py:334` |
| `max_seq_length` → `SFTConfig.max_length` | **3072** | `sft.yaml:10` | `sft_trainer.py:242` **and** `model_loader.py:100` (load window) |
| `bf16` | true | `sft.yaml:93` | `sft_trainer.py:222` |
| `seed` | 42 | `base.yaml:6` | `sft_trainer.py:223` (`base_cfg`, not merged) |
| `auto_find_batch_size` | false | `sft.yaml:105` | `sft_trainer.py:226` |
| `load_in_4bit` | true | `sft.yaml:28` | `model_loader.py:99` |
| `use_gradient_checkpointing` | `unsloth` | `sft.yaml:90` | `model_loader.py:101` |
| `image_min_pixels` / `image_max_pixels` | **200704 / 1204224** | `sft.yaml:106-107` | `model_loader.py:184-185` → `apply_pixel_bounds` |
| `finetune_vision_layers` | **false** | `sft.yaml:86` | `model_loader.py:104` |
| `finetune_{language,attention,mlp}_*` | true / true / true | `sft.yaml:87-89` | `model_loader.py:107,110,113` |
| `use_stratified_rare_sampling` | true | `sft.yaml:97` | `sft_trainer.py:355` |
| `use_resolution_bucketing` | false | `sft.yaml:96` | `sft_trainer.py:356` |
| `bucket_shuffle_seed` | 42 | `sft.yaml:98` | `sft_trainer.py:61` |
| `oversample_rule{24,3}_multiplier` | 1 / 1 | `sft.yaml:101-102` | **never reached** — `run_sft.py:141` gates on `CAP_VIOLATIONS` |
| `sft_dataset_subdir` | **`datasets/processed`** | `tasks/{object_only:54,caption_only:27}` | `run_sft.py:89` → `:109` |

**LoRA, per tier** — `model_registry.yaml:33-36 lora_by_tier`, resolved in `model_loader.py:271-304`
(*not* `core/config.py`: `merge_configs` descends only one level, so anything under `models.<tier>`
would be inert). `dropout`/`target_modules` come from `sft.yaml:79-83`.

| tier | r | alpha | alpha/r | dropout | target_modules | trained | base | fraction |
|---|---|---|---|---|---|---|---|---|
| 2b | 16 | 16 | 1.0 | 0.05 | `all-linear` | 23.7 M | 2.15 B | **1.10%** |
| 4b | 20 | 20 | 1.0 | 0.05 | `all-linear` | ~49.1 M | 4.48 B | **~1.10%** |
| 8b | 32 | 32 | 1.0 | 0.05 | `all-linear` | ~102.6 M | 8.82 B | **~1.16%** |

## A.2 GRPO

| Key | Value | Source | Read at |
|---|---|---|---|
| `learning_rate` | **1.0e-5** | `grpo.yaml:18` | `grpo_trainer.py:288` |
| `lr_scheduler_type` | **`constant_with_warmup`** | `grpo.yaml:102` | `grpo_trainer.py:304` |
| `warmup_ratio` / `weight_decay` | 0.05 / 0.01 | `grpo.yaml:100-101` | `grpo_trainer.py:302-303` |
| `optim` | `adamw_8bit` | `grpo.yaml:99` | `grpo_trainer.py:311` |
| `max_grad_norm` | **0.3** | `grpo.yaml:109` | `grpo_trainer.py:305` |
| `beta` (KL) | 0.04 | `grpo.yaml:97` | `grpo_trainer.py:306` |
| `num_generations` | **8** | `grpo.yaml:5` | `grpo_trainer.py:285` |
| `per_device_train_batch_size` | **16** (top-level, all tiers) | `grpo.yaml:67` | `grpo_trainer.py:289` |
| `steps_per_generation` | **4** | `grpo.yaml:78` | `grpo_trainer.py:297` |
| `gradient_accumulation_steps` | **16** | `grpo.yaml:88` | `grpo_trainer.py:290` |
| `generation_batch_size` | 16×4 = **64**; `64 % 8 == 0` ✓ | derived | TRL divisibility check |
| K (unique imgs / `generate()`) | 64/8 = **8** | derived | verified NO COLLAPSE |
| unique imgs / optimizer update | 16×16/8 = **32** | derived | — |
| `num_train_epochs` | 2 | `grpo.yaml:92` | `grpo_trainer.py:298` |
| **computed steps** | `1732 // 32 × 2 =` **108** | derived | matches v2's 21 logged points |
| `logging_steps` | 5 | `grpo.yaml:93` | `grpo_trainer.py:299` |
| `save_steps` | 20 ⇒ ckpts at 20…100 | `grpo.yaml:94` | `grpo_trainer.py:300` |
| `save_total_limit` | 3 | `grpo.yaml:96` | `grpo_trainer.py:301` |
| persistent-checkpoint freq | 100 ⇒ `persistent-checkpoint-100` only | — | `grpo_trainer.py:347` |
| `loss_type` | **`dapo`** | `grpo.yaml:113` | `grpo_trainer.py:314` |
| `scale_rewards` | **`group`** | `grpo.yaml:148` | `grpo_trainer.py:317` |
| `mask_truncated_completions` | **true** | `grpo.yaml:125` | `grpo_trainer.py:315` |
| `dataloader_drop_last` | **true** | `grpo.yaml:137` | `grpo_trainer.py:316` |
| `seed` | **42** | `base.yaml:6` via merge | `grpo_trainer.py:323` |
| `bf16` | true | `grpo.yaml:98` | `grpo_trainer.py:310` |
| `temperature` | **0.9 — HARDCODED** | `grpo_trainer.py:307` | `GRPOConfig` |
| `top_p` | **0.95 — HARDCODED** | `grpo_trainer.py:308` | `GRPOConfig` |
| `top_k` | **50 — HARDCODED**, on `model.generation_config` only | `grpo_trainer.py:157` | not a `GRPOConfig` field |
| `generation_kwargs` | `{"do_sample": True}` | `grpo_trainer.py:309` | `GRPOConfig` |
| `log_completions` / `num_completions_to_print` | true / 4 | — | `grpo_trainer.py:326-327` |
| `max_prompt_length` | **2304** | `grpo.yaml:6` | `grpo_trainer.py:286` |
| `max_completion_length` | **768** | task YAML, **task-first** | `grpo_trainer.py:287` and `:159` |
| `max_seq_length` (load window) | **3600** | `grpo.yaml:14` | copied → `sft_cfg`, `grpo_trainer.py:89-90` |
| `load_in_4bit` / `use_gradient_checkpointing` | true / `unsloth` | `grpo.yaml:170-171` | copied at `:96-97`, `:93-94` |
| LoRA + `finetune_*` | `grpo.yaml:192-201` + `lora_by_tier` | — | copied at `:107-124` → `model_loader.py:120` |
| `image_{min,max}_pixels` | **inherited from `sft_cfg`**: 200704 / **1204224** | `sft.yaml:106-107` | `grpo.yaml` sets neither ⇒ the `if "image_max_pixels" in cfg` block at `:129-132` does **not** fire, so the SFT cap stands. **The 1.2 MP cap IS applied to GRPO.** |
| GRPO pool | `datasets/grpo_pool`, **1732 rows** | `base.yaml:14` via `resolve_grpo_pool_subdir` | `grpo_trainer.py:192,196` |

## A.3 Inference — baseline / post-SFT / post-GRPO, same code path

| Key | `object_only` | `caption_only` | Source | Read at |
|---|---|---|---|---|
| `max_new_tokens` | **768** | **768** | task YAML (task-first → sft/base → 1000) | `run_inference.py:139-144` → `inference.py:241` |
| `inference_max_seq_length` | **2688** | **2944** | task YAML | `model_loader.py:230` |
| derived prompt cap | 2688−768 = **1920** | 2944−768 = **2176** | derived | `run_inference.py:156-157` → `inference.py:226` |
| `repetition_penalty` | **1.0** (off) | **1.0** (off) | task YAML | `run_inference.py:145-149` → `inference.py:244` |
| `batch_size` | **32** | **32** | `hpc_*.sh` CLI | `run_inference.py:58` |
| `temperature` / `do_sample` | 0.0 / **False** (greedy) | same | `inference.py:172-174` defaults | `generate_batch` |
| `image_{min,max}_pixels` | 200704 / **1204224** | same | `sft.yaml` via `load_config(training_kind="sft")` | `model_loader.py:233-236` |
| `load_in_4bit` | **true** (hardcoded) | same | — | `model_loader.py:245,254` |
| split | `datasets/augmented`::`test`, **3004 rows** | same | — | `run_inference.py:189` |
| resume | **none** — file opened `"w"`, truncated each run | same | — | `inference.py:314` |

## A.4 Measured token budgets

| | text prompt | vision @1.2 MP (1204224/1024) | worst target | **SFT total vs 3072** | GRPO prompt vs 2304 | completion vs 768 |
|---|---|---|---|---|---|---|
| `object_only` | **297** | 1176 | **390** (23 boxes, `0012470`) | **1863** ✓ (1209 margin) | ~1473 ✓ | **2.0×** headroom |
| `caption_only` | **164** | 1176 | **195** | **1535** ✓ (1537 margin) | ~1340 ✓ | **3.9×** headroom |

Target length distribution, characters, real cleaned split:

| split | task | median | p95 | p99 | max |
|---|---|---|---|---|---|
| train | `oo` | 85 | 139 | 199 | **483** |
| train | `co` | 323 | 487 | 559 | **742** |
| test | `oo` | 86 | 122 | 142 | 214 |
| test | `co` | 231 | 585 | 737 | **944** |

The on-ARC `--census` (§1) independently confirms both: `oo` worst 1846, `co` worst 1477, vs 3072.

## A.5 SLURM resources — CLI and in-file are byte-identical, so precedence is a no-op

| Stage | partition | `--gres` | cpus | `--mem` | `--time` | v2 `vo` measured (a *larger* job) |
|---|---|---|---|---|---|---|
| baseline | `gpu-h100` | `gpu:h100:1` | 8 | 150G | 12:00:00 | 0:47 / 1:19 / 1:19 |
| sft | `gpu-h100` | `gpu:h100:1` | 8 | 150G | 12:00:00 | 1:09 / 1:46 / 2:02 |
| merge | `gpu-h100` | `gpu:h100:1` | 8 | 80G | 01:30:00 | 0:01 / 0:04 / 0:02 |
| grpo | `gpu-h100` | `gpu:h100:1` | 8 | 250G | **24:00:00** | 5:24 / 9:22 / **11:27** |

`MEM_CONFIG` (`submit_pipeline.py:50-55`) and `TIME_CONFIG` (`:57-78`) match each script's own
`#SBATCH` exactly. No `--gres` is passed (no escape hatch used), so the in-file `gpu:h100:1` governs.
24:00:00 is **at**, not over, `gpu-h100`'s real `MaxTime` of `1-00:00:00`, so nothing is silently
unsubmittable. `--partition` and `--cpus-per-task` are never overridden.

## A.6 Rewards

| task | components (YAML order) | weights | constants actually read |
|---|---|---|---|
| `object_only` | `reward_format`, `reward_grounding` | 0.10, 0.90 | `grounding_tn_constant` = `{excavator: 0.283, rebar: 0.048, worker_with_white_hard_hat: 0.065, default: 0.15}`; `repetition_penalty` = 1.0 (short-circuits at `unified_reward.py:96`) |
| `caption_only` | `reward_format`, `reward_caption` | 0.10, 0.90 | `repetition_penalty` = 1.0 |

`violation_tn_constant` and `violation_fbeta` resolve to their defaults (0.15 / 2.0) but are **never
read** — those modules are imported at `unified_reward.py:36-41` and never called for these tasks.
All constants funnel through `reward_utils.py:65-81 reward_constant`, which reads **only**
`configs/tasks/<task>.yaml` — not the merge chain — so `grpo.yaml`/`base.yaml` cannot override them.
