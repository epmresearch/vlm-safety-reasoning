# V3_THINK_IMPLEMENTATION.md — what was built for the `violations_think` arm

**Status: implementation complete and verified. Nothing has been trained.** The arm is ready to
submit as soon as `datasets/augmented_v3` and `datasets/grpo_pool_v3` exist.

Companion to [`PLAN_V3_THINK.md`](PLAN_V3_THINK.md), which is the design. This file is the record of
what was actually done, what changed, what was deliberately left out, and how to run it.

| | |
|---|---|
| Implemented | 2026-09-21 |
| Test suite | **797 passed, 1 skipped** = 798 collected (was 686 collected — **+112**) |
| New files | 10 |
| Modified files | 26 |
| Bugs found and fixed | **6**, plus 8 doc/code mismatches. One was found only by the independent audit (§4.7) |
| Independently audited | yes — a separate agent read all 36 files, ran the new + modified test files, probed the parsers with constructed inputs, and emulated the shell argument handling. Every substantiated finding is fixed; §13 lists them. |
| Trained yet | **No.** Blocked only on the two datasets |

---

## 1. The three arms this gives you

| arm | command | SFT data | GRPO pool | think block | isolates |
|---|---|---|---|---|---|
| **v2** | `--task violations_only --version v2` | `datasets/augmented` | `datasets/grpo_pool` | no | the existing reference, **untouched** |
| **v4** | `--task violations_only --version v4 --sft-dataset datasets/augmented_v3 --grpo-pool datasets/grpo_pool_v3` | `augmented_v3` | `grpo_pool_v3` | no | **the data** (v4 − v2) |
| **v3** | `--task violations_think --version v3` | `augmented_v3` | `grpo_pool_v3` | **yes** | **the block** (v3 − v4) |

Any tier subset, any number of arms at once:

```bash
python scripts/submit_vt_pipeline.py --tiers 2b --version v3                 # 4 jobs
python scripts/submit_vt_pipeline.py --tiers 2b 4b 8b --version v3           # 12 jobs
python scripts/submit_pipeline.py --task violations_think --tiers 4b --version v3
```

Verified by dry-run across **all 5 tasks × {1, 2, 3} tiers**: always 4 jobs per tier, correct names,
no collisions. v3 and v4 can be in the queue simultaneously with v2's results on disk.

---

## 2. What was added

| file | lines | what it is |
|---|---|---|
| `core/think_format.py` | 434 | **The wire format, defined once.** Tags, the canonical body builder, a tolerant parser, the raw-completion extractor, and `think_row_problems` — the row validator. Three consumers with deliberately different strictness (see §4.1). |
| `configs/tasks/violations_think.yaml` | 84 | The task config. Everything reward-related copied **unchanged** from `violations_only.yaml`; only the dataset routing and the prompt key differ. |
| `evaluation/metrics_think.py` | 184 | The block diagnostics — the only visibility into the block after training. |
| `scripts/validate_think_dataset.py` | 304 | Pre-flight validation of a pre-baked `thinking` column. Exit 0 / 1 / 2. |
| `scripts/submit_vt_pipeline.py` | 37 | A 3-line shim, for symmetry with the other four. |
| `tests/test_core/test_think_format.py` | 44 tests | The format, its parser, and every failure mode of the validator. |
| `tests/test_data/test_preprocessor_vt.py` | 14 tests | The SFT target, including the byte-identity guarantee. |
| `tests/test_evaluation/test_metrics_think.py` | 19 tests | The diagnostics and the evaluator gating. |
| `PLAN_V3_THINK.md` | — | the design |
| `V3_THINK_IMPLEMENTATION.md` | — | this file |

## 3. What was changed

### The task itself (5 files, all additive)

| file | change |
|---|---|
| `core/tasks.py` | one `TaskSpec`: `violations_think`, prefix `vt`, `{CAP_VIOLATIONS}`, fenced JSON |
| `data/prompt_templates.py` | `VIOLATIONS_THINK_PROMPT` + one registry entry. Composed from the **same** `_SAFETY_RULES` / `_VIOLATION_INSTRUCTIONS` fragments as `violations_only`, so rule wording cannot drift and the LLM judge's rubric stays in sync. |
| `data/schemas.py` | registers the **existing** `ViolationsOnlyOutput` object — not a copy |
| `data/preprocessor.py` | `_build_violations_think_target_json`, which validates the row then **delegates the JSON half to `_build_violations_only_target_json` verbatim**; registered in both dispatch tables, with `violations_only`'s GT builder reused as-is |

### The data-routing override chain (7 files)

This is what makes v4 possible without touching any file v2 resolves through.

| file | change |
|---|---|
| `data/loader.py` | `load_grpo_pool(subdir=None)` + `resolve_grpo_pool_subdir()` — **the bug fix**, §4.2 |
| `models/grpo_trainer.py` | `run_grpo(..., grpo_pool_subdir=None)`; passes `subdir=`; records `resolved_grpo_pool_subdir` and `grpo_pool_rows` in `run_manifest.json` |
| `experiments/run_grpo.py` | `--grpo_pool_subdir` |
| `experiments/run_sft.py` | `--sft_dataset_subdir`, which **beats** the task YAML; writes the effective value back into the config so `run_config.json` records what actually trained |
| `scripts/hpc_sft.sh` | optional 4th positional |
| `scripts/hpc_grpo.sh` | optional 5th positional |
| `scripts/submit_pipeline.py` | `--sft-dataset`, `--grpo-pool`; appended to the SFT and GRPO argument lists **only when given** |
| `scripts/validate_rewards.py` | `--grpo-pool-subdir` and `--sft-dataset-subdir`; `--pool-stats` now defaults to the first `--task`'s own pool |
| `scripts/export_data_for_inspection.py` | fail-soft per task — §4.7 |

Precedence everywhere: **CLI flag → task YAML → `base.yaml` → literal default.** Passing nothing
resolves to the historical path byte-for-byte, for the SFT dataset, the GRPO pool, the token census
and the pool statistics alike.

One artifact does change for the existing four tasks, and only an artifact: every GRPO
`run_manifest.json` now also carries `grpo_pool_subdir_arg`, `resolved_grpo_pool_subdir` and
`grpo_pool_rows`. Training behaviour is identical (all four resolve `None` → `datasets/grpo_pool`);
the keys are recorded unconditionally because "which pool trained this run" is provenance you always
want on disk, not only when someone passed a flag.

### The diagnostics (3 files)

| file | change |
|---|---|
| `evaluation/evaluator.py` | `run_full_evaluation(..., model_texts=None)`; computes `think_*` when the task's **prompt** asks for a block |
| `experiments/run_evaluation.py` | extracts `model_texts` as `original_raw_output` falling back to `raw_output` — §4.5 |
| `experiments/results_lib.py` | a `think` metric family, three headline rates, seven support counts |

### The analysis toolset (1 file)

| file | change |
|---|---|
| `experiments/compare_all.py` | both table printers now iterate `FAMILY_ORDER` instead of a hardcoded family tuple — §4.3 |

### The phase-script comments (4 files)

All four `hpc_*.sh` headers listed four task names; they serve five.

### Tests (7 files)

`test_task_registry.py` (five tasks), and six files whose hardcoded 4-task loops now derive from
`VALID_TASKS` — §4.4. Plus the `test_v3_*` block appended to `test_blocker_fixes.py` (15 tests).

---

## 4. Bugs found and fixed

### 4.1 Not a bug, but the core design decision: one format, three strictnesses

`core/think_format.py` is the only definition of the format. `parse_think_body` **never raises** —
every deviation lands in a `problems` tuple — and each caller picks the policy:

| consumer | policy | why |
|---|---|---|
| `data/preprocessor.py` | **raises on the first bad row** | a contradictory target trains the model to invert evidence, and nothing downstream can detect it; better to fail before any GPU time is spent |
| `scripts/validate_think_dataset.py` | **collects every offender** | fixing a dataset one SFT crash at a time is miserable |
| `evaluation/metrics_think.py` | **tolerant** | it reads raw model output, where a deviation is the measurement; raising would lose a 3,004-image evaluation |

### 4.2 `load_grpo_pool` ignored the task config — pre-existing and silent

`configs/base.yaml:14` has carried `dataset.grpo_pool_subdir` all along, but `load_grpo_pool()` read
it from `load_base_config()` **only**, never from the merged task config. A task YAML setting that key
was silently ignored: the key existed, looked overridable, and was not.

Left unfixed, `violations_think` would have **trained on the v2 pool while its manifest claimed the v3
one** — the exact shape of every entry in CLAUDE.md's ghost-variable table, now added to it.

### 4.3 `compare_all.py` could never print a new metric family

Both table printers iterated a hardcoded
`("structural", "captioning", "grounding", "violation", "reasoning")`. The index is long-format and
derives `metric_family` from the key prefix, so the `think_*` keys reached the CSVs and charts but were
**invisible in the terminal tables** — the view anyone actually reads. Now both iterate `FAMILY_ORDER`,
which is self-maintaining.

### 4.4 Six test files silently covered one task less than they claimed

`test_blocker_fixes.py` (×4 sites), `test_ledger_fixes.py` (×2), `test_unified_reward.py`,
`test_evaluator_co.py` hardcoded a 4-task tuple while pinning **task-generic** invariants: the flat
`1.0e-4` SFT LR, the pinned GRPO defaults (`loss_type`, `mask_truncated_completions`, `scale_rewards`,
`seed`), the `_final` results suffix, `image_id` reaching the GT dict, the token-budget arithmetic, the
repetition penalty being off, and the JSON parse path. They now derive from `VALID_TASKS`, so the new
task is covered and any future one will be too. That is where **+2** of the new tests came from.

(One site was deliberately **left** hardcoded: `test_llm_judge.py:463` is parametrized over the tasks
*without* the violations capability. `violations_think` has it, so adding it there would assert the
judge is skipped for a task that must run it.)

### 4.5 The think metrics would have measured the repair stage

Evaluation runs on `repair_applied/predictions_repaired.jsonl`, and `structural_repair.py:1526`
**replaces** `raw_output` with re-serialized JSON for every record whose status is `fixed_valid`,
stashing the model's own text in `original_raw_output`. Records it left alone keep the original in
`raw_output` and carry no `original_raw_output` key at all.

So reading `raw_output` would have produced a block-presence rate that is partly a measurement of
repair — plausible-looking and wrong, which is precisely the failure this repo already paid for once
with `structural_json_validity_rate`. `run_evaluation.py` now passes
`original_raw_output or raw_output`, and when `model_texts` is absent the metrics are **skipped
entirely** (keys absent, never zero) rather than computed from the wrong text.

### 4.6 One of mine, caught before it shipped

`think_row_problems` initially read `raw["rule_1"]`, but dataset rows are keyed `rule_1_violation`.
Every genuinely violated row would have been reported as a block/label contradiction. Fixed by a named
`violations_from_row()` conversion, and pinned by
`test_think_format.py::test_violations_from_row_reads_the_suffixed_keys`.

### 4.7 Found by the independent audit: a script outside the change set

`scripts/export_data_for_inspection.py` defaults `--tasks` to `list(VALID_TASKS)` and iterated it with
**no error handling**. `violations_think` is registered third, and its `sft_dataset_subdir` points at
`datasets/augmented_v3` — which does not exist yet — so `load_processed_dataset` raised
`FileNotFoundError` and the documented no-flag invocation exported `unified` and `violations_only`,
then **died before reaching `object_only` and `caption_only`**.

A genuine additivity violation: registering a task silently broke an unrelated script. Now fail-soft
per task (the pattern `validate_rewards.py` already used), with a closing summary of what was skipped.
Verified against a root with `augmented_v3` deliberately absent — the ARC state today: `violations_think`
is skipped and all four other tasks still export.

**Two real logic defects the audit also caught**, both in code I wrote:

- **`build_think_body` guarded the caption but not the reason.** A GT reason containing a newline
  produced a valid-looking **6-line** body that `think_row_problems` then rejected with a positional
  line-count complaint — useless for locating the offending reason. Since the docs tell data prep to
  *import* this function to generate the column, a generator following that instruction would have
  produced a dataset that SFT refuses. Both now go through one `_reject_unusable_text` guard (blank,
  newline, brace, fence).
- **The problem tally bucketed on `problem.split(":")[0]`,** which for every per-rule message is the
  rule's own *name* — so `think_top_problem` read `"rule_1"` and the validator's tally told data prep
  *which rule* rather than *what was wrong*. Now a shared `problem_bucket()` strips the `rule_N: `
  prefix first; the same probe now returns `"asserted with an empty reason"`.

---

## 5. Deviations from the plan

| plan said | what was built | why |
|---|---|---|
| a `think_field:` key in the task YAML | **dropped**; the column name is `core/think_format.py::THINK_FIELD` | the SFT target builders are called as `builder(raw)` and never see a config, so the key could not reach the code that reads it. It would have read as configuration while being inert — a ghost variable created on purpose. Pinned by `test_v3_think_column_name_is_not_a_yaml_key`. |
| a ConstructionSite-only think dataset, to isolate the block | **not needed** | the v4 arm isolates the block better: v3 − v4 compares the block with the *same* data on both sides, instead of removing data to do it. One fewer dataset to build. |
| `reward_think_consistency`, off by default | **not implemented at all** | your decision ("keep it off"). Not writing it is cleaner than shipping unused reward code; §6 has the exact shape if you want it later. |
| gate the diagnostics on the task YAML's `think_field` | gated on whether the **prompt** asks for a block | `core/tasks.py` is explicit that no conditional should compare a task name to a literal, and "were we asking for one?" is the actual question — so a future think-style arm is covered with no edit. It also makes an all-zero result meaningful: for `violations_think` the keys are always present, so `think_block_present_rate: 0.0` reads as "the model stopped emitting blocks", not "the metric never ran". |
| pool sizing / walltime arithmetic | **out of scope** | your instruction. The plan's §5.3 table is still there if it becomes relevant; nothing in the code caps the pool. |
| — | `resolve_grpo_pool_subdir()` extracted as a pure function | so the precedence chain is unit-testable with no dataset on disk, and the trainer can record the *effective* path in its manifest |
| — | the validator also checks test-split identity and train/val→test id leakage | the plan listed these as dataset-level checks; they are implemented and both fire on a seeded fixture |

---

## 6. What was intentionally skipped, and exactly what

Each of these is a deliberate omission, not an oversight.

| skipped | exactly what is missing | why / what it would take |
|---|---|---|
| **`reward_think_consistency`** | a 5th reward component scoring whether the block's four verdicts match the JSON's four nullities | your call. It is the **only** lever RL has to shape the block directly (GRPO has no target — §7). Would be ~30 lines in `rewards/`, one entry in `REWARD_COMPONENTS`, one line in the task YAML, and a re-run of `validate_rewards.py --probe` because it shifts the `p*` arithmetic. |
| **Thinking-variant weights** | new `*-think` tier keys in `model_registry.yaml` | a separate arm. Task and weights are already orthogonal axes (`TaskSpec` carries no model), so it needs no change to anything built here — but it needs matching `lora_by_tier` entries or it silently falls back to `sft.yaml`'s flat `r=16` and reintroduces the adapted-fraction confound v2 fixed, **and** its own inference budget or its baseline is truncation-dead rather than bad. |
| **Free / loss-masked blocks** | masking the block out of the SFT loss | only coherent with Thinking weights; needs `max_seq_length` roughly doubled. |
| **Rejection-sampled reasoning (STaR)** | generate blocks, keep those whose JSON matches GT | the natural upgrade if a shared caption turns out to be worth less than per-rule evidence. Needs a generation pass over the whole SFT split. |
| **Per-rule reward weights / thresholds** | `README_v2.md` §13 P1-4 | orthogonal, and a better fit once there is a per-rule confidence to threshold on. |
| **Cross-arm pairing in `compare_all`** | a `--compare-to task:tier:version` flag for a paired bootstrap between v2/v3/v4 | **deferred, and this is the one real gap.** All three arms carry `violation_per_image_outcomes_b64`, so the comparison needs no reconstruction — but there is nothing to pair until runs exist, and I could only have tested it against fabricated outcome vectors. Worth doing the day the first v3 run lands; the CSVs and `significance.csv` are unaffected in the meantime. |
| **`P0-3` (`structural_*_raw`)** | pre-repair structural keys | pre-existing v3 priority, not part of this arm. Worth landing **first**: without it you cannot see whether the block hurt *raw* compliance, because repair papers over exactly that. |
| **The 36-item GT negation review** | hand-checking positive reasons with no negation cue | a data-prep prerequisite, not code. `rule_3: "Either side of the excavation trench is protected." → violated` would train a contradiction, and with a pre-baked column it is frozen in. |
| **Building the two datasets** | `datasets/augmented_v3`, `datasets/grpo_pool_v3` | explicitly yours to build later. §8 is the contract they must satisfy. |
| **Anything touching `unified` / `object_only` / `caption_only`** | — | untouched by design. |

---

## 7. Why GRPO needs no `thinking` column

Because **GRPO has no target text.** Its loss is `−A_i·log π(token) + β·KL`, where
`A_i = (r_i − mean₈)/(std₈ + 1e-4)` and `r_i` is the weighted sum of the reward functions scored
against ground truth. There is no teacher forcing and no reference sequence, so there is nowhere a
`thinking` string could go — this is structural, not a choice.

What GRPO *does* need is the **ground-truth dict** (labels, boxes, and reasons — `reward_reasoning`
compares against the GT reason), and that already flows:
`to_grpo_prompt_for_task` → `build_gt_dict` → a `ground_truth` column → TRL forwards it as a kwarg →
`unified_reward.py:116` parses it back. Verified: `to_grpo_prompt_for_task` reads only the prompt, the
GT dict, `image_id` and the image, and never touches `thinking` — so **the pool needs no code change
and no extra column.**

The block still *receives* gradient during GRPO (every completion token is weighted by the
sequence-level advantage), it just gets no *direct* signal. That is what the `think_*` diagnostics are
for, and `reward_think_consistency` is the only way to change it.

**SFT teaches the block. GRPO refines the decision.**

---

## 8. The dataset contract

### `datasets/augmented_v3/` — SFT input

| field | rule |
|---|---|
| `image` | **named `image`, singular** (invariant 1 — TRL looks for exactly that key) |
| `image_id` | unique |
| `image_caption` | non-blank, one line |
| `rule_1..4_violation` | `null` or `{bounding_box, reason}`, boxes in **[0,1]** |
| `thinking` | the block body, **inner text only, no `<think>` tags** |
| `provenance` | `constructionsite` \| `mocs` (reported, not enforced) |

`train` and `val` carry the combined data. **`test` must hold the same `image_id` set as `datasets/augmented`'s
test split** — `run_inference.py:189` loads the default root for every task, so every arm is scored on
the same 3,004 images automatically. The test split needs no `thinking` column (no target at
inference).

The column lives **in this dataset** rather than a separate one on purpose: a task that does not
declare a block ignores it, so one directory serves both v3 and v4 — which is what makes the ablation
free.

### The `thinking` body — exactly five lines

```
A worker is sitting on a scaffold attached to the formwork holding a pile of fracture rocks.
rule_1: The worker with a green jacket sitting on a scaffolding is not wearing a hard hat -> yes (1)
rule_2: no
rule_3: no
rule_4: no
```

- **violated** → `rule_N: {GT reason, trailing period stripped} -> yes ({box count})`
- **violated, no boxes** (MOCS reason-only rows) → `rule_N: {reason} -> yes`, count omitted
- **not violated** → `rule_N: no` — nothing else

All four lines on **every** image including safe ones (omitting them leaks the answer through the
block's presence), always in `rule_1 → rule_4` order (sorting by violation status leaks it through
position), and **no coordinates anywhere** in the block.

Every token is ground truth. The period is stripped only inside the block; the JSON payload below
keeps it, because `reward_reasoning` and the LLM judge score against the exact GT string.

**`core/think_format.py::build_think_body` is the executable definition** — if data prep generates the
column, import that rather than re-implementing the format.

### `datasets/grpo_pool_v3/` — GRPO input

Flat dataset, ~50/50 violation/safe, **no `thinking` column**. Keep the 50/50 ratio: `p* = 0.298` is
solved against it. Admit a MOCS row only if it carries a reason — a reason-less violation scores 0 on
`reward_reasoning`, a silent handicap.

---

## 9. The workflow, A to Z

```bash
# ---- 0. LOCAL: the suite must be green -------------------------------------
python -m pytest tests/ -v                 # 797 passed, 1 skipped
# On ARC, NEVER the full suite -- one test shells out to a real sbatch:
#   pytest tests/ -k "not submitter_can_override_gres"

# ---- 1. build the two datasets (yours; §8 is the contract) ----------------

# ---- 2. ARC, CPU login node: validate before spending a GPU-hour ---------
python scripts/validate_think_dataset.py                         # train + val + test identity
python scripts/validate_think_dataset.py --strict                 # also byte-identity vs canonical
#   exit 0 = safe to train | 1 = offending rows (with a tally) | 2 = dataset not found
#   report -> datasets/stats/think_validation_augmented_v3.json

python scripts/validate_rewards.py --task violations_think --probe --census --pool-stats
python scripts/validate_rewards.py --task violations_only  --probe --pool-stats   # for v4's pool
python scripts/preflight_grpo.py --tier 2b --task violations_think

# ---- 3. submit ------------------------------------------------------------
python scripts/submit_vt_pipeline.py --tiers 2b 4b 8b --version v3

python scripts/submit_pipeline.py --task violations_only --version v4 --tiers 2b 4b 8b \
    --sft-dataset datasets/augmented_v3 --grpo-pool datasets/grpo_pool_v3

# ---- 4. analyse -----------------------------------------------------------
python -m experiments.build_results_index --out results_index/index.json
python -m experiments.compare_all --index results_index/index.json --out results_index/
```

`--census` is not optional: a truncated completion is indistinguishable from a terrible model, because
every reward component returns exactly `0.0` for both "parse failed" and "parsed fine but scored zero".

### Reading the results, in this order

1. `violation_pred_positive_rate` against `violation_gt_positive_rate` (0.1368) — **before any recall
   number**
2. the support-counts table — rule_2/3/4 have 25/63/24 positives in the whole test split
3. `significance.csv` — paired tests, not point estimates
4. `repair_stats.csv::status:valid_raw:pct` — the honest structural number
5. the new `think_*` keys:

| key | what a bad value means |
|---|---|
| `think_block_present_rate` | low → the model stopped emitting blocks |
| `think_block_closed_rate` | low → completions are being truncated mid-block |
| `think_block_wellformed_rate` | low → the five-line shape is degrading |
| **`think_verdict_json_agreement_rate`** (+ per rule) | low → **the stated reasoning does not describe the answer given.** The interesting failure. |
| `think_block_words_{mean,p50,p95,max}` | drift / truncation pressure |
| `think_top_problem` | the most common structural defect, as text |

---

## 10. Verification performed

| check | result |
|---|---|
| Full suite | **797 passed, 1 skipped** |
| `test_v3_*` regression block | 15/15 |
| Format round-trip + 12 seeded row faults + 4 cosmetic variants | all caught / all accepted |
| `is_violation_asserted` vs `rewards/reward_utils._is_violation_present` | agree on all 19 value shapes |
| Target byte-identity (`vt` target ends with the `vo` target verbatim) | pass, plus a literal-string pin of the payload |
| GT dict + schema object identity with `violations_only` | pass |
| Contradictory / missing / junk `thinking` rows refused at build time | pass |
| `violations_only` ignores a junk `thinking` column | pass |
| Diagnostics vs a 5-record scenario (perfect / disagreeing / truncated / absent / malformed) | all 9 assertions exact |
| Evaluator gating: think keys absent for the other 4 tasks, present for `vt`, skipped without `model_texts` | pass |
| Validator against an on-disk fixture (clean / seeded-fault) | exit 0 / exit 1 / exit 2, all 6 fault classes reported |
| Submitter dry-run, 5 tasks × {1,2,3} tiers | 15/15 correct job counts and names |
| v2 submission unchanged (no extra positionals) | pass, asserted in a test |
| v4 overrides reach SFT + GRPO only, baseline/merge untouched | pass, asserted in a test |
| Reward probe for `violations_think` | PASS, `c=0.300`, `p*=0.298` — identical to `violations_only` |
| `build_results_index` + `compare_all` over a fake `vt` results tree | 4 runs indexed, think family in headline + support tables, 26 charts, no crash |
| `bash -n` on both phase scripts; CRLF count 0 | pass |

---

## 11. Traps

1. **Re-running `--version v2` destroys the existing v2 results.** `run_inference_batched` opens
   `predictions.jsonl` in `"w"` with no resume, so it truncates the file `README_v2.md` was built
   from. Archive `results/inference/vo-*-v2*` first, or re-run as `v2r`.
2. **A contradictory `thinking` row** is the one silently damaging input. `validate_think_dataset.py`
   is not optional.
3. **MOCS rows must never reach the test split** — checks 9–10 of the validator.
4. **`--census` before training**, every time the block format or the budgets move.
5. **The pool's violation ratio moves the operating point.** `p* = 0.298` assumes ~50/50. If
   `--probe` fails, the pool composition is wrong — do not "fix" it by editing
   `violation_tn_constant`.
6. **Never run the full suite on ARC** — `test_submitter_can_override_gres_for_every_stage` shells out
   to a real `sbatch` and submitted 8 unwanted GPU jobs on 2026-09-14.
7. **`.gitattributes` forces LF** on `*.sh`/`*.py`/`*.yaml`; both phase scripts were written with LF
   and verified.
8. **`--tiers` accepts any string** (no `choices=`), so `--tiers 8B` submits four jobs that fail on
   the compute node after the queue wait. Check spelling by eye.

---

## 12. Audit findings and their disposition

An independent agent audited the change set. Disposition of every finding:

| # | finding | disposition |
|---|---|---|
| F1 | `export_data_for_inspection.py` aborts before `object_only`/`caption_only` | **fixed** — §4.7 |
| F2 | GRPO `run_manifest.json` gains 3 keys for existing tasks; `base.yaml` parsed twice | **kept, documented** (§3: provenance is wanted unconditionally); the double parse is **fixed** — `resolve_grpo_pool_subdir` short-circuits and the trainer passes the already-resolved value |
| F3 | `ParsedThink.n_lines` written, never read | **fixed** — field removed |
| F4 | `check_split(ds, split_name, …)` never uses `split_name` | **fixed** — parameter removed |
| F5 | `build_think_body` guards the caption but not the reason | **fixed** — §4.7 |
| F6 | problem tally buckets on the rule name, not the defect | **fixed** — §4.7 |
| F7 | docstring claimed the caption comparison is normalised; it is verbatim | **fixed** — the docstring now says verbatim, and why |
| F8 | `loader.py` named `preflight_grpo.py` as a caller; it never calls `load_grpo_pool` | **fixed** — docstring corrected, including what preflight does and does not catch |
| F9 | the `violations_think` note split CLAUDE.md's naming table in two | **fixed** — it now has a proper column instead |
| F10 | all four `hpc_*.sh` headers list four tasks | **fixed** in all four |
| F11 | validator docstring said the test split is checked "IDENTICAL"; it compares id sets | **fixed** — both the docstring and this file now say `image_id` set |
| F12 | "reports every offender" but stdout is capped and the JSON report had no ids | **fixed** — the report now carries the complete `offenders` list |
| F13 | the prompt asks for `-> yes (n)` unconditionally; the target omits the count when there are no boxes | **resolved the other way, 2026-09-21.** The clause covering the box-less case was added and then removed. A box-less violation still earns identification credit but scores **0 on `reward_violation_grounding`** (0.317 of the weight), so telling the model it may report no box invites it to forfeit that. The prompt describes what the model should DO; `build_think_body` still handles the shape, because it is legal in ground truth — the review UI lets a human assert a rule without drawing a box, so a verified MOCS row can carry a reason and no geometry. The rare target teaches itself from the data. |
| F14 | stale line counts; test-count convention flipped | **fixed** — counts regenerated from disk, and CLAUDE.md is back to counting collected tests (798) |
| — | `validate_rewards.py --census` had no dataset override, so the precedence claim did not hold for it | **fixed** — `--sft-dataset-subdir` added and threaded to `census` and `sft_stats` |
| — | `error_analyzer.py` calls the evaluator without `model_texts` | **left as-is.** CLAUDE.md already lists it as not called from any live pipeline path; it logs the skip warning and emits no `think_*` keys, which is the intended policy. |
| — | `compare_all.py:94` assigns `caps` and never uses it | **left as-is** — pre-existing, outside this change. |

## 13. Lifecycle of these two documents

Both are temporary. Once the arm has run:

- results → [`README_v2.md`](README_v2.md)
- decisions and traps worth keeping → [`CLAUDE.md`](CLAUDE.md) (already updated with the task, the new
  files, the ghost-variable entry and the corrected `PROMPT_REGISTRY` claim)
- then **delete `PLAN_V3_THINK.md` and this file** — the documentation map allows four documents, and
  neither of these is one of them.
