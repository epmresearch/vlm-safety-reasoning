# HANDOFF.md — session context, 2026-09-23

**Temporary.** Written to carry state into a fresh session. Delete it once the
`object_only` / `caption_only` v1 runs have landed and their outcome is recorded in
`README_v2.md`.

`CLAUDE.md` is loaded automatically and remains the engineering reference — architecture,
config layering, invariants, settled decisions, traps. **This file does not repeat it.**
It carries only what `CLAUDE.md` cannot know: what happened in the previous session, what
is uncommitted, and what the next job is.

---

## 1. Where things stand

| | |
|---|---|
| Cluster | **idle.** Nothing queued, nothing running. |
| Last commit | `3809e34` ("final") — the MOCS review UI. **None of the work below is committed.** |
| Working tree | **26 modified + 10 new files, all uncommitted** |
| Test suite | 802 passed, 1 skipped, local, ~60 s |
| Evaluated results on disk | `violations_only` v2 only (9 runs). Everything else has never produced numbers. |

### ⚠️ The uncommitted tree is the single most important fact here

The previous session built the `violations_think` arm. It is complete, independently
audited and fully tested — but **not committed**, and its changes touch shared code that
every task runs through:

```
core/tasks.py            data/preprocessor.py      evaluation/evaluator.py
data/loader.py           data/prompt_templates.py  data/schemas.py
experiments/results_lib.py  experiments/compare_all.py
experiments/run_{sft,grpo,evaluation}.py
models/grpo_trainer.py   scripts/submit_pipeline.py  scripts/validate_rewards.py
scripts/hpc_{baseline,sft,merge_sft,grpo}.sh        scripts/export_data_for_inspection.py
```

Two consequences:

1. **`object_only` and `caption_only` will execute on top of these changes.** They have
   *never* run end to end, so this is a first run of two pipelines on a tree that just
   gained a fifth task and a data-routing override chain. That combination is why the
   next job is an audit, not a submission.
2. **Every run manifest will record `git_is_dirty: true`** unless the tree is committed
   first — the exact problem `README_v2.md` §13 P0-4 exists to fix, and all nine v2
   manifests already carry it.

The additive claim is asserted and tested (`test_v3_*` in
`tests/test_core/test_blocker_fixes.py`), but it was written by the same session that made
the changes. It has not been checked by fresh eyes against `object_only` / `caption_only`
specifically.

---

## 2. The next job

**Run `object_only` and `caption_only` at 2b / 4b / 8b, version `v1`** — the full chain
each: baseline ‖ (SFT → merge → GRPO), with inference + structural repair + evaluation at
every trained phase. 24 jobs.

**Before submitting: a deep audit of the whole pipeline, no code edits.** The master
prompt for that audit is §6 below.

### Why these two tasks, now

The MOCS harvest is with a civil-engineering student and will take days to weeks. It
cannot help either of these tasks even in principle:

- `object_only` needs `excavator` / `rebar` / `worker_with_white_hard_hat`. MOCS has 13
  categories of its own and **none of them are these three** — its rows would be
  *unannotated*, not empty.
- `caption_only` needs captions. MOCS captions are Qwen3-VL-32B teacher output, not human
  ground truth.

Neither has a scarcity problem anyway. Measured on ARC: excavator 34.5% prevalence → 0.0%
of batches empty, rebar 12.1% → 1.6%, hard-hat 9.8% → 3.7%, against rule_4's 0.67% →
80.8%. That contrast is the entire reason the harvest exists, and it does not apply here.

So these two are data-complete today and will be no better after the harvest. Running them
now is free parallelism, and it takes the project from one evaluated pipeline to three.

### Version and a leftover to check

Use `--version v1` for both — genuinely their first run. But two 4-job `co-*-v1` chains
were submitted **by accident** on 2026-09-14 (`48501131`–`48501137`) and cancelled; their
artifacts were deleted and verified gone. Confirm nothing survived before reusing the name:

```bash
ls $VLM_DATA_ROOT/checkpoints/qwen3vl-*/ | grep -E 'co-|oo-'
ls $VLM_DATA_ROOT/results/inference/    | grep -E '^co-|^oo-'
```

---

## 3. What the previous session did

1. **Shipped the MOCS review package** to the civil student — the offline review app
   (`mocs_annotation/review_ui.py`, `build_review.py`), a reviewer README, 5,056 staged
   images across four queues. Committed as `3809e34`. **Awaiting `review_results.json`.**
2. **Built the `violations_think` arm** end to end. See
   [`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md) for what exists and
   [`PLAN_V3_THINK.md`](PLAN_V3_THINK.md) for the design. Not run; blocked on the two
   datasets that do not exist yet (`datasets/augmented_v3`, `datasets/grpo_pool_v3`).
3. **Designed the MOCS → ConstructionSite combine** (four scripts: harvest, combine,
   computed augmentation, pool rebuild). **Not written.** Blocked on the student.
4. **Decided to run oo/co now** — this file's §2.

---

## 4. Settled this session — do not re-litigate

These were argued and closed. Reopening one needs new evidence.

| decision | why |
|---|---|
| `violations_think` is its **own task** (`vt`), not a flag on `violations_only` | keeps `vo-*-v2` byte-reproducible; every writable path is prefix-namespaced |
| **No `think_field:` YAML key** — the column name lives in `core/think_format.py::THINK_FIELD` | SFT target builders are called `builder(raw)` and never see a config, so the key could not reach the code that reads it — a ghost variable by construction |
| **`reward_think_consistency` not implemented at all** | user decision. It is the only lever RL has on the block; deliberately absent for the first run so v3 stays comparable to v4 |
| Prompt: the worked example sits on **rule_2, not rule_1** | rule_1 is the dominant, over-flagged class; `prompt_templates.py`'s own header records why an earlier rule_1 example was removed |
| Prompt: **no single-line constraint** on the description, and **no** "report no box" clause | a newline in the description is harmless (the parser accepts multi-line); a box-less violation scores 0 on `reward_violation_grounding`, so inviting it forfeits 0.317 of the weight |
| v3 augmentation multipliers will be **computed from realised counts**; `RULE_MULTIPLIERS` and `datasets/augmented` stay **frozen** | v2 reproducibility + `test_v2_augmentation_multipliers_unchanged` |
| The arm matrix: **v2** (reference) / **v4** (data effect) / **v3** (block effect) | v3 − v4 isolates the block with the same data on both sides |

---

## 5. Open items, in order

| # | item | blocked on |
|---|---|---|
| 1 | **Audit, then submit oo/co v1 × 3 tiers** | nothing — this is the next job |
| 2 | Commit the 36-file tree | a decision; it should probably happen *before* (1) so manifests are clean |
| 3 | Harvest + combine the verified MOCS data (4 scripts, designed not written) | the student's `review_results.json` |
| 4 | Build `datasets/augmented_v3` + `datasets/grpo_pool_v3` | (3) |
| 5 | Run v4 and v3 | (4) |
| 6 | Cross-arm pairing in `compare_all` (`--compare-to task:tier:version`) | nothing to pair until (5) lands. The known, deliberate gap. |
| 7 | `README_v2.md` §13 P0-1 … P0-3 (checkpoint sweep, merged-no-adapter eval, `structural_*_raw`) | nothing — independent of all the above |

---

## 6. The master prompt for the new session

Paste this after pointing the session at this file.

---

> Read `HANDOFF.md`, then `CLAUDE.md`, then `V3_THINK_IMPLEMENTATION.md`. Do not trust any
> of them — they are written by previous sessions and several claims in `CLAUDE.md` have
> already been found stale. Verify everything you rely on against the actual code.
>
> **Mission: a deep, adversarial audit of this repo before I submit 24 GPU jobs.** I am
> about to run `object_only` and `caption_only` at 2b, 4b and 8b, `--version v1`, full
> chain each (baseline ‖ SFT → merge → GRPO, with inference + structural repair +
> evaluation at every trained phase). Neither task has ever run end to end, and the
> working tree has 36 uncommitted files that touch shared code paths. I want **zero bugs**
> before I submit.
>
> **Do not edit any code.** This turn is audit only. Report findings; I will decide what
> to change.
>
> Produce a written report (a new `.md` file) covering, at minimum:
>
> **A. Every parameter that will actually apply.** For `object_only` and `caption_only`,
> at each of 2b/4b/8b, resolve the full merged config (`base → model_registry → {sft,grpo}
> → tasks/<task>`) and tabulate every value that reaches runtime: learning rate and
> schedule, warmup, epochs, per-device batch size, gradient accumulation, effective batch,
> **computed step count**, LoRA r/alpha/dropout/target modules and the resulting trained
> parameter fraction, all three length keys (`max_seq_length`, `SFTConfig.max_length`,
> `max_prompt_length`/`max_completion_length`), `max_new_tokens`,
> `inference_max_seq_length`, image pixel bounds, save/eval steps, `save_total_limit`,
> persistent-checkpoint frequency, and every GRPO-specific key (`num_generations`,
> `steps_per_generation`, `beta`, `max_grad_norm`, `loss_type`, `scale_rewards`,
> `mask_truncated_completions`, `dataloader_drop_last`, `seed`, generation temperature /
> top_p / top_k). State where each value comes from.
>
> **B. Declared vs applied — the ghost-variable hunt.** For every key in A, prove it is
> actually *read* at runtime, not merely present in a YAML. This repo has eleven
> documented cases where config said one thing and the code did another, including one
> found last week. Find any that remain, especially on the `object_only` / `caption_only`
> paths, which have never been exercised. Check the `cfg` → `sft_cfg` copy-over in
> `models/grpo_trainer.py` specifically.
>
> **C. The complete data and artifact flow.** For every one of the 24 jobs and every stage
> inside it: what is loaded, from which directory and which split, by which line of code;
> and what is written, to which path. Include the HF cache, the SLURM logs, W&B, the
> oversample manifests, checkpoints, merged checkpoints, predictions, repair output and
> evaluation output. Confirm no two of the 24 jobs can write to the same path, and that
> nothing collides with the existing `violations_only` v2 results on disk.
>
> **D. Save/load correctness.** The `final/` vs `best/` handoff, the merge guard, the
> merged-checkpoint name round trip across `submit_pipeline.py` / `hpc_merge_sft.sh` /
> `hpc_grpo.sh` / `run_inference.py`'s reverse-engineering regex, and
> `results_lib.py::parse_run_name` as its inverse. Prove a name written by one stage is the
> name the next stage looks for.
>
> **E. Rewards, A to Z.** Which components are active for each of the two tasks, their
> weights, every constant they read (`grounding_tn_constant` per class,
> `violation_tn_constant`, `violation_fbeta`, `repetition_penalty`), the break-even
> arithmetic, and whether any degenerate policy beats the honest one. Run
> `scripts/validate_rewards.py` for both tasks and interpret the output rather than just
> reporting pass/fail.
>
> **F. Evaluation, A to Z.** Which metric families run for each task and why
> (capability gating), which need a JVM / images / CLIP / sentence-transformers, exactly
> which metric keys each task will emit, and which keys will be **absent** versus **zero**.
> Confirm the LLM judge is correctly skipped for both.
>
> **G. Prompts and targets.** The exact prompt and exact SFT target for each task, and
> whether they agree with each other and with the output schema and the parser.
>
> **H. Runtime-only failure modes** — the things that cannot fail locally: missing caches
> on a compute node with no internet, Java, VRAM/OOM at 8b, walltime against the
> partition's real `MaxTime`, completion truncation, host RAM during dataset
> materialisation, and disk/quota for six new merged 16-bit checkpoints.
>
> **I. What I am missing.** Anything not covered above that would cost a job, corrupt a
> result, or make the numbers uninterpretable.
>
> Rank every finding: **blocker / should-fix / cosmetic**, each with a file:line and a
> concrete failure scenario. Distinguish "documented and correct", "documented but stale",
> "undocumented", and "actual bug". End with an explicit **go / no-go** on submitting the
> 24 jobs.
>
> Constraints: run the test suite **locally only**, never on ARC — one test shells out to a
> real `sbatch` and has already submitted 8 unwanted GPU jobs. Verify TRL-default claims
> against the ARC-pinned `trl==0.23.0`, not whatever is in the local venv. Do not commit or
> push. Take as long as you need; correctness matters far more than speed.

---

## 7. Environment reminders

- Local dev box is Windows; `experiments/run_{sft,grpo,inference}.py` cannot even be
  imported here (`run_sft.py` imports `unsloth` before transformers). The submitters run
  locally as a dry run — no `sbatch`, so they print the exact commands with
  `DUMMY_JOB_ID`.
- On ARC: `pytest tests/ -k "not submitter_can_override_gres"`. **Never the bare suite.**
- Never commit or push unless asked.
