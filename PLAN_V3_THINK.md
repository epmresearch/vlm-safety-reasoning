# PLAN_V3_THINK.md — the `violations_think` arm

> ## ⚠️ This is the DESIGN. It has since been implemented, with deviations.
>
> **Read [`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md) for what was actually built.**
> Where the two disagree, the implementation doc wins. The deviations that matter here:
>
> - **§5.2's `think_field:` YAML key was dropped.** The SFT target builders are called as
>   `builder(raw)` and never see a config, so the key could not reach the code that reads the column —
>   it would have been a ghost variable created on purpose. The column name lives in
>   `core/think_format.py::THINK_FIELD` and nowhere else.
> - **§8.2's `reward_think_consistency` was not implemented at all**, by decision, rather than shipped
>   and defaulted off.
> - **§7's diagnostics are gated on the task's prompt**, not on a YAML key.
> - **§2's ConstructionSite-only ablation is unnecessary** — the v4 arm isolates the block better.
> - **§5.3's pool-sizing / walltime arithmetic is out of scope**; nothing in the code caps the pool.
> - **§10's cross-arm pairing in `compare_all` is deferred** — the one real gap, and the only one.
>
> Everything else below was built as written.

**Status of the RUNS: not started. Nothing has been trained.**

This is an **implementation plan**, not a results document and not a reference. It is deliberately
temporary:

| when | what happens to this file |
|---|---|
| now | the spec being built against |
| after the arm runs | results → [`README_v2.md`](README_v2.md), decisions + traps → [`CLAUDE.md`](CLAUDE.md) |
| then | **delete it** — the documentation map allows four documents, and this is not one of them |

`README_v2.md` §13 owns the *v3 priorities*. This file owns *how one of those arms is built*. It does
not restate v2 results; read `README_v2.md` for those.

---

## 1. What this is, in one paragraph

Add a fifth, fully independent pipeline — **`violations_think`**, prefix **`vt`** — whose SFT target is
a `<think>` block prepended to a **byte-identical** `violations_only` JSON payload. Same Instruct
weights as v2, same schema, same four reward components at the same weights, same shared GRPO pool
machinery. The only variable is the target text.

Everything existing stays runnable and byte-reproducible: `violations_only` v2 can be re-run, and a new
**v4** arm runs v2's exact settings on the new data *without* a think block, which is what makes the
data effect and the block effect separately measurable.

## 2. The three arms

| arm | task | `--version` | SFT data | GRPO pool | think block | what it isolates |
|---|---|---|---|---|---|---|
| **v2** | `violations_only` | `v2` | `datasets/augmented` | `datasets/grpo_pool` | no | the existing reference; **untouched** |
| **v4** | `violations_only` | `v4` | `datasets/augmented_v3` | `datasets/grpo_pool_v3` | no | **the data** (v4 − v2) |
| **v3** | `violations_think` | `v3` | `datasets/augmented_v3` | `datasets/grpo_pool_v3` | **yes** | **the block** (v3 − v4) |

All three are safe to have on the cluster simultaneously. Every generated name carries both the task
prefix and the version (`vo-sft-4b-v2`, `vo-sft-4b-v4`, `vt-sft-4b-v3`), and
`tests/test_core/test_name_isolation.py` enumerates every writable name across all tasks × tiers ×
versions and asserts the set has no duplicates. That test is the proof, not a promise.

**v3 and v4 are the comparison that matters.** v3 alone changes two things at once (data *and* target),
so on its own it cannot attribute its own result. v4 costs one extra 12-job grid and converts
"v3 is better" into "the data is worth X, the block is worth Y".

### Test-split comparability is automatic

`experiments/run_inference.py:189` calls `load_processed_dataset()` with **no subdir override**, so
inference always reads `datasets/augmented`'s test split — the original 3,004 ConstructionSite images —
for every task and every version. v2, v3 and v4 are therefore scored on byte-identical images with no
special handling.

**Consequence, and it is a hard requirement:** MOCS rows may enter `train` and the GRPO pool. They must
**never** enter the test split. `datasets/augmented_v3`'s test split must be byte-identical to
`datasets/augmented`'s (validator check §6.1). Test rows need no `thinking` column — there is no target
at inference.

---

## 3. Why GRPO needs no thinking ground truth

**GRPO has no target text at all.** For each prompt (one image) the policy generates
`num_generations: 8` completions; each is scored by the task's reward functions; then
`A_i = (r_i − mean₈) / (std₈ + 1e-4)` becomes that completion's gradient weight. There is no teacher
forcing and no reference sequence, so there is nothing a `thinking` string could be compared against.
**SFT is the only phase with a target, and is therefore the only phase where the block is taught.**

What GRPO *does* need from the data:

| needs | supplied by |
|---|---|
| the image, in a column named exactly `image` | the pool (invariant 1 — TRL's rollout code looks for that key) |
| the prompt | `data/preprocessor.py::to_grpo_prompt_for_task` → `PROMPT_REGISTRY` |
| **ground truth** — rule labels, boxes, **and reasons** | `build_gt_dict(raw, task)` → `build_violations_only_ground_truth` |

`reward_reasoning` compares the model's JSON reason against the GT reason, so GT *reasons* are
load-bearing in the pool. GRPO needs ground truth; it does not need ground-truth text to imitate.

Verified: `to_grpo_prompt_for_task` (`preprocessor.py:479`) reads only the prompt, `build_gt_dict`,
`image_id` and the PIL image. It never touches a `thinking` field, so **no code change is needed for the
pool**, and a `thinking` column in the pool would simply be dead weight.

Three consequences to hold on to:

1. **The think block still receives gradient during GRPO.** The policy-gradient loss applies to *every*
   completion token, weighted by the sequence-level advantage — so block tokens are reinforced whenever
   the JSON that followed them scored well. The block is not frozen; it is shaped indirectly by
   outcome. What is missing is any *direct* signal about the block's own quality.
2. **Nothing constrains it, so it can drift.** No reward reads it, and `repetition_penalty` is `1.0`
   (off) by explicit decision. The `think_*` diagnostics (§7) exist to catch this. The optional
   `reward_think_consistency` component (§8.2) is the **only** lever RL has to shape the block directly,
   because RL's only mechanism is a reward, not a target.
3. **If the block degrades while F1 improves, that is a result, not a bug** — and it is only visible
   because the diagnostics are there.

---

## 4. Dataset contracts

Both directories are assumed to exist. Building them is out of scope for this plan; this section is the
spec they must satisfy.

### 4.1 `datasets/augmented_v3/` — SFT input

ConstructionSite + student-verified MOCS, pixel-augmented, **carrying a `thinking` column**.

| field | type | rule |
|---|---|---|
| `image` | HF `Image()` | **named `image`, singular** |
| `image_id` | str | unique |
| `image_caption` | str | non-blank |
| `rule_1..4_violation` | `null` \| `{bounding_box, reason}` | boxes in **[0,1]** — the existing builder scales to 0–1000 |
| `thinking` | str | **inner text only, no `<think>` tags** — see §4.3 |
| `provenance` | str | `constructionsite` \| `mocs` |

Splits: `train` and `val` carry the combined data. **`test` must be byte-identical to
`datasets/augmented`'s `test`** — it is what every arm is scored on.

The `thinking` column lives here rather than in a separate directory on purpose: a task that does not
declare a think block simply ignores the column, so **this one dataset serves both v3 and v4**. That is
what makes the block-vs-data ablation free.

### 4.2 `datasets/grpo_pool_v3/` — GRPO input

Same sources, flat (non-split) dataset, ~50/50 violation/safe. **No `thinking` column needed.**

Composition rules, carried over from `data/build_grpo_pool.py` and extended:

- Keep **~50/50 violation/safe**. The `violation_tn_constant: 0.30` break-even arithmetic is solved
  against 50/50, not against the ~88%-safe test distribution. A different ratio silently moves the
  operating point → §6.3 re-validation is mandatory, not optional.
- **Admit a MOCS row only if it carries a reason.** A verified label with no reason scores 0 on
  `reward_reasoning` — a silent handicap that looks like a bad model.
- **Target a stated rule ratio and record it in the manifest.** Today's pool is 677 rule_1 : 46 rule_4 =
  **14.7 : 1**, which is v2 finding #3 and `README_v2.md` §13 P1-3. MOCS rare-rule images fix this by
  *addition* rather than by discarding rule_1 rows.
- **Cap the total.** See §5.3 — pool size directly sets GRPO walltime, and 8B is already at 11.5 h of a
  24 h wall.

This is **not** a per-task pool, and does not reopen that settled decision. It is a new *generation* of
the same task-blind pool: any task can point at either, and `grpo_pool_v3` would serve `unified`,
`object_only` or `caption_only` equally well.

### 4.3 The `thinking` field format

Inner text only. Exactly five lines: the caption verbatim, then `rule_1:` … `rule_4:` in that order, on
**every** image including safe ones — omitting or shortening the block on safe images would leak the
answer through its mere presence.

```
A worker is sitting on a scaffold attached to the formwork holding a pile of fracture rocks.
rule_1: The worker with a green jacket sitting on a scaffolding is not wearing a hard hat -> yes (1)
rule_2: no
rule_3: no
rule_4: no
```

- **violated** → `rule_N: {GT reason, trailing period stripped} -> yes ({box count})`
- **violated, no boxes** (MOCS reason-only rows, which `schemas.py::_violation_to_dataset_scale`
  deliberately preserves) → `rule_N: {reason} -> yes`, count omitted
- **not violated** → `rule_N: no`. Nothing else. No authored text anywhere in the block.

Every token is ground truth: GT caption, GT reason, GT label, GT box count. That property is what makes
it defensible, and it is the reason a templated negative line ("no scaffold described") is excluded — a
regex-derived clause is invented text and can be flatly false.

**Why the `<think>` tags are not in the column:** the builder adds them. Tag changes then need no data
rebuild, and a half-written tag cannot enter the dataset silently.

**Measured cost** (from the other session's scan of all 3,004 test rows): mean 60.7 words, median 56,
max 187 — safe rows 58.2, violation rows 76.6. Comfortably short-CoT; no long-CoT degradation risk even
at 2B.

### 4.4 Prerequisite: the 36-item ground-truth negation review

`rule_3: "Either side of the excavation trench is protected." → violated` is a real annotation bug in
the ConstructionSite ground truth. Today it costs a little `reward_reasoning`. In a think block it
trains *"evidence says protected, therefore violated"* — and with a pre-baked column it is **frozen into
the dataset**.

A negation scan flags 36 of 435 positive reasons as candidates; most are legitimate ("The person on the
left is wearing shorts" correctly violates rule_1's uncovered-legs clause), only a couple are genuine
contradictions. **~10 minutes of hand review, before baking.** This moved from "nice to have" to a
prerequisite the moment the column became pre-baked.

---

## 5. Code changes

### 5.1 The six-step new-task checklist, made concrete

| # | file | change |
|---|---|---|
| 1 | `core/tasks.py` | one `TaskSpec`: `name="violations_think"`, `prefix="vt"`, `capabilities=frozenset({CAP_VIOLATIONS})`, `output_format=FORMAT_FENCED_JSON`. Nothing else in the file moves. |
| 2 | `configs/tasks/violations_think.yaml` | new file — §5.2 |
| 3 | `data/prompt_templates.py` | `VIOLATIONS_THINK_PROMPT` + one `PROMPT_REGISTRY` entry |
| 4 | `data/schemas.py` | register the **existing** `ViolationsOnlyOutput` under the new task. Zero new schema — the wire format below the fence is byte-identical. |
| 5 | `data/preprocessor.py` | `_build_violations_think_target_json` + GT builder, registered in `_TARGET_BUILDERS` / `_GT_BUILDERS` |
| 6 | `tests/` | mirror the `_oo`/`_co` suites; a `test_v3_*` block in `test_blocker_fixes.py` — §9 |

**Nothing to add in** the SLURM phase scripts, the evaluator, structural repair, reward assembly, the
comparison tables, or the plots. All of them read the registry and gate on capabilities. No new GRPO
pool *mechanism* either.

#### The target builder must delegate

```
<think>
{thinking}
</think>
{ _build_violations_only_target_json(raw) verbatim }
```

Delegating to the existing v2 builder — rather than re-implementing the JSON — is what guarantees the
payload can never drift from `vo`'s. Pinned by a byte-identity test (§9).

#### The GT builder must delegate too

`_GT_BUILDERS["violations_think"] = build_violations_only_ground_truth`. Identical GT dict ⇒ identical
reward behaviour ⇒ the only variable is the target text.

#### The prompt must reuse `SAFETY_RULE_TEXTS`

`data/prompt_templates.py::SAFETY_RULE_TEXTS` is the single source of rule wording, and
`tests/test_evaluation/test_llm_judge.py` pins the training prompts by sha256 so the judge's rule text
and the training prompt provably cannot diverge. The new prompt **must** compose from that dict, never
restate the rules.

The prompt must also show the think block **and** the fenced JSON inline, or the instructions and the
worked template disagree with each other.

### 5.2 `configs/tasks/violations_think.yaml`

Copied from `violations_only.yaml`, with everything reward-related **unchanged** so the reward surface is
identical and the target is the only variable:

```yaml
task_name: violations_think
prompt_key: violations_think

sft_dataset_subdir: "datasets/augmented_v3"
grpo_pool_subdir:   "datasets/grpo_pool_v3"     # requires the §5.4 fix to be honoured

think_field: "thinking"                          # which column carries the block

# --- unchanged from violations_only.yaml ---
reward_components: [format, violation_id, violation_grounding, reasoning]
reward_weights:  {format: 0.05, violation_id: 0.422, violation_grounding: 0.317, reasoning: 0.211}
violation_tn_constant: 0.30
violation_fbeta: 2.0
repetition_penalty: 1.0
max_completion_length: 1024
max_new_tokens: 1024
inference_max_seq_length: 3200
```

`sft_dataset_subdir` is a **top-level** key — that is the existing convention
(`experiments/run_sft.py:77` reads `sft_cfg.get("sft_dataset_subdir")`), and `object_only`/`caption_only`
already use it. `grpo_pool_subdir` mirrors it at top level for symmetry, deliberately *not* nested under
`dataset:` where `base.yaml` keeps its own copy.

### 5.3 Step counts and walltime — the binding constraint

`dataloader_drop_last: true`, 2 epochs, effective batch 32 ⇒ **`steps = floor(N / 32) × 2`**.

| dataset | N today | steps today | 8B measured |
|---|---|---|---|
| SFT (`augmented`) | 8,198 | **512** | 2:01:50 (whole job) |
| GRPO (`grpo_pool`) | 1,732 | **108** | **11:26:53** against a **24 h** wall |

**GRPO is the constraint.** Scaling from the measured 8B run:

| pool rows | steps | projected 8B GRPO | verdict |
|---|---|---|---|
| 1,732 | 108 | 11.5 h | today |
| ~2,000 | 124 | ~13 h | comfortable |
| ~2,600 | 162 | ~17 h | acceptable |
| ~3,000 | 186 | ~19–20 h | tight — little room for a slow node |
| >3,500 | 216+ | >22 h | **expect a wall kill** |

A wall kill is survivable but not free: `models/grpo_trainer.py` auto-resumes from the last checkpoint
and `save_steps: 20`, so at most ~20 steps are lost and the response is to **re-submit the identical
GRPO job** (same variant name), which continues rather than restarts. Still, plan the pool at
**~2,000–2,600 rows** — enough to fix the 14.7:1 rule imbalance without spending the margin.

SFT is not the constraint: even at ~12,000 rows (750 steps) 8B SFT lands near 2:45. Two second-order
effects to note:

- **RAM scales linearly.** `--mem=150G` is load-bearing because `build_sft_dataset` fully materializes
  every decoded PIL image of the split into one Python list with no streaming path. 8,198 rows ≈ 30 GB
  decoded; 12,000 ≈ 43 GB. Fine, but re-measure if the combined set passes ~20,000 rows.
- **More `persistent-checkpoint-N`.** `PersistentCheckpointCallback` copies at every multiple of 100
  steps, so 750 steps yields seven per variant instead of five. Disk only.

**Data-prep note, not a plan item:** with ~300 real rule_2 images from the harvest, the
`RULE_MULTIPLIERS = {4: 16, 2: 12, 3: 6}` augmentation could be retired for those rules — which would
*shrink* the SFT set, remove the near-duplicate correlation, and reduce the memorisation of the rarest
rules' exact phrasing. Decide that when building `augmented_v3`.

### 5.4 The one real bug to fix: `load_grpo_pool` ignores task config

`configs/base.yaml:14` already carries `dataset.grpo_pool_subdir: "datasets/grpo_pool"`, and
`data/loader.py:222` reads it with a default — so the path is **not** hardcoded.

But `load_grpo_pool()` **takes no parameter and reads `load_base_config()`**, never the merged task
config. A task YAML setting that key today is **silently ignored**: the key exists, looks overridable,
and is not. This is the exact shape of every entry in CLAUDE.md's ghost-variable table.

Fix, mirroring the pattern `load_processed_dataset(subdir=None)` already proves:

```python
def load_grpo_pool(subdir: str = None):
    base_cfg = load_base_config()
    resolved = subdir or base_cfg["dataset"].get("grpo_pool_subdir", "datasets/grpo_pool")
```

and at the one call site, `models/grpo_trainer.py:178`:

```python
train_split = load_grpo_pool(subdir=cfg.get("grpo_pool_subdir"))
```

Resolution order: explicit argument → `base.yaml` → literal default. **With no task key set, the
resolved path is byte-identical to today**, which is what keeps v2 reproducible. Pinned by a test (§9).

The other two call sites (`scripts/validate_rewards.py:506`, `scripts/dataset_report.py`) gain the same
optional argument so a pool can be validated before it is trained on.

### 5.5 The override mechanism that makes v4 possible

v4 is `violations_only` — its YAML has no `sft_dataset_subdir` and no `grpo_pool_subdir`, and **adding
them would change v2's resolution**, which is not additive. So the override has to arrive at runtime.

Add **optional** flags, absent by default:

| entry point | flag |
|---|---|
| `experiments/run_sft.py` | `--sft_dataset_subdir` |
| `experiments/run_grpo.py` | `--grpo_pool_subdir` |
| `scripts/preflight_grpo.py` | `--grpo_pool_subdir` |
| `scripts/hpc_sft.sh`, `scripts/hpc_grpo.sh` | forwarded only when non-empty |
| `scripts/submit_pipeline.py` | `--sft-dataset`, `--grpo-pool` |

Precedence: **CLI flag → task YAML → `base.yaml` → literal default.** Every resolved path is written
into `run_config.json` / `run_manifest.json`, so a run's data provenance is recoverable from disk
without consulting this file.

Unset ⇒ today's behaviour, byte-for-byte. That is the whole robustness claim, and §9 pins it.

---

## 6. Validation

### 6.1 `scripts/validate_think_dataset.py` — new, CPU-only, fails loudly

The pre-baked column *is* the format, so a bad row is frozen in and nothing downstream would notice.
This validator is the safeguard that replaces the flexibility a runtime builder would have given.

Per row:

1. `thinking` present, a `str`, non-blank.
2. Exactly 5 lines; lines 2–5 begin `rule_1: ` … `rule_4: `, in that order.
3. **Verdict ↔ label agreement:** the `rule_N` line asserts a violation **iff** `rule_N_violation` is
   non-null. *The single most important check here* — a block disagreeing with its own label trains an
   outright contradiction, and no reward, metric or repair stage can see it.
4. If violated: the line's reason equals the GT `reason` with the trailing period stripped.
5. `(n)`, where present, equals `len(bounding_box)`; absent iff the box list is empty.
6. Line 1 equals `image_caption.strip()`.
7. **No `{` or `}` anywhere in `thinking`** — `structural_repair.py::_extract_outermost_braces` is the
   no-fence fallback path and could otherwise pick up the block instead of the JSON.
8. No backtick-fence characters in `thinking`.

Per dataset:

9. `test` split image-id set is **identical** to `datasets/augmented`'s `test`.
10. No `image_id` appears in both `train`/`val` and `test`.
11. Provenance histogram, per-rule counts, and row counts → written to a report next to the dataset.

Also asserted cheaply inside the preprocessor, so a hand-built dataset cannot slip past on a machine
where the script was never run.

### 6.2 Token budgets — the arithmetic says no change, but prove it

| | current `vo` | with the block |
|---|---|---|
| SFT (`max_seq_length: 3072`, feeds both the load window **and** `SFTConfig.max_length`) | prompt ~1,660 incl. vision + target | +~90 typical / +~280 worst ⇒ ~2,250 worst. Fits. |
| GRPO | `max_prompt_length` 2,304 + `max_completion_length` 1,024 = **3,328** vs `max_seq_length: 3600` | block ~280 worst + JSON ~300 worst ≈ 580 < 1,024. **Unchanged.** |
| inference | `max_new_tokens` 1,024, window 3,200 | 1,660 + 1,024 = 2,684. Fits. |

The prompt also grows (the block instruction + worked template, perhaps +80–120 text tokens) against a
measured worst-case prompt of 1,519 and a 2,304 cap. Room exists.

Proven by `scripts/validate_rewards.py --census` (tokenizes every target, derives the vision ceiling
analytically rather than sampling one image) and `scripts/preflight_grpo.py` (measures the real
text-only prompt length for the task).

**If census disagrees, rebalance rather than raise the total:** drop `max_prompt_length` from 2,304
toward ~1,792 and hand the slack to `max_completion_length`, preserving the 3,600 ceiling and its
272-token margin. Raising `max_seq_length` changes the model load window and the memory profile, and
would make v3 non-comparable to v2 on cost.

A truncated completion is indistinguishable from a terrible model — every reward component returns
exactly `0.0` for "parse failed" and for "parsed fine but scored zero" alike — so this is not a
theoretical check.

### 6.3 Reward re-validation is mandatory, because the pool changed

`violation_tn_constant: 0.30` implies a break-even confidence

```
p* = c(w_id + w_gnd + w_rsn) / [ c(w_id + w_gnd + w_rsn) + w_id + w_gnd·E[IoU] + w_rsn·E[reason] ]
   = 0.285 / 0.955 = 0.298
```

solved against a **50/50** pool and the measured class prevalences. A new pool with a different
violation ratio or different per-rule prevalences moves the unconditional-assertion expected value that
`--probe` tests.

```
python scripts/validate_rewards.py --task violations_think --probe --census --pool-stats
python scripts/validate_rewards.py --task violations_only  --probe --pool-stats   # for the v4 pool
```

`--pool-stats` recomputes prevalence from the pool on disk instead of using the hardcoded v2 figures.
The validator asserts `p*` stays inside `VIOLATION_BREAKEVEN_BAND = (0.20, 0.50)` and that no degenerate
policy beats the honest one. **If it fails, the pool composition is wrong — do not "fix" it by editing
the constant.**

---

## 7. New diagnostics — the block is invisible to every existing metric

New `evaluation/metrics_think.py`, keyed off the task YAML's `think_field` rather than a capability
(capabilities map to output *field families*; a think block is not one). Emits:

| key | why |
|---|---|
| `think_block_present_rate` | did the model emit a block at all |
| `think_block_wellformed_rate` | 5 lines, four rule lines, correct order |
| **`think_verdict_json_agreement_rate`** (micro + per rule) | **the interesting one** — disagreement means the stated reasoning and the answer diverged |
| `think_block_words_{mean,p50,p95,max}` | drift and truncation-pressure monitor |
| `think_caption_*` *(optional)* | score the **generated** caption line against GT with the existing captioning metrics |

That last one matters more than it looks. At training the caption is ground truth; at inference it is
generated, and four verdicts are conditioned on it. **A hallucinated caption poisoning all four verdicts
is a brand-new failure mode that no current metric can see.**

New keys appear in `compare_all` automatically — the index is long-format and `metric_family` is derived
from the key prefix, so nothing has to be maintained by hand. Register the rates in `results_lib.py`'s
headline/support lists; the word counts are already excluded from the win tally by
`is_non_comparable_key`, which skips counts and word lengths.

---

## 8. Rewards

### 8.1 Unchanged, deliberately

All four components at v2's weights — format `0.05`, violation_id `0.422`, violation_grounding `0.317`,
reasoning `0.211`. Identical GT dict via the delegated GT builder. **The reward surface is identical to
v2's, so the target text is the only variable.**

Verified against the code that the block passes through untouched:

| stage | why the block is harmless |
|---|---|
| parsing | `output_parser.py:20` — `strip_fences` uses `re.search(..., DOTALL)`, explicitly "ignoring any pre-text" |
| `reward_format` | `reward_format.py:38` — returns 1.0 for any completion whose fenced JSON validates; the prose check applies only to plain-text tasks |
| the other three rewards | all go through `_strict_parse_for_task`, i.e. the same fence-strip |
| structural repair | `structural_repair.py:10` — handles "fences, preamble, postamble" by design |

### 8.2 Optional, opt-in: `reward_think_consistency`

A fifth component: do the block's four verdicts agree with the JSON's four nullities. A 4-bit check,
nearly free, registered in `rewards/unified_reward.py::REWARD_COMPONENTS` and activated **only** by the
new task's YAML, so the other four tasks are untouched.

It is the only way to give the block direct gradient during RL (§3). **Off by default**, because it
changes the reward surface and weakens comparability with v2 and v4. If enabled, re-run
`validate_rewards.py --probe` — it shifts the `p*` arithmetic in §6.3.

---

## 9. The robustness contract — tests that must exist

These are the tests that make "additive, no conflicts, v2 still runnable" mechanically true rather than
asserted. All CPU-only, all local.

| test | pins |
|---|---|
| **config snapshot** | `load_config("violations_only", "sft")` and `(..., "grpo")` resolve byte-identical to a committed snapshot. Any accidental edit to `base.yaml` / `sft.yaml` / `grpo.yaml` that would perturb v2 fails here, immediately. |
| **target byte-identity** | for a fixture row, `_build_violations_only_target_json(row)` is unchanged, **and** `_build_violations_think_target_json(row)` ends with exactly that string |
| **default path resolution** | with no overrides: `load_grpo_pool` → `datasets/grpo_pool`, `load_processed_dataset()` → `datasets/augmented` |
| **override precedence** | CLI → task YAML → `base.yaml` → default, for both dataset keys; and the resolved value lands in the manifest |
| **name isolation** | extends automatically; additionally assert `vt` collides with no existing prefix and that the v2/v3/v4 name sets are pairwise disjoint |
| **prompt sha** | `VIOLATIONS_ONLY_PROMPT`'s sha256 is pinned, so adding a new prompt cannot perturb the existing one |
| **rule-text reuse** | `VIOLATIONS_THINK_PROMPT` contains every `SAFETY_RULE_TEXTS` value verbatim |
| **GRPO prompt purity** | `to_grpo_prompt_for_task` output for the new task is identical to `violations_only`'s, and does not reference `thinking` |
| **schema identity** | `SCHEMA_REGISTRY["violations_think"] is SCHEMA_REGISTRY["violations_only"]` |
| **think-block validator** | the §6.1 checks, against a fixture with one deliberately contradictory row |

Add them as a `test_v3_*` block in `tests/test_core/test_blocker_fixes.py`, alongside the existing
`test_v2_*` block that pins every v2 decision.

> ### ⛔ Never run the full test suite on ARC
>
> `test_blocker_fixes.py::test_submitter_can_override_gres_for_every_stage` runs the real submitter
> through `subprocess`, and `submit_pipeline.py::submit_job` only falls back to `DUMMY_JOB_ID` on
> `FileNotFoundError` — so on a login node, where `sbatch` exists, **it submits 8 real GPU jobs.** That
> happened on 2026-09-14. On ARC always
> `pytest tests/ -k "not submitter_can_override_gres"`.

---

## 10. Comparing the arms

`compare_all.py::print_significance` only compares runs inside one `(task, tier, version)`, so
`vt-v3` ↔ `vo-v4` ↔ `vo-v2` will not happen on its own.

The good news: **all three arms carry `violation_per_image_outcomes_b64`** in `metrics.json` (~4 KB of
base64 per run, one byte per image). A *paired* bootstrap between any two of them therefore needs no
reconstruction — unlike the v1↔v2 comparison, which had to be rebuilt from
`repair_applied/predictions_repaired.jsonl` by hand.

Plan item: add cross-arm pairing to the toolset — `--compare-to task:tier:version` on `compare_all` —
rather than writing another one-off script. CLAUDE.md already names this as a genuine improvement, and
explicitly prefers extending the results toolset over adding a fifth plotting script.

Read in this order when the runs land, per `README_v2.md`'s conventions:

1. `violation_pred_positive_rate` against `violation_gt_positive_rate` (0.1368) — **before any recall
   number.** A model that flags 60–90 % of images has bought its recall by over-flagging.
2. the support-counts table — rule_2/3/4 have 25/63/24 positives in the whole 3,004-image test split, so
   a per-rule F1 there carries a 95 % interval up to 0.33 wide
3. `significance.csv` — paired tests, not point estimates
4. `repair_stats.csv::status:valid_raw:pct` — the honest structural number, not the post-repair one
5. the new `think_*` keys

---

## 11. Order of work

| step | why it comes here |
|---|---|
| 1. **P0-3** — emit `structural_*_raw` from the pre-repair file | without it you cannot tell whether the block hurt **raw** compliance, because repair papers over exactly that (the 2B baseline's reported 0.981 vs its real 20.11 %) |
| 2. **P0-4** — re-run `validate_rewards.py` + the suite, then commit | all nine v2 manifests say `git_is_dirty: true`; the new arms' manifests should not inherit that |
| 3. the **36-item negation review** (§4.4) | must happen before baking, not after |
| 4. build `augmented_v3` + `grpo_pool_v3`, run `validate_think_dataset.py` | out of scope here, gated on the student's verification |
| 5. implement §5, run the §9 tests locally | |
| 6. pre-flight (§12) | |
| 7. submit **v4 and v3 together** | independent, non-colliding, and v4 is what lets v3 be attributed |

**Everything downstream of step 4 is gated on the student's `sample` queue.** The 150-image random
sample is the only unbiased estimate of teacher precision, and every yield projection for the harvest
(rule_2 53 → ~300 distinct, rule_3 98 → ~1,200, rule_4 42 → ~300) is speculative until it comes back.
Do not size the pool or plan the grid before then.

---

## 12. Pre-flight, in order

```bash
# local
python -m pytest tests/ -v                                   # all must pass, incl. the new test_v3_* block

# ARC, no GPU
python scripts/validate_think_dataset.py                      # train + val + the test-identity check
python scripts/validate_rewards.py --task violations_think --probe --census --pool-stats
python scripts/validate_rewards.py --task violations_only  --probe --pool-stats
python scripts/preflight_grpo.py --tier 2b --task violations_think
python -m experiments.build_results_index --out results_index/index.json   # confirm parse_run_name accepts vt-*

# caches (compute nodes have no internet)
ls ~/scratch/hf_cache/hub    # judge: meta-llama/Meta-Llama-3-8B-Instruct; plus all-MiniLM-L6-v2
```

Then dry-run the submitter on Windows — no `sbatch`, so it prints the exact commands with
`DUMMY_JOB_ID` — and eyeball that no `vt-*-v3` or `vo-*-v4` path collides with an existing `vo-*-v2` one.

---

## 13. Submission

```bash
# v3 — the think arm
python scripts/submit_pipeline.py --task violations_think --tiers 2b 4b 8b --version v3

# v4 — v2's exact settings on the new data, no think block
python scripts/submit_pipeline.py --task violations_only --version v4 --tiers 2b 4b 8b \
    --sft-dataset datasets/augmented_v3 --grpo-pool datasets/grpo_pool_v3
```

24 jobs, ~4 days wall-clock for both grids. Each tier is 4 jobs: `baseline` (independent) ‖
`sft → merge → grpo` with `afterok` dependencies. Optionally add a 3-line `scripts/submit_vt_pipeline.py`
shim for symmetry with the other four.

**`--tiers` accepts any string** (no `choices=`), so `--tiers 8B` is accepted, submits four jobs, and
only fails on the compute node after the queue wait. Check spelling by eye.

---

## 14. Traps, ranked

1. **Re-running `--version v2` destroys the existing v2 results.** `run_inference_batched` opens
   `predictions.jsonl` in `"w"` mode and has no auto-resume, so a v2 re-run truncates the file the
   current report is built from. Archive `results/inference/vo-*-v2*` first, or re-run as `v2r`.
2. **`load_grpo_pool` ignoring the task config** (§5.4). Until fixed, a task-YAML `grpo_pool_subdir` is
   silently ignored and v3 would train on the *old* pool while its manifest claimed the new one.
3. **A contradictory `thinking` row** (§6.1 check 3). Trains the model to invert evidence, and nothing
   downstream can see it.
4. **MOCS rows leaking into the test split.** Would break comparability with v2 *and* inflate results.
   Checks 9–10.
5. **Pool size overrunning the 24 h wall at 8B** (§5.3).
6. **`--census` not run**, and a silent truncation that is indistinguishable from a bad model.
7. **The prompt restating the rules** instead of composing from `SAFETY_RULE_TEXTS`, desyncing the
   training prompt from the LLM judge's rubric.
8. **A `{` in the block** reaching the repair fallback path (check 7).
9. **`unified` is unaffected and must stay so.** CLAUDE.md is explicit that MOCS rows must never be
   written into `datasets/augmented` — MOCS has 13 categories of its own, none of them
   `excavator`/`rebar`/`worker_with_white_hard_hat`, so empty means *unannotated*, not *absent*, and
   would teach `unified` that those objects are missing on thousands of images. The new-directory
   approach avoids this by construction; do not shortcut it.
10. **`.gitattributes` forces LF** on `*.sh`/`*.py`/`*.yaml`. Don't defeat it from Windows.

---

## 15. Explicitly out of scope

- Building `augmented_v3` / `grpo_pool_v3` — this file specifies them, does not create them.
- The Thinking-variant weights. `core/tasks.py` carries no model field, so task and weights are already
  orthogonal axes: a Thinking arm would be new `model_registry.yaml` tier keys (`*-think`) plus matching
  **`lora_by_tier` entries** — without those it would silently fall back to `sft.yaml`'s flat `r=16` and
  reintroduce the adapted-fraction confound v2 fixed (0.58 % at 8b). It would also need its own inference
  budget, or its baseline is truncation-dead rather than bad. A later arm, cleanly.
- Free/unsupervised think blocks with the loss masked over them.
- Rejection-sampled or self-distilled reasoning (STaR-style): generate blocks, keep only those whose JSON
  matches GT. The natural upgrade if per-rule evidence turns out to be worth more than a shared caption.
- Per-rule reward weighting / thresholds (`README_v2.md` §13 P1-4) — orthogonal, and a better fit once
  there is a per-rule confidence to threshold on.
- Any change to `unified`, `object_only` or `caption_only`.
