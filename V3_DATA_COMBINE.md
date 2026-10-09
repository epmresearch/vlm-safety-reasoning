# V3_DATA_COMBINE.md — building `augmented_v3` / `grpo_pool_v3`, and running v4 and v3

**Status: code complete, tested, rehearsed against the real review file. Nothing has been built on
ARC and no job has been submitted.** This is the piece
[`V3_THINK_IMPLEMENTATION.md`](V3_THINK_IMPLEMENTATION.md) §6 listed as "explicitly yours to build
later"; §8 of that file is the contract this satisfies.

| | |
|---|---|
| Written | 2026-10-08 |
| New files | 4 (2 code, 2 test) |
| Modified files | **1**, pointer-only — `CLAUDE.md` gains a row in its temporary-documents table and two filenames in its `data/` row. No claim changed, no behaviour touched. |
| Test suite | **1094 passed, 1 skipped** = 1095 collected (was 1021 — **+74**) |
| Rehearsed | all 264 surviving review rows converted, blocked, validated and written end to end |
| Independently reviewed | yes — a separate agent ran the code, probed every guard and error path, and verified determinism across `--only` invocations. 12 findings; **every substantiated one is fixed** (§2.5) |
| Built on ARC | **No.** `$VLM_DATA_ROOT` is empty locally and the MOCS pixels are on the cluster |
| Submitted | **No** |

Temporary, like its two companions: fold the decisions into `CLAUDE.md` and delete this once v3/v4
have run.

---

## 1. Current state — what existed, verified

### 1.1 The `violations_think` arm is real and committed

Everything `V3_THINK_IMPLEMENTATION.md` §2 claims was built exists, is tracked, and works. Verified
by reading it, not by trusting the doc:

| file | verified |
|---|---|
| `core/think_format.py` | 455 lines. `build_think_body` (`:182`) is the executable format definition; `parse_think_body` (`:252`) never raises; `think_row_problems` (`:371`) is the row validator. Committed in `a44603a`. |
| `configs/tasks/violations_think.yaml:38-39` | routes to `datasets/augmented_v3` + `datasets/grpo_pool_v3`; reward block byte-identical to `violations_only.yaml` |
| `data/preprocessor.py:307-331` | `_build_violations_think_target_json` validates the row, then delegates the JSON half to `_build_violations_only_target_json` verbatim |
| `scripts/validate_think_dataset.py` | 304 lines, exit 0/1/2, checks rows **and** test-split identity **and** train/val→test leakage |
| `scripts/submit_pipeline.py:256-264` | `--sft-dataset` / `--grpo-pool` exist, appended **positionally and only when given** (`:350`, `:375`) |
| `data/loader.py:209-277` | `resolve_grpo_pool_subdir` / `load_grpo_pool(subdir=)` — the ghost-variable fix, and it is **production-proven**: all six `oo`/`co` v1 GRPO manifests record `resolved_grpo_pool_subdir` |
| `scripts/hpc_sft.sh:42-63`, `scripts/hpc_grpo.sh:65-84` | optional 4th / 5th positional, passed as a bash **array** so an empty value expands to zero argv entries |

The three-arm framing (v2 untouched / v4 = data / v3 = block) is **still right**, and I did not
change it.

### 1.2 What was missing

Exactly one thing: a script that turns the verified review rows plus ConstructionSite into the two
directories. Nothing else. No shared file needed an edit — every override the two arms require was
already in place and already covered by tests.

### 1.3 The verified data, measured rather than quoted

`C:\Users\Nabeel\Downloads\review_results.json`, `schema: mocs_review/2`, `box_scale: xyxy_0_1`,
266 `dataset_rows` (266 accept / 718 reject / 1 unsure of 985 decided).

Measured directly, and every number in the prompt reproduced:

| | measured |
|---|---|
| rows | 266 — 199 MOCS-`val`, 67 MOCS-`test` (a MOCS split; nothing to do with ConstructionSite's) |
| images per rule | rule_1 51 · rule_2 100 · rule_3 67 · rule_4 114 |
| boxes per rule | 59 · 241 · 68 · 135 (503 total, **all inside [0,1]**, 0 degenerate) |
| rows tripping ≥1 of rules 2/3/4 | **260** |
| rows with **no** violation at all | 2 (`mocs_0019640`, `mocs_0021130`) — legitimate confirmed-safe rows |
| blank caption | 1 (`mocs_0022103`, `caption_ok: "n"`, carries rule_1 + rule_4) |
| empty assertion (no reason, no box) | 1 (`mocs_0023379`/rule_1 — the row's rule_2 is good, with 2 boxes) |
| reason-only violations (no box) | 2, both rule_4 |
| newlines / braces / fences in any caption or reason | **0 / 0 / 0** |
| duplicate `new_image_id` | 0. `mocs_image_id` **does** collide (2 pairs) — never key on it |
| multi-box rule_4 | **18 of 114** |
| rule_4 reasons saying "operating radius" | **107 of 113** (the dataset's convention is "operation radius") |

Images: 0 of 266 `image_path_original` paths resolve on Windows (they are ARC paths); **266 of 266
staged renders exist locally** at `results_index/mocs_annotation_combined/review/images/`, at exactly
the original pixel dimensions and with no boxes drawn on them.

---

## 2. What I built

Four new files, and **no modified code at all** — the strongest form of the additivity requirement,
achievable because every routing override the two arms need already existed. The only edit anywhere is
three pointer lines in `CLAUDE.md`, so that this file and the two scripts are discoverable.

| file | lines | what |
|---|---|---|
| `data/mocs_rows.py` | 464 | **Pure.** Review JSON → ConstructionSite-shaped row dicts + a conversion report. No HF, no PIL, no albumentations, no data root — so every drop/repair decision is testable on a laptop. |
| `data/build_v3_datasets.py` | 945 | The ARC-side orchestrator: load, combine, bake the block, drop, fan out to both outputs, write manifests. |
| `tests/test_data/test_mocs_rows.py` | 357 / 31 tests | One per documented decision. |
| `tests/test_data/test_build_v3_datasets.py` | 635 / 43 tests | End-to-end over a synthetic data root. |

### 2.1 `data/mocs_rows.py`

Three repairs, each a policy with a stated cost, all recorded by image id in the manifest:

**`provenance` is stripped from every violation object.** Not cosmetic: `datasets/processed`'s
violation struct is exactly `{bounding_box, reason}`, and an extra key changes the Arrow struct type,
so `concatenate_datasets` with the ConstructionSite rows would fail outright — or silently coerce.
The provenance is tallied into the manifest first.

**A row that cannot carry a clean, fully-grounded block is DROPPED, whole — one rule, no
exceptions.** Nothing is repaired in place. Two rows hit it in the current file: `mocs_0022103`
(blank caption) and `mocs_0023379` (rule_1 marked violated with no reason and no box). Leaving either
alone is not an option: `build_think_body` refuses to emit a line for a reason-less assertion, so the
row would fail the SFT job outright, and even for `violations_only` the shape costs twice —
`_is_substantive_violation` scores it a **miss** on a real violation while `_is_violation_present`
still counts it a **false alarm**.

**`mocs_0023379` is the expensive one, and it is worth being explicit about.** Its rule_1 carries
nothing (its own provenance says the model never proposed it and the reviewer drew nothing — it reads
as a mis-click), but its **rule_2 is genuinely verified, with two boxes**, and that is lost with the
row: one of the 153 rule_2 images this harvest exists to buy. The alternative was to null just the
rule_1 and keep the image. That is cheaper in rows and more expensive in everything else — it writes
an assertion the reviewer *made* back out as "not violated", a label no human here produced; and it
splits the policy in two, so "which rows are in, and why" stops being one sentence anyone can check.
§2.3-Q7 has the full argument and every shape the rule sorts.

**Multi-box `rule_4` violations are unioned to one box, by default — and as of 2026-10-08 this is a
NO-OP.** The reviewer re-opened the 18 offending rows in the review tool and redrew each as a single
group box, so the file now holds 111 boxes on 113 boxed rule_4 images (1.00/image, matching the test
key exactly) and `--rule4-box-policy union` and `keep` produce byte-identical output. The policy stays
in as a guard against a future review file regressing. Spot-checked visually: `mocs_0023313` encloses
exactly the two crouching workers and excludes the machine; `mocs_0021074` is large (48.9% of frame)
only because its two workers stand on opposite sides of the trench, which the one-box rule forces.
The case for the policy, verified against code and data rather than quoted from `REVIEW_RUBRIC.md`:

**1. The scoring is area-based, not count-based.** `rewards/reward_violation_grounding.py:51` calls
`compute_mask_union_iou(pred_boxes, gt_boxes)` — it rasterises the **union** of all predicted boxes
against the union of all GT boxes. So "one box or three" is not the question the reward asks; *"does
the covered region match GT's covered region"* is. That reframes the whole decision.

**2. ConstructionSite's rule_4 GT is single-box, measured.** Not from the rubric — from
`results_index/oo_co_v1_dump/datasets_stats/dataset_report.json`, a `dataset_report.py` artifact built
on ARC:

| rule | images | boxes | boxes/img | test split |
|---|---:|---:|---:|---|
| rule_1 | 1000 | 1662 | **1.66** | 1.28 |
| rule_2 | 84 | 146 | **1.74** | 1.52 |
| rule_3 | 172 | 199 | **1.16** | **1.00** |
| rule_4 | 70 | 71 | **1.01** | **1.00** (24 images, 24 boxes) |

In the **test split — the thing every number is scored against — rule_4 is 24 boxes on 24 images.
Not one multi-box rule_4 exists there.** CS train is 41 single-box images and 1 double.

**2b. "If one box covers the whole group, how do we know it was more than one person?" — from the
REASON, not from the geometry.** Read straight off the v2 test-split ground truth in
`results_index/v2_dump/.../predictions_with_eval.json`:

```
TEST GT rule_4: 24 images, boxes-per-image = {1: 24}       <- not one multi-box case
  4 of those 24 reasons name MULTIPLE people, and ALL 4 carry ONE box:
    0000120  1 box  "The two workers on the left are very close to the red excavator in operation."
    0002093  1 box  "The two workers on the right are close to the excavator and are in the blind spot."
    0004725  1 box  "The two people in the middle are close to the excavator and are in its blind spot."
    0000421  1 box  "At least three workers are very close to the excavator."
TEST GT rule_3: 63 images, boxes-per-image = {1: 63}; 6 plural reasons, all 1 box
    0000895  1 box  "The two sides of the excavation trench are not guarded."
```

So GT's division of labour is explicit: **the box says WHERE the hazard region is, the reason says WHO
and HOW MANY.** Nothing in the pipeline reads a person count off the geometry — `reward_violation_id`
scores presence, `reward_violation_grounding` scores a rasterised region, and the count lives in the
reason text that `reward_reasoning` and the LLM judge score. The `(n)` in a think block is a BOX count
by the prompt's own definition, not a person count. Unioning therefore discards nothing the pipeline
uses, and the per-person geometry survives in `review_results.json` and in the manifest's
before/after list, so a `keep` rebuild is always available.

**3. So the real argument is consistency of the training signal, not fidelity to a rubric.** 94 of the
113 MOCS rule_4 rows already carry a single box. `keep` therefore produces a **mixed** convention —
136 single-box against 19 multi-box across the combined set — which is arguably the worst of the three
options, because there is no consistent habit to learn. `union` makes it 154 single / 1 double,
matching both CS and the test key.

**4. And the gap is not cosmetic.** Mask-union IoU between the 18 rows' boxes as drawn and their
enclosing box: **mean 0.571, median 0.626, min 0.098**, with 13 of 18 below 0.8. On a component
weighted 0.317 that is a real difference, in both directions — which is exactly why it is a flag and
not a hardcoded choice.

**What this is NOT.** It is not a claim that the single box is *better annotation*. A GT group box is
drawn by a human and may be generous; `union_box` produces the *minimal* enclosing rectangle of tight
person boxes, so it approximates GT's convention rather than reproducing it. If you think GT's
single-box habit is itself an annotation shortcut, `--rule4-box-policy keep` is the coherent opposite
position — you would then be optimising against a key you believe is wrong, and measuring worse for
it. The manifest lists every before/after pair either way.

rule_1 and rule_2 are **untouched**: at 1.66 and 1.74 boxes per image, one box per person genuinely is
their convention.

What it deliberately does **not** do, each with the reason in the module docstring: no rescaling
(the `box_scale` header is **checked**, and a mismatch raises — that is the one failure nothing
downstream can see); no invented object annotations; no guessed metadata; no reason rewording; no
`thinking` column (one code path bakes it for both provenances, so they cannot drift).

### 2.2 `data/build_v3_datasets.py`

```
datasets/processed  (6308 / 701 / 3004, un-augmented)
        + provenance
        |
review_results.json → data/mocs_rows.py → 264 rows + image refs
        |
base_train = CS train 6308 + MOCS 264 = 6572        ← un-augmented
        |
        + bake `thinking` for every train/val row, DROP the unbuildable (guarded)
        |
        ├──────────────────────────────→ grpo_pool_v3   (from the UN-augmented base)
        |
  augment rules 2/3/4 ×2 ─────────────→ augmented_v3
```

Design points that are load-bearing rather than tidy:

- **One script, two outputs.** The combine is the expensive and error-prone part. If the SFT set and
  the pool were built separately they could disagree about which MOCS rows survived or which rule_4
  boxes were kept, and *nothing downstream would notice*. Running it once and fanning out makes that
  impossible instead of merely unlikely. `--only sft|pool` still rebuilds one half; the conversion is
  deterministic, so they stay consistent.
- **The pool branches off before augmentation**, for the reason `data/build_grpo_pool.py`'s own
  docstring gives: a pixel-jittered near-duplicate in the same rollout pool is a correlated reward
  group, not signal.
- **Features are taken from `ds["train"].features`, never declared.** Measured hazard: a column that
  is `None` in *every* row of a `Dataset.from_list` batch infers as `Value('null')`, not the struct.
  Reusing the source features also means the inner Arrow types (float64 vs float32, nullability)
  are whatever `datasets/processed` actually holds, which I cannot see from here.
- **`RULE_MULTIPLIERS` is imported, never touched.** `augment_sample` and
  `get_pixel_augmentation_pipeline` are reused verbatim, so the transforms, the `_aug{n}` id suffix
  and the carry-all-columns behaviour are identical to what produced `datasets/augmented`; only the
  counts differ, and they are a parameter. `tests/test_core/test_blocker_fixes.py:1183` regex-reads
  that constant out of the source and would fail on an edit; a new test asserts the same thing from
  the v3 side.
- **Protected output paths.** `datasets/{processed,augmented,grpo_pool,raw,raw_cleaned}` are refused
  as `--out-*`. MOCS rows in `datasets/augmented` would teach `unified` that excavators, rebar and
  hard hats are *absent* on thousands of images where they were merely never annotated.
- **Reproducible jitter.** v2's augmentation was unseeded. This seeds `random`, `numpy` and —
  best-effort, because albumentations ≥1.4.21 carries its own RNG — `pipeline.set_random_seed`. The
  manifest records which one took.

### 2.3 The §6 questions, answered

**Q1 — what the build does, column by column.** Inputs: `datasets/processed` (a `DatasetDict` of
train/val/test, 15 columns) and `review_results.json`. Outputs: `datasets/augmented_v3`
(`DatasetDict`, 17 columns) and `datasets/grpo_pool_v3` (a **flat** `Dataset`, 16 columns).

| column | ConstructionSite rows | MOCS rows |
|---|---|---|
| `image` | carried, HF `Image()` | the original JPEG bytes, unmodified, under the same `Image()` feature |
| `image_id` | carried | `new_image_id` (`mocs_0020121`), unique; **never** `mocs_image_id`, which collides across MOCS splits |
| `image_caption` | carried | the reviewer's final caption |
| `rule_1..4_violation` | carried, `[0,1]` | `{bounding_box, reason}`, `[0,1]`, `provenance` stripped |
| `excavator` / `rebar` / `worker_with_white_hard_hat` | carried | `[]` — see Q3 |
| `illumination` / `camera_distance` / `view` / `quality_of_info` | carried | `""` — see Q3 |
| `resolution` | carried | `width × height` (pixel **area**, matching how the notebook wrote it) |
| **`provenance`** *(new)* | `constructionsite` | `mocs` |
| **`thinking`** *(new)* | `build_think_body(caption, violations)` | same function, same code path |

`train` carries both provenances; `val` and `test` are ConstructionSite-only. The pool drops
`thinking` (GRPO has no target text — there is nowhere for it to go) and keeps `provenance`.

**Q2 — where the images come from, and where the build runs.** **On ARC.** `vlm_data_root/` is empty
locally, so the ConstructionSite base simply is not here. Three image sources are tried in
decreasing order of fidelity and the choice is recorded per image:

1. `image_path_original` — `$VLM_DATA_ROOT/datasets/filtered/instances_{val,test}/<file_name>`, the
   untouched MOCS JPEGs. This is what will hit on ARC.
2. `--mocs-images-root <dir>/instances_<source>/<file_name>` — the same files if the root moved.
3. `--mocs-review-images <dir>/<review_image>` — the staged renders. Same pixel dimensions as the
   originals for all 266 (nothing exceeded the 1600 px render cap) and carrying no burned-in boxes,
   but a q88 JPEG re-encode, hence the fallback.

**What has to be staged first:** `review_results.json` is on your Windows box and nowhere else.
`scp` it to ARC before anything (§3.1).

**Q3 — the columns MOCS lacks.** `excavator`/`rebar`/`worker_with_white_hard_hat` are written `[]`.
That is the only honest option and it is a **known, labelled** lie of omission: `[]` means
"unannotated", not "absent". It is safe here because neither `violations_only` nor `violations_think`
reads those columns at all — not in the target builder, not in the GT builder, not in the sampler.
It would be a lie for `unified`/`object_only`, which is exactly why MOCS rows never reach
`datasets/augmented` and why the `provenance` column exists. I did **not** derive `excavator` from
`"Excavator" in mocs_categories`: the column holds *boxes*, not a boolean, and `mocs_categories` is
empty on all 67 MOCS-`test` rows because that split ships no annotations — so it would be both
geometry-free and systematically wrong for a quarter of the rows.
The four metadata strings are `""`. Nothing trains on them; their only consumer is
`evaluation/error_analyzer.py`, which strata results by them, and `""` honestly reads as
"unlabelled". A guessed value would be a fabricated stratum. `resolution` is real (`w × h`).

**Q4 — implementing the 2× augmentation, and what assumes 8198/512.**
`--rule-copies 4=2,2=2,3=2` (the default) means **two rows in total** per rare image — one extra
copy. That is one more than `RULE_MULTIPLIERS`' convention, which counts *extra* copies, and I chose
the "total" spelling deliberately: the decision is written as "2×", and "appears twice" is the
reading that matches its own arithmetic (+452 on 452 rare images). `{4:2, 2:2, 3:2}` read as *extra*
copies would give +904 and 466 steps. The manifest records both forms, so there is nothing to
misread later.

**Nothing in executable code assumes 8198 or 512.** SFT steps come from `len(train_dataset)` ×
`num_train_epochs` at runtime, and the formula `rows // 32 × 2` was confirmed exactly against a real
run (`oo-sft-2b-v1/training_state.json`: `global_step: 394` from 6308 rows). Every occurrence of the
old numbers is a comment or a doc — listed in §5.

What does change, and is worth knowing before you read a log:

- **438 steps, not 512** → `persistent-checkpoint-{100,200,300,400}`, **four** per SFT variant, not
  five. The `PersistentCheckpointCallback` fires at `N % 100 == 0`.
- **17 eval points**, not 20 (`eval_steps: 25`).
- `configs/sft.yaml:2`'s and `:37`'s comments become wrong for these runs. I did not edit them
  because they are correct for `datasets/augmented`, which `unified` still uses.

**Is 438 acceptable, or should epochs rise?** **Acceptable — keep `num_train_epochs: 2`, do not
raise it.** Three reasons, in order of weight:

1. Holding the *recipe* fixed and letting the data set the step count is what makes v4 − v2 a
   one-variable comparison at the level that matters. Raising epochs to recover 512 steps would mean
   v4 sees each image 2.33 times against v2's 2.0 — that is also a change, just a less visible one.
   You cannot change the dataset and hold both epochs and steps; epochs is the one to hold.
2. The budget was already past its useful end. v2's own evidence (invariant 7, `README_v2.md`) is
   that `eval_loss` bottoms at step **125 of 512** at 8B and then drifts +11.2%; `README_v2.md` §13
   P1-5 proposes *cutting* 8B to 256 steps. 438 moves in the direction the measurement already
   points.
3. Information per step goes **up**, not down: 6,573 unique images against v2's 6,308, with far
   fewer near-duplicate rows (450 extra copies against 1,890).

**Q5 — `grpo_pool_v3`.** Built by the same script, from the combined **un-augmented** base, with
`data/build_grpo_pool.py`'s predicates imported rather than restated: the whole val split + every
train violation (ConstructionSite *and* MOCS) + enough random train-safe rows to reach 50/50,
`seed=42`, shuffled. **Yes, 50/50 holds** — it is 50/50 by construction, because the safe top-up is
computed as `total_violations − val_safe` and there are 5,530 safe train rows to draw 514 from. The
contract's "admit a MOCS row only if it carries a reason" needs no filter here: a reason-less,
reason-less assertion already took its row out upstream, and a box-without-reason violation does not exist in
the file (measured: 0).

**Q6 — train only, or train + val?** **Train only.** The 266 rows all go to `train`; `val` and `test`
stay exactly as `datasets/processed` has them. Reasons: it keeps `val` byte-identical to
`datasets/augmented`'s, so `eval_loss` curves remain comparable across v2 / v4 / v3 and the *only*
changed split is the one that trains; the MOCS rows still reach GRPO regardless, because the pool
takes every train violation; and a 90/10 split would have put ~27 rows in val — too few to improve
any estimate, enough to break the comparison. Pinned by
`test_val_split_is_unchanged_so_eval_loss_stays_comparable`.

**Q7 — the empty assertion and the reason-less rows. ONE rule, no exceptions: if a row cannot
produce a clean, fully-grounded think block, the whole row is dropped.** Nothing is ever repaired in
place. Five shapes, and the rule sorts them:

| case | count | decision | why |
|---|---|---|---|
| **blank caption** (`mocs_0022103`) | 1 | **drop the row** | no honest repair: `caption_ok: "n"` means the reviewer rejected the model's caption too, and hand-writing one would put authored text on the first line of every think block — the one property that makes the arm defensible |
| rule asserted with **no reason and no box** (`mocs_0023379`/rule_1) | 1 | **drop the row** | `build_think_body` cannot write a line for it, so the row could not train either arm. **This row's verified rule_2 (2 boxes) is lost with it** — one of the 153 rule_2 images. That cost is the decision, not an oversight |
| **box(es), no reason** | 0 today | **drop the row** | same rule: a box-only assertion has no sentence to put in the block either |
| **a violation value that is not an object** (a bare `true`) | 0 today | **drop the row** | reading it as `null` would not drop a label, it would **flip** it to "not violated" — the same failure class as the `structural_repair` list-drop bug that cost one run 182 true positives. Dropping cannot invert anything |
| reason, **no box** (2 rule_4 rows) | 2 | **keep** | not a bad row at all: legal in the schema *and* in the block format (`rule_N: <reason> -> yes`, count omitted), and `_is_substantive_violation` counts it. It scores 0 on `reward_violation_grounding`, so it is counted in the manifest rather than waved through |

**Why drop, rather than null the one offending rule and keep the image.** Nulling is cheaper in rows
and more expensive in everything else. It writes an assertion the reviewer *made* back out as "not
violated" — a label no human in this pipeline ever produced. Where boxes are present it also discards
verified geometry while leaving the image in the dataset asserting the opposite. And it splits the
policy in two, so "which rows are in, and why" stops being one sentence anyone can check. Every drop
is listed by `image_id` and `cause` in the manifest, with a `dropped_by_cause` tally, and
`rows_in − rows_out == len(dropped)` holds exactly — pinned by a test.

**The same rule covers ConstructionSite, and one of its rows hits it.** The train split holds exactly
one rule_1 violation with a box and no reason (§3.0), so `augmented_v3`'s CS train side is v2's minus
that one row. Identical treatment on both provenances — which also matters because `augmented_v3`
serves **both** arms: a row kept for v4 and dropped for v3 would turn "does the block help?" into a
two-variable question. `--max-drop-rate` (1% corpus-wide) and a second arm at 2% of the harvest alone
**abort the build** if this ever happens at scale, because a large count is a ground-truth problem to
look at, not a dataset to shrink quietly.

### 2.4 What the independent review found, and what changed

A separate agent reviewed both files adversarially — running the code and probing the guards, not
reading them. It cleared the three categories I was most worried about: **index alignment** (every
view→dataset indexing path and the `mocs_dicts`/`image_values` zip), **pool composition** (line-for-line
equivalent to `data/build_grpo_pool.py::main`), and **determinism** (two runs produced byte-identical
augmented *pixels* and identical pool membership; `--only sft` then `--only pool` reproduced
`--only both` exactly).

The damage was concentrated in the guards. Every substantiated finding is fixed:

| # | finding | fix |
|---|---|---|
| 1 | **The protected-path guard was defeated by any non-literal spelling.** `--out-sft-subdir ./datasets/augmented` returned 0 and overwrote `datasets/augmented` with MOCS-contaminated rows. So did `datasets//augmented`, `datasets/augmented/.` and `datasets/../datasets/augmented`. The highest-value guard in the file, and it only caught two spellings | `_guard_outputs` now compares **resolved** filesystem paths. Seven spellings are in the parametrized test |
| 2 | Zero surviving MOCS rows crashed in `Dataset.from_list([], features=…)` with an Arrow "Keys mismatch" that names the *ConstructionSite* dataset, not the review file | explicit `SystemExit` naming the review file |
| 3 | **`--max-drop-rate` could not see a MOCS-specific failure.** 1% of ~7,274 rows is 72 rows of slack against a 264-row harvest — 27% of the thing the build exists to acquire could vanish into scrolling warnings | a second arm at **2% of the harvest alone**, plus the drop counts split by provenance in the log line and the manifest |
| 4 | **A violation with a box but no reason** was tracked nowhere: it converted cleanly, then made `build_think_body` raise, so the whole row was dropped later by an unrelated code path with no named cause | now an explicit, named drop cause alongside the empty assertion. Measured count in the real file: 0 |
| 5 | `report.dropped` conflated whole-row drops with single-value rejections, so `rows_in − rows_out ≠ dropped_count`; and a non-dict violation value (a bare `true`) was **inverted** to "not violated" — the same failure class as the `structural_repair` list-drop bug | `dropped` now holds rows only, each with a `cause`, plus a `dropped_by_cause` tally; a non-object value drops its row instead of being read as "not violated", so nothing can invert. A test pins `rows_in − rows_out == len(dropped)` |
| 6 | A partial `--rule-copies 4=3` silently set rules 2 and 3 to **no augmentation** | unnamed rules keep their default; disabling one now has to be said (`2=1`) |
| 7 | The manifest's 50/50 was arithmetic over the same locals that built the pool, and the test asserting it was a **tautology** — a wrong `select()` would have printed a perfect 50/50 | `build_pool` now **counts** the written pool and aborts on a mismatch; the safe-heavy branch `build_grpo_pool.py` never warned about is covered; the test counts rows instead of comparing the manifest to itself |
| 8 | `boxes_rejected` compared *after* `normalize_boxes`, which silently discards malformed elements — so the one thing it exists to surface stayed invisible | compares against what the reviewer submitted |
| 9 | Nothing checked that two `--only` invocations used the same review file, policy and seed | `_warn_on_sibling_drift` reads the sibling's manifest and warns loudly |
| 10 | Four manifest-vs-dataset disagreements a later reader would file as bugs: `by_provenance` was pre-augmentation while the written split is post-; `rows_out` was pre-block-bake; `provenance_tallies` described the review file; stdout printed truncated, invalid JSON | `train_by_provenance_after`, `mocs_rows_in_dataset`, `provenance_tallies_over_review_file`, and a structured summary instead of a slice |
| 11 | `--smoke 0` was falsy, so it wrote the **real** output directories; `--smoke -5` produced empty splits | `is not None` plus a `< 1` rejection |
| 12 | the empty-plan path returned the train split **unshuffled** while every other path shuffled; albumentations was seeded *after* the pipeline was constructed; `--out-sft-subdir X --out-pool-subdir X` would clobber; a bad review file raised a traceback instead of a message | all four fixed |

Two more guards came out of thinking about #3: **`val` losing even one row now aborts the build**
(`scripts/validate_think_dataset.py` enforces only the *test* id set, so a shrunken val — which would
make `eval_loss` non-comparable with v2 — was flagged nowhere), and the three drop guards fire
**most-specific first** so the message names the real problem.

Test gaps it named are closed: the protected-path spellings, the tautological pool assertion, the
box-with-no-reason shape, the sub-threshold "drop quietly and record it" path that will actually run in
production, partial `--rule-copies`, zero MOCS rows, `--smoke 0`, the safe-heavy pool branch, and
`--only sft` + `--only pool` reproducing `--only both`. **+20 tests over the first pass.**

One note I acted on in the run plan rather than the code: the reviewer could only exercise this on
`datasets==5.0.1` and ARC pins `4.3.0`. The module never constructs a `Features` literal — it copies
them off the on-disk dataset, which is immune to the 4.x `Sequence` → 5.x `List` rename — but
`add_column` on a `select()`ed view and `Dataset.from_list(..., features=…)` with a **PIL object** in an
`Image()` column are the two calls worth proving on ARC first. `--smoke 400` exercises both, which is
why it is step 2 of §3.3 rather than optional.

### 2.5 Verification performed

| check | result |
|---|---|
| Full suite, locally | **1094 passed, 1 skipped** (1095 collected, +74) |
| All 264 surviving **real** review rows → `build_think_body` → `think_row_problems` | **0 failures**, under both `--rule4-box-policy` values |
| Full `main()` over the real review file + the real 266 staged renders + a synthetic ConstructionSite base | wrote both datasets, no errors |
| `scripts/validate_think_dataset.py --strict` on that output | **ALL CHECKS PASSED** — 0 invalid rows, test split matches, no leakage |
| `scripts/validate_rewards.py --task violations_think --probe --pool-stats` on that pool | **PASS**, `c=0.300`, `p*=0.298` — identical to v2 |
| `build_target_json(row, "violations_think")` over every output row | 0 failures; target **ends with** the `violations_only` target verbatim; GT dicts identical between the two tasks |
| boxes in the written dataset | all inside `[0,1]`; `[0,1000]` only in the SFT target string, as intended |
| the real downstream plumbing over that output: `build_sft_dataset` (both tasks), `build_oversampled_indices`, `build_rare_mask_for_task`, `build_grpo_dataset_for_task` | all four work; the `vt` SFT target starts `<think>`, the `vo` one starts the fence; the GRPO prompt dataset's image column is `image`, **singular** (invariant 1) |
| `scripts/dataset_report.py --roots augmented_v3 grpo_pool_v3` | runs; "0 out of [0,1], 0 degenerate" |
| Submitter dry-run, both arms × 3 tiers | 24 jobs, correct positionals, **no collision** with any `vo-*-v2` name |
| `data/augment_rare_classes.py::RULE_MULTIPLIERS` | still `{4: 16, 2: 12, 3: 6}` — asserted both from the source text and, after a real build, from the imported dict |
| Determinism: two full runs; `--only sft` + `--only pool` vs `--only both` | byte-identical augmented pixels, identical row order, identical pool membership |
| Every guard, driven to its failure: 7 protected-path spellings, same-output, zero MOCS rows, bad `box_scale`, missing images, val shrink, harvest drop >2%, corpus drop >1%, `--smoke 0` | all abort before anything is written |
| Everything above **re-run after the review fixes, and again after the drop-both-rows policy change** | unchanged results; 264 rows, `dropped_by_cause {'blank_caption': 1, 'asserted with no reason': 1}`, `rows_in − rows_out == dropped_count`, pool measured 50/50 |

---

## 3. The run plan

### 3.0 Counts and step budgets

**These are computed, not guessed.** `$VLM_DATA_ROOT` is empty locally, but
`results_index/oo_co_v1_dump/datasets_stats/dataset_report.json` is a real ARC artifact carrying
ConstructionSite's exact per-split rule **co-occurrence** table, and the MOCS side was measured off the
review file directly. Everything below follows arithmetically from those two.

| | v2 (on disk) | v4 / v3 |
|---|---|---|
| SFT train rows | 8,198 | **7,021** (6,307 CS + 264 MOCS + 450 augmented copies) |
| rare images augmented | 192 | **450** (192 CS + 258 MOCS) |
| SFT steps | 512 | **438** |
| `persistent-checkpoint-N` | 100…500 | 100…400 |
| val / test rows | 701 / 3,004 | 701 / 3,004 — **unchanged** |
| GRPO pool rows | 1,732 | **2,254** (1,127 violation / 1,127 safe, exactly 50/50) |
| GRPO steps | 108 | **140** |
| pool rule_1 : rule_4 | 677 : 46 = **14.7 : 1** | 726 : 159 = **4.6 : 1** |
| pool P(rule_1) | 0.391 | 0.322 |

ConstructionSite train's co-occurrence table is
`{rule_1: 588, rule_3: 92, rule_2: 47, rule_4: 31, rule_1+rule_4: 10, rule_1+rule_2: 6,
rule_1+rule_3: 5, rule_3+rule_4: 1}` — 780 violation rows, 5,528 safe, per-rule 609/53/98/42, and
**exactly 192** images tripping at least one of rules 2/3/4. Cross-check: under the existing
precedence those 192 split 42/53/97 across rule_4/rule_2/rule_3, and `42·16 + 53·12 + 97·6 = 1890`
— precisely v2's realised `6308 → 8198`. The same table gives val's 86 violation / 615 safe and
reproduces today's pool (677/59/109/46, 866+866) to the row. The build manifest will print all of
it again from the live data; if any number differs, trust the manifest and tell me.

**Three rows are expected to drop, and they are all identified in advance.** Two MOCS rows
(`mocs_0022103`, blank caption; `mocs_0023379`, rule_1 asserted with nothing) — measured directly, and
reproduced by the build. One ConstructionSite **train** row: that split has **0 blank captions** but
**one rule_1 violation with no reason** (`reason_words.n = 608` against `images_with = 609`, with
`contentless_assertions: 0`, so it carries a box but no sentence), per
`results_index/oo_co_v1_dump/datasets_stats/dataset_report.json`. `val` and `test` are clean on both
counts, which is what makes the "val stays byte-identical" guarantee safe.

That is **3 drops of 7,272 checked rows = 0.041%**, far below the 1% corpus threshold, and
**2 of 264 = 0.76%** of the harvest, below its own 2% arm. The step counts are **438 / 140**
regardless — all three candidates land inside the same floor-division bucket, so the budget does not
move whether they drop or not.

**That 4.6 : 1 is `README_v2.md` §13 P1-3 landing as a side effect** — the pool imbalance is fixed by
*addition*, with no rule_1 row discarded.

### 3.1 Stage the review file (do this first — nothing works without it)

```bash
# from your Windows box, in MobaXterm's local shell or a terminal
scp "C:/Users/Nabeel/Downloads/review_results.json" \
    nabeel.shan@arc.ucalgary.ca:~/vlm-finetuning-project1/datasets/review_results.json
```

Also make sure the repo on ARC is at the commit that contains `data/build_v3_datasets.py` and
`data/mocs_rows.py` (`git pull` after you commit them).

### 3.2 ARC session preamble (every session)

```bash
cd $HOME/vlm-safety-reasoning
module purge && module load gcc/13.3.0 python/3.12.5
source $HOME/envs/vlm_grpo/bin/activate
export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH" \
       VLM_DATA_ROOT="$HOME/vlm-finetuning-project1" \
       HF_HOME="$HOME/scratch/hf_cache"
```

### 3.3 Build the datasets — CPU only, no GPU, no SLURM

```bash
# 1. dry run: every count, nothing written. READ THE OUTPUT.
python data/build_v3_datasets.py \
    --review $VLM_DATA_ROOT/datasets/review_results.json --dry-run

# 2. a rehearsal that cannot touch the real outputs. NOT optional: ARC pins
#    datasets==4.3.0 and nothing here has run against that version.
python data/build_v3_datasets.py \
    --review $VLM_DATA_ROOT/datasets/review_results.json --smoke 400
#    -> datasets/augmented_v3_smoke, datasets/grpo_pool_v3_smoke
python scripts/validate_think_dataset.py --subdir datasets/augmented_v3_smoke \r
    --reference-subdir ''        # '' because a smoke test split is not the real 3,004

# 3. the real build (~20-40 min; it reads ~10k images and writes ~12k)
python data/build_v3_datasets.py \
    --review $VLM_DATA_ROOT/datasets/review_results.json 2>&1 | tee ~/build_v3.log
```

`--dry-run` is a full rehearsal, not an arithmetic preview: it converts, loads every MOCS image, bakes
every block and runs the real pixel augmentation — it just does not `save_to_disk`. So it also proves
albumentations works and the RAM holds before you commit to writing ~12k images. Expect a few minutes.
`--smoke` is the fast path.

**Check in the dry-run output before you let it write:**

| | expect |
|---|---|
| `MOCS rows : 264 of 266 accepted (2 dropped in conversion)` | both named: `mocs_0022103` `[blank_caption]`, `mocs_0023379` `[asserted with no reason]` |
| `drops by cause : {'blank_caption': 1, 'asserted with no reason': 1}` | anything else here is news |
| `unioned rule_4 boxes on 0 row(s)` | **0** — the 18 were hand-corrected in the review tool on 2026-10-08, so the union has nothing left to do. A nonzero number means an older review file got staged |
| `MOCS images from : {'image_path_original': 266}` | if it says `review_render`, the originals were not found — fine, but know it |
| `think blocks : … 1 row(s) dropped (0.014%) = 1 constructionsite + 0 mocs` | exactly one, the ConstructionSite **train** rule_1 with a box and no reason (predicted in §3.0). **Any MOCS drop here is news** — the two bad MOCS rows were already removed above, so this stage should find none |
| `combined train : 6572` | 6308 + 264 |
| `augmented train : 7021 … -> 438 SFT steps` | |
| `GRPO pool : 2254 rows (50.0% violation / 50.0% safe) -> 140 steps` | |

If `rule_4` boxes should stay as the reviewer drew them, add `--rule4-box-policy keep` — and then
say so in the paper, because the 18 rows are a departure from the dataset's own convention in the
other direction.

### 3.4 Pre-flight — every gate, before a GPU-hour

```bash
# the datasets exist and have the right shape
python -c "from datasets import load_from_disk as L; d=L('$VLM_DATA_ROOT/datasets/augmented_v3'); print({k:(len(v),v.column_names) for k,v in d.items()})"
python -c "from datasets import load_from_disk as L; p=L('$VLM_DATA_ROOT/datasets/grpo_pool_v3'); print(len(p), p.column_names)"

# THE gate for this arm: rows, test-split identity, train/val->test leakage.
# Covers BOTH arms -- it checks the dataset, not the task.
python scripts/validate_think_dataset.py --subdir datasets/augmented_v3 --strict

# reward surface + token census, per arm. NOTE the --grpo-pool-subdir on the v4 line:
# without it, --pool-stats silently measures violations_only's OWN (v2) pool.
python scripts/validate_rewards.py --task violations_think --probe --census --pool-stats
python scripts/validate_rewards.py --task violations_only  --probe --pool-stats \
    --grpo-pool-subdir datasets/grpo_pool_v3
python scripts/validate_rewards.py --census --task violations_only \
    --sft-dataset-subdir datasets/augmented_v3

# reward assembly + prompt length
python scripts/preflight_grpo.py --tier 2b --task violations_think
python scripts/preflight_grpo.py --tier 2b --task violations_only

# inventory, and the suite -- NEVER the full suite on ARC
python scripts/dataset_report.py --roots processed augmented_v3 grpo_pool_v3
pytest tests/ -k "not submitter_can_override_gres"

# caches (compute nodes have no internet)
ls ~/scratch/hf_cache/hub | grep -i -E "llama|qwen"
ls ~/scratch/st_cache 2>/dev/null

git status --short && git log --oneline -1
```

Pass criteria: validator **exit 0**; `--probe` **PASS** with `p* = 0.298`; `--census` reporting no
truncation; `preflight_grpo` clean.

### 3.5 Submit — arm v4 first, then v3

**24 jobs total, 12 per arm, 4 per tier** (`baseline` independent ‖ `sft → merge → grpo` chained by
`afterok`).

```bash
# ---- v4: v2's exact recipe on the new data, no think block ----------------
python scripts/submit_pipeline.py --task violations_only --version v4 \
    --tiers 2b 4b 8b \
    --sft-dataset datasets/augmented_v3 \
    --grpo-pool   datasets/grpo_pool_v3 \
    --time-baseline 03:00:00 --time-sft 05:00:00 \
    --time-merge 01:00:00 --time-grpo 20:00:00 \
    2>&1 | tee ~/submit_vo_v4.log

# ---- v3: the think arm. NO --sft-dataset / --grpo-pool: ------------------
#      configs/tasks/violations_think.yaml already routes both.
python scripts/submit_pipeline.py --task violations_think --version v3 \
    --tiers 2b 4b 8b \
    --time-baseline 04:00:00 --time-sft 06:00:00 \
    --time-merge 01:00:00 --time-grpo 24:00:00 \
    2>&1 | tee ~/submit_vt_v3.log
#      (scripts/submit_vt_pipeline.py --tiers 2b 4b 8b --version v3 is the same thing)
```

**Then immediately hold the 8B v3 GRPO job** until 4B v3 GRPO has shown its real per-step rate — see
the walltime risk below:

```bash
squeue -u $USER -o "%.10i %.24j %.8T %R" | grep vlm-grpo-vt-8b-v3
scontrol hold <that jobid>
# ... after 4b v3 GRPO has logged ~20 steps and you have a seconds/step figure:
scontrol release <that jobid>
```

Nothing else needs holding. The walltime overrides above are not required — the defaults
(12/12/1.5/24 h) are legal and sized for `unified` — but a smaller reservation backfills sooner. Give
SFT and merge margin: they are `afterok` chain-killers and **do not auto-resume**. GRPO can run
tighter because nothing depends on it and it resumes from `save_steps: 20`.

**Do not pass `--version v2` for anything.** `run_inference_batched` opens `predictions.jsonl` in
`"w"` with no resume, so a v2 re-run truncates the file `README_v2.md` was built from.

### 3.6 Walltimes, and the one that is genuinely at risk

Decomposed from the measured v2 jobs (each v2 figure is the *whole* job; the baseline job is
inference over 3,004 images + repair + eval + judge with no training, so subtracting it isolates
training):

| tier | v2 GRPO job | − baseline | = training | s/step @108 | v4 @140 steps | v4 job |
|---|---|---|---|---|---|---|
| 2b | 5:23:31 | 1:19:02 | 4:04 | 136 | 5:17 | **~6:40** |
| 4b | 9:21:57 | 1:18:36 | 8:03 | 268 | 10:26 | **~11:45** |
| 8b | 11:26:53 | 0:47:37 | 10:39 | 355 | 13:48 | **~14:40** |

v4 SFT scales the other way (438/512 = 0.855): 8B SFT ≈ **1:51** against its 2:01:50.

**Total cost.** Summing the projected tier totals: v4 ≈ **41 H100-hours** (2b ~9 h, 4b ~15 h,
8b ~17 h), v3 ≈ **60 H100-hours** (2b ~13 h, 4b ~21 h, 8b ~26 h) — about **100 H100-hours** for
both grids, against v2's measured 35 for one. Wall-clock depends entirely on the queue; 24 jobs
against `gpu-h100`'s 10 cards will serialise.

**v3 is the risk, and only at 8B.** The think block roughly doubles-to-triples the generated tokens
per rollout — a safe image's completion goes from ~40 tokens of JSON to ~130 with the block — and
GRPO generates 8 rollouts per prompt, so generation time is the dominant term. At a conservative
1.5–1.8× on the training portion, 8B v3 GRPO projects to **21:30 – 25:40**, i.e. straddling the 24 h
partition wall.

This is survivable, not fatal, and here is exactly why: `models/grpo_trainer.py:261-266` calls
`get_last_checkpoint(output_dir)` and resumes, with `save_steps: 20`, so a wall kill costs at most
~20 steps. The response is to **re-submit the identical GRPO job** — same variant name, same command
— which continues rather than restarts. If it is killed *after* training finished but during
inference, the resume completes instantly and the job proceeds to inference.

```bash
# if vlm-grpo-vt-8b-v3 is killed by the wall:
sbatch --mem=250G --time=24:00:00 --job-name=vlm-grpo-vt-8b-v3 \
  --output=$VLM_DATA_ROOT/logs/grpo_vt_8b_v3_%j.out \
  --error=$VLM_DATA_ROOT/logs/grpo_vt_8b_v3_%j.err \
  scripts/hpc_grpo.sh violations_think 8b vt-grpo-8b-v3 merged-vt-sft-8b-v3
```

I do **not** know the real multiplier — it depends on how fast Qwen3-VL generates 90 extra tokens
against a 1,700-token prompt on an H100, which only the 2B and 4B v3 runs will tell you. That is why
the 8B v3 GRPO job is the one to hold.

### 3.7 What can run concurrently, and what must not

**Can:** both arms, all six tiers, all 24 jobs, at once. Isolation is structural — every writable
path is namespaced by `task_prefix` and `--version`, and
`tests/test_core/test_name_isolation.py` enumerates all of them and asserts no duplicates. Verified
by dry-run: no `vt-*-v3` or `vo-*-v4` path touches a `vo-*-v2` one. Everything they share
(`datasets/*`, the HF cache) is read-only during training.

**Must not:**

- Anything at `--version v2`, ever (§3.5).
- Building the datasets while a v3/v4 job is reading them. Finish §3.3 before §3.5.
- The full test suite on ARC — `test_submitter_can_override_gres_for_every_stage` shells out to a
  real `sbatch` and submitted 8 unwanted GPU jobs on 2026-09-14. Always
  `pytest tests/ -k "not submitter_can_override_gres"`.
- Re-running the build over an existing `augmented_v3` while something trains from it.

### 3.8 What to check after each stage

| stage | it worked if | failure looks like |
|---|---|---|
| **build** | both `build_manifest.json` files exist; `pool_violation == pool_safe`; `train_rows_after == 7021`; `dropped_count == 2` | a `SystemExit` naming the guard it hit — all four guards (protected path, missing images, drop rate, bad box scale) abort before writing |
| **validate** | validator exit 0; `--probe` PASS at `p*=0.298` | exit 1 lists every offending row id, and the full list is in `datasets/stats/think_validation_augmented_v3.json` |
| **baseline** | `results/inference/{vo-baseline-T-v4,vt-baseline-T-v3}/evaluation_results/metrics.json`; `structural_total_samples_count == 3004` | `vt` baselines will look *bad* — an un-fine-tuned model asked for a `<think>` block often never reaches the JSON. That is the measurement, not a bug. Read `think_block_present_rate`. |
| **sft** | log says `SFT input: 7021 train / 701 val`, `sft_dataset_subdir: datasets/augmented_v3` in `run_config.json`, `global_step: 438` in `training_state.json`, `final/` written | a `ValueError: Bad think block for image_id=…` means the validator was skipped — it cannot happen if §3.4 passed |
| **merge** | `checkpoints/qwen3vl-T/merged-{vo,vt}-sft-T-v{4,3}/config.json` exists | `hpc_merge_sft.sh:112` refuses without `final/adapter_config.json`; the whole `afterok` chain then dies |
| **grpo** | `run_manifest.json` carries `resolved_grpo_pool_subdir: datasets/grpo_pool_v3` and `grpo_pool_rows: 2254`; `reward/mean` climbs ~0.06 over the first 40–70 steps; `kl` 0.003–0.02 | `resolved_grpo_pool_subdir: datasets/grpo_pool` means the override did not arrive — **stop and check the positional**. `grep -c "Error in reward function" …out` must be 0 (`.out`, not `.err`) |
| **eval** | 9 `metrics.json` per arm; `llm_judge_status.json` `status: ok`, `unparsed 0`, matching `rubric_sha256` | a fail-soft judge writes `status: failed` and omits its keys — the job still exits 0 |

Since W&B is offline, read curves from the SLURM `.out` (GRPO logs every 5 steps, SFT every step) or
`wandb sync $HOME/scratch/wandb/offline-run-*` afterwards.

**If a tier has to be redone**, follow `OPERATIONS.md` §8 with `P=vo; V=v4` or `P=vt; V=v3` and resubmit
only that tier (`--tiers 4b`), never the whole version — the healthy tiers are still running and their
names would collide. `OPERATIONS.md` §7 covers `NODE_FAIL`, `DependencyNeverSatisfied` and the wall kill.

### 3.9 Analysis

```bash
python -m experiments.build_results_index --out results_index/index.json
python -m experiments.compare_all --index results_index/index.json --out results_index/
```

Read in this order: `violation_pred_positive_rate` against the 0.1368 GT rate **before any recall
number**; the support-counts table; `significance.csv`; `repair_stats.csv::status:valid_raw:pct`;
then the `think_*` family, where `think_verdict_json_agreement_rate` is the interesting one.

**One real gap:** `compare_all.py::print_significance` only pairs runs inside one
`(task, tier, version)`. **v4 − v2 and v3 − v4 are both cross-version/cross-task and will not be
paired for you.** All three arms carry `violation_per_image_outcomes_b64`, so no reconstruction is
needed — but the pairing code does not exist. `V3_THINK_IMPLEMENTATION.md` §6 already flags this as
"the one real gap"; it is worth writing the day the first v3 run lands.

---

## 4. Risk register

Ranked by (probability × damage × how long it would stay invisible).

| # | risk | why it is bad | the check that catches it |
|---|---|---|---|
| 1 | **A contradictory `thinking` row** — block verdict disagreeing with the row's own label | Trains the model to invert evidence. No reward, metric or repair stage can see it; the run looks entirely normal | The build bakes and validates in one pass (`build_think_body` → `think_row_problems`), then `scripts/validate_think_dataset.py --strict`. Rehearsed on all 264 real rows: 0 problems |
| 2 | **The GRPO override does not arrive** and v3/v4 train on the *v2* pool | Both arms become v2 with extra steps, and the manifest would once have lied about it | `run_manifest.json::resolved_grpo_pool_subdir` — check it on every GRPO job. The fix is committed and production-proven on the oo/co runs |
| 3 | **8B v3 GRPO overruns the 24 h wall** | Looks like a crash; costs a day | §3.6. Hold the job, measure 4B first, re-submit identically to resume from `save_steps: 20` |
| 4 | **`--pool-stats` measures the wrong pool** for the v4 arm | `p*` is validated against a pool that is not the one training, and the operating point moves silently | Always pass `--grpo-pool-subdir datasets/grpo_pool_v3` with `--task violations_only` (§3.4) |
| 5 | **MOCS rows leak into `val` or `test`** | Breaks comparability with v2 *and* inflates the numbers | Structural (the build never puts them there) + `test_no_mocs_row_reaches_val_or_test` + validator checks 8–10 |
| 6 | **Someone points `unified` or `object_only` at `augmented_v3`** | 264 rows teach it that excavators/rebar/hard hats are absent where they were never annotated | `provenance` column; the protected-path guard; and say it out loud: **`augmented_v3` and `grpo_pool_v3` are for the two violation tasks only.** `PLAN_V3_THINK.md` §4.2's claim that the pool "would serve `unified`, `object_only` or `caption_only` equally well" is **wrong** for this generation of it |
| 7 | **Object-class statistics read off `augmented_v3`** | `dataset_report.py`'s break-even IoU table will be diluted by 264 rows of `[]` and read low | Same as #6. Read object statistics from `datasets/processed` |
| 8 | **Boxes rescaled twice** | Every box collapses to a point; every IoU silently zero | `load_review_file` **refuses** a `box_scale` other than `xyxy_0_1`; `test_boxes_are_carried_through_unscaled`; `dataset_report`'s box-hygiene line |
| 9 | **A `{` or a fence inside a block** diverts `structural_repair`'s brace fallback onto the block instead of the JSON | Corrupts parsing at evaluation | `_reject_unusable_text` at build time, `think_row_problems` check 7, and measured: 0 in the real file |
| 10 | **A re-run of the build while a job reads the dataset** | Arrow files replaced under a running reader | Sequence §3.3 before §3.5. `--smoke` writes to `*_smoke` paths and can never collide |
| 11 | **`--rule-copies` misread as extra copies** | 7,475 rows / 466 steps instead of 7,021 / 438 | The manifest records both spellings (`rule_copies_total` and `rule_copies_extra_legacy_form`); the dry run prints the row count |
| 12 | **ConstructionSite rows silently dropped** for an unbuildable block | `augmented_v3` quietly smaller than v2 minus nothing | `--max-drop-rate` aborts above 1%; every id is logged and in the manifest |
| 12b | **An output path that resolves onto `datasets/augmented`** by a different spelling | the silent poison of risk #6, reachable by one flag | `_guard_outputs` compares resolved paths; seven spellings are in the test |
| 12c | **The harvest quietly shrinking** — e.g. a reviewer-typed caption containing a newline, which `build_think_body` refuses | 27% of the harvest could drop inside the 1% corpus-wide threshold | a second drop arm at 2% of the MOCS rows alone, plus per-provenance counts in the log and the manifest |
| 12d | **`val` shrinking** | `eval_loss` stops being comparable with v2, and the think validator only enforces the *test* id set | the build **aborts** if val loses a row |
| 13 | **`--tiers 8B`** (capital B) | Accepted with no `choices=`; submits 4 jobs that die on the compute node after the queue wait | Read the dry-run output. There is no code-level guard |
| 14 | **The `_aug1` duplicates land in the GRPO pool** | Correlated reward groups instead of signal | Structural (the pool branches before augmentation) + `test_pool_holds_no_augmented_duplicates` |
| 15 | **`datasets` version skew**: ARC pins `4.3.0`, this box has `5.0.1` | an Arrow schema written by one and read by the other; nothing here has ever run on 4.3.0 | the build never declares a `Features` literal (it copies them off the source dataset, so the 4.x `Sequence` → 5.x `List` rename cannot bite), but `add_column` on a `select()`ed view and `Dataset.from_list(..., features=…)` with a **PIL object** are the two calls to prove first. **`--smoke 400` exercises both** — that is why it is a step, not an option |
| 15b | **`albumentations` is pinned nowhere** (not `requirements.txt`, not `setup_arc.sh`) | below 1.4.21 the per-pipeline seeding call does not exist and the jitter falls back on the globals | the manifest records `augmentation.rng_seeded_via`; `albumentations.set_random_seed` means byte-reproducible, `globals` means best-effort |
| 16 | **`preload_model()` fills the wrong HF cache** (`scripts/submit_pipeline.py:207` runs `hf download` with no `HF_HOME`) | The cache-lock race the preload exists to prevent still happens | Pre-existing, unfixed. `ls ~/.cache/huggingface/hub` vs `ls ~/scratch/hf_cache/hub` after a preload |

**Two things that are reassuringly *not* risks.** Nine hundred-odd rows of text were scanned: the
real review file has **zero** newlines, braces, backticks or fences in any caption or reason, and
**zero** out-of-range or degenerate boxes. And `vo-*-v2`'s results cannot be touched by any of this —
different version token in every path, verified by dry-run.

---

## 5. Stale-doc list

Everything here is a tracked document contradicted by the code or by the repo's actual state.

### 5.1 `CLAUDE.md` — the status header is badly out of date

| claim | file | reality |
|---|---|---|
| "Last commit is `3809e34` … **26 modified + 10 new files are uncommitted**", "the whole `violations_think` arm is uncommitted" | `CLAUDE.md:16-20` | HEAD is **`c582ac0`**, three commits later. The tree has 2 modified + 2 untracked files, **all four MOCS/doc files**; `git status --short core/ data/ evaluation/ experiments/ scripts/ configs/ tests/` is **empty**. The entire vt arm was committed in `a44603a` |
| "**Nothing is running on the cluster.** The last SLURM job finished 2026-09-15" | `CLAUDE.md:11-12` | Jobs `48911273`+ ran **2026-09-24/25** |
| `object_only` / `caption_only`: "never run \| never run" | `CLAUDE.md:32-33` | **Both ran fully at all three tiers** — 18 evaluated run dirs in `results_index/oo_co_v1_dump/results/inference/` |
| "**Next up: `object_only` and `caption_only`** … the first end-to-end runs either has ever had" | `CLAUDE.md:36-39` | Done. The real next thing is v3/v4 |
| "798 tests (797 pass, 1 skipped)" | `CLAUDE.md:194`, `:239` | **1021 before this work, 1075 after.** `HANDOFF.md:21` says 802, `AUDIT_OO_CO_V1.md:16` says 1009, `V3_THINK_IMPLEMENTATION.md:12` says 798 — four documents, four wrong numbers |
| `README_v2.md` §13 P0-4: "`git_is_dirty: true` in all **nine** v2 manifests" | `README_v2.md:752` and four echoes | **Fifteen.** The six oo/co v1 GRPO manifests also record `git_is_dirty: true` at `c582ac0`. The code was committed; the discipline was not |
| `results_index/` contents list | `CLAUDE.md:199` | also holds `oo_co_v1_dump/`, `mocs_annotation_combined/`, `mocs_corpus.tar.gz`, `mocs_review.tar` |

`HANDOFF.md` and `AUDIT_OO_CO_V1.md` both self-describe as temporary with the delete condition "once
the oo/co runs have landed". **That condition is met.** By the repo's own convention they are due for
deletion, not repair.

### 5.2 Claims contradicted by code

| claim | where | reality |
|---|---|---|
| "GRPO trains for a **single epoch** (configs/grpo.yaml)" | `data/build_grpo_pool.py:12`, `data/loader.py:234` | `configs/grpo.yaml:92` is `num_train_epochs: 2`. Two docstrings, same stale claim. The *conclusion* (keep duplicates out of the pool) still holds and is arguably stronger at 2 epochs |
| "`albumentations` … is **ARC-only** and absent from the local dev venv" | `PROMPT_V3_COMBINE.md` §5a; `tests/test_core/test_blocker_fixes.py`'s `test_v2_augmentation_multipliers_unchanged` docstring | It is **installed locally** (2.0.8), which is why `tests/test_data/test_build_v3_datasets.py` can run the real augmentation path. The test's regex-over-source approach is still right — it just has a different justification now (not importing a module whose `main()` writes to `datasets/augmented`) |
| `grpo_pool_v3` "would serve `unified`, `object_only` or `caption_only` equally well" | `PLAN_V3_THINK.md:160` | **False for this generation.** Its MOCS rows carry `[]` for all three object classes and a teacher-written caption. It is task-blind only among the violation tasks |
| "`--census` … the block measures ~60 words (mean) / ~187 worst case" | `configs/tasks/violations_think.yaml`; `PLAN_V3_THINK.md:189` | That was measured on ConstructionSite test rows. **MOCS blocks are longer** — mean **82.2** words, max **126** — because MOCS captions are longer (median 55 words). Still far inside the 1024-token completion budget, but run `--census` rather than quoting the old figure |
| TIME_CONFIG comment: "24h has **NO** confirmed headroom … the only recorded GRPO walltime is from a prompt-only run" | `scripts/submit_pipeline.py:69-73`, `scripts/hpc_grpo.sh:35-37` | Contradicted 200 lines later in the same file (`:272-274`) and by `OPERATIONS.md:522-526`, both listing the real measured v2 times. The stale version is the comment |
| `submit_pipeline.py` docstring lists **four** pipelines / "all four pipelines" | `scripts/submit_pipeline.py:18-23`; also `core/naming.py:12`, `:50`, `:62`, `OPERATIONS.md:83`, `scripts/validate_rewards.py:602` | Five tasks are registered. Comment-only; the code is registry-driven |
| F4 "preload fills the wrong cache" at `submit_pipeline.py:154`; F9 "`--tiers` accepts any string" at `:173` | `AUDIT_OO_CO_V1.md:132`, `:137` | Both bugs are real; both line numbers moved to **207** and **226** when `c582ac0` added the `--time` flags |
| `README_v2.md` §13 **P0-2** ("evaluate `merged-vo-sft-<tier>-v2` with no adapter") | `README_v2.md:750` | Its own stated prerequisite now resolves **negatively**: `--base_model_override` exists only in `models/grpo_trainer.py`; there is no such flag in `models/inference.py` or `experiments/`. P0-2 needs code before it needs GPU time |
| **The prompt tells the model "one box per instance"; the GT does not follow that for rule_3 or rule_4** | `data/prompt_templates.py` (`_VIOLATION_INSTRUCTIONS`, in both `VIOLATIONS_ONLY_PROMPT` and `VIOLATIONS_THINK_PROMPT`): *"List more than one box if more than one instance violates the same rule."* — one generic sentence for all four rules | **Pre-existing and live in v2.** Measured GT is 1.66 / 1.74 boxes per image for rule_1 / rule_2 but **1.01 for rule_4 and 1.16 for rule_3**, and in the *test* split rule_3 and rule_4 are **exactly 1.00**. So for those two rules the training prompt and the scoring key already contradict each other, independently of anything built here. Not fixed: changing the prompt would change it between v2 and v4 and make the comparison two-variable. See §6 open question 7 |
| `OPERATIONS.md:311` — "a wall kill means something else changed" | `OPERATIONS.md` §7.3 | True for v2/v4. **For 8B v3 a wall kill is an expected outcome, not a symptom** (§3.6) — resume is the normal path there, not an investigation trigger |
| `configs/sft.yaml:2`, `:37`, `:66` comments (8198 rows → 512 steps, 20 eval points) | | Correct for `datasets/augmented`, **wrong for these runs** (7,021 → 438 → 17). Left unedited on purpose: `unified` still reads `datasets/augmented` |

### 5.3 Where this prompt was wrong

| the prompt said | reality |
|---|---|
| "`RULE_MULTIPLIERS = {4:2, 2:2, 3:2}`" **and** "+452 duplicates = 7,026 rows … 438 steps" | These disagree. Under the module's own convention (`16` = sixteen *extra* copies) `{4:2,2:2,3:2}` yields **+904** rows and **466** steps. I implemented the arithmetic, not the literal, and renamed the knob to `--rule-copies` (*total* copies) so the ambiguity cannot recur. Both forms are in the manifest |
| "base train rows 6,308 CS + 266 MOCS = 6,574 → 7,026 rows → 438 steps" | **264**, not 266 — one row has a blank caption, one asserts a rule with no reason and no box, and neither can carry a think block. One ConstructionSite train row goes the same way. Base 6,571 → 7,021 → still **438 steps** |
| "265 usable captions, 26 reviewer-edited, 1 empty assertion … decide whether to drop the row or the assertion" | Correct, and they are **two different rows**: the blank caption is `mocs_0022103`, the empty assertion is `mocs_0023379`. Both are dropped, by one uniform rule (§2.3 Q7). Dropping `mocs_0023379` also costs its genuinely verified rule_2 |
| "`albumentations` … is ARC-only and absent from the local dev venv. That constrains where the build can run" | The constraint is real but the cause is different: the build must run on ARC because **`$VLM_DATA_ROOT` is empty locally**, not because of albumentations (which is installed) |
| "`image_path_original` … Does the build run on ARC or locally?" | ARC — but worth knowing that **all 266 images do exist locally** as staged review renders, which is how the end-to-end rehearsal was possible |

---

## 6. Open questions

**1. ~~Does any ConstructionSite row fail to produce a think block?~~ Answered: exactly one, in train.**
`results_index/oo_co_v1_dump/datasets_stats/dataset_report.json` records 0 blank captions in every
split and one train rule_1 violation with a box but no reason. It drops; `val` and `test` are clean.
The residual unknown is narrow: that report does not count **newlines** inside a caption or reason,
which `build_think_body` also refuses. *Settled by:* the dry run in §3.3, which prints the count and
every offending id. Anything above 2 drops is worth looking at before you let it write.

**2. The 36-item negation review** (`PLAN_V3_THINK.md` §4.4) is still not done. A ground-truth reason
like `rule_3: "Either side of the excavation trench is protected." → violated` costs a little
`reward_reasoning` today; in a pre-baked think block it trains *"evidence says protected, therefore
violated"* and is frozen in. ~10 minutes of hand review over 36 of 435 positive reasons. *Settled by:*
doing it, before §3.3. It is the one prerequisite I could not do for you, because it is a judgement
about ground-truth wording, not a build step.

**3. "operating radius" vs "operation radius" — 107 of 113 verified rule_4 reasons.** The dataset's
convention is "operation radius" (`REVIEW_RUBRIC.md:290`). `reward_reasoning` mixes a semantic term
(MiniLM embeddings — essentially blind to this) with `_ngram_f1` (which is not: one unigram and two
bigrams differ per reason), and the LLM judge is semantic. So the cost is small, systematic, and
concentrated in the rarest rule. I **reported it and changed nothing**, because editing verified human
text to chase a string-similarity metric is a data decision, not a build step. *Settled by:* your
call. It is a one-line `str.replace` in `data/mocs_rows.py` if you want it.

**4. The real v3 GRPO per-step cost.** §3.6's 1.5–1.8× is an estimate from token counts, not a
measurement. *Settled by:* the 2B and 4B v3 GRPO runs; read seconds/step off the `.out` log and
decide about 8B then.

**7. The prompt and the GT disagree about box counts for rule_3 and rule_4 — and this is pre-existing.**
`_VIOLATION_INSTRUCTIONS` says *"List more than one box if more than one instance violates the same
rule"* for all four rules, but measured GT is 1.01 boxes/image for rule_4 and 1.16 for rule_3, and
**exactly 1.00 for both in the test split**. v2 trained and was scored under that contradiction too.
`--rule4-box-policy union` makes the *data* self-consistent and consistent with the scoring key; it
does not touch the prompt. The alternatives are to leave it (what v2 did), or to make the box-count
sentence per-rule — which would change the prompt between v2 and v4 and turn that comparison into a
two-variable one, so it is out of scope here. *Settled by:* a decision about whether the prompt or the
GT is wrong, which is a research call, not a build step. Worth noting
`reward_violation_grounding` uses **mask-union IoU**, so the model is never scored on the *number* of
boxes — only on the area they cover.

**5. `rule_2` averages 2.41 boxes per image against ground truth's 1.74.** Unlike rule_4, this is
*not* a convention violation — rule_2's documented convention **is** one box per person, so MOCS
images simply have more people. I changed nothing. *Settled by:* nothing needs settling unless rule_2
grounding comes out low in v4, in which case this is the first thing to look at.

**6. Whether the v4 baseline is worth 3 GPU-jobs.** `vo-baseline-T-v4` is a bit-for-bit repetition of
`vo-baseline-T-v2`: same no-adapter model, same prompt, and inference loads the *default* root
regardless of the dataset override. There is no `--skip-baseline` flag. I would **run it** and treat
the match against v2's baseline metrics as a free reproducibility check on the environment — but it is
~3.4 H100-hours of new information that is zero by construction, and you may prefer to spend them
elsewhere. (The `vt` baselines are *not* redundant: a different prompt gives a genuinely different
zero-shot measurement.)
