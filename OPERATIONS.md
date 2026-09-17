# OPERATIONS.md — the ARC / SLURM runbook

How to actually run this project on the University of Calgary's ARC cluster: set up, submit, watch, recover,
clean up, and get results back to a laptop.

This file is **procedure**. It deliberately contains no architecture and no results.

| You want | Go to |
|---|---|
| why the code is shaped this way, invariants, settled decisions | [`CLAUDE.md`](CLAUDE.md) |
| what the numbers are | [`README_v2.md`](README_v2.md) |
| what the project is | [`README.md`](README.md) |

---

## Contents

1. [Every-session preamble](#1-every-session-preamble)
2. [One-time setup](#2-one-time-setup)
3. [Pre-flight, before you burn a GPU hour](#3-pre-flight-before-you-burn-a-gpu-hour)
4. [Submitting a version](#4-submitting-a-version)
5. [Monitoring](#5-monitoring)
6. [Holding and releasing queued jobs](#6-holding-and-releasing-queued-jobs)
7. [Failure recovery playbook](#7-failure-recovery-playbook)
8. [Deleting the artifacts of a failed or wrong run](#8-deleting-the-artifacts-of-a-failed-or-wrong-run)
9. [Verifying a finished version](#9-verifying-a-finished-version)
10. [Building the results index and comparison tables](#10-building-the-results-index-and-comparison-tables)
11. [Packing a dump and downloading it](#11-packing-a-dump-and-downloading-it)
12. [Local analysis](#12-local-analysis)
13. [Cost reference](#13-cost-reference)

---

## 1. Every-session preamble

Paste this after every fresh SSH login. Nothing below works without it.

```bash
cd $HOME/vlm-safety-reasoning
module purge && module load gcc/13.3.0 python/3.12.5
source $HOME/envs/vlm_grpo/bin/activate
export PYTHONPATH="$HOME/vlm-safety-reasoning:$PYTHONPATH" \
       VLM_DATA_ROOT="$HOME/vlm-finetuning-project1" \
       HF_HOME="$HOME/scratch/hf_cache"
```

`HF_HOME` matters more than it looks: every `scripts/hpc_*.sh` sets it to `$HOME/scratch/hf_cache` for itself,
but a login-node download without it lands in `~/.cache/huggingface`, where no job will ever look — and burns
home-directory quota. Set it before any `hf download`.

> ### ⛔ Never run the full test suite on ARC
>
> `tests/test_core/test_blocker_fixes.py::test_submitter_can_override_gres_for_every_stage` invokes the real
> `scripts/submit_pipeline.py` twice through `subprocess`. `submit_job()` shells out to `sbatch` and only falls
> back to a dummy job id when `sbatch` is *missing*, so on a login node **it submits 8 real GPU jobs**
> (`caption_only`, 2b, v1). That is exactly what happened on 2026-09-14 — jobs `48501131`–`48501137`, all
> cancelled and cleaned. On ARC always use:
>
> ```bash
> pytest tests/ -k "not submitter_can_override_gres"
> ```
>
> Locally on Windows the same test is harmless (no `sbatch` on PATH). The test itself has not been made safe
> yet.

---

## 2. One-time setup

### 2.1 Environment

`scripts/setup_arc.sh` builds `$HOME/envs/vlm_grpo` and holds the authoritative version pins
(`transformers==5.4.0`, `trl==0.23.0`, `datasets==4.3.0`, `unsloth_zoo==2026.8.12`). `requirements.txt` has
loose lower bounds and is **not** the version set — never pin from it.

### 2.2 Data

Both derive from `datasets/processed`, neither reads the other, so they can run in parallel. Built and verified
2026-09-02; only re-run if the source dataset changes.

```bash
sbatch scripts/augment_data.sh    # -> datasets/augmented   (SFT input for unified + vo; CPU-only, ~30 min)
python data/build_grpo_pool.py    # -> datasets/grpo_pool   (GRPO input for all four tasks; ~2 min)
python scripts/dataset_report.py  # -> datasets/stats/dataset_report.json
```

**`scripts/augment_data.sh` has no `set -eo pipefail`, no guarded `cd`, and unconditionally prints "completed
successfully"** — a failed `cd` or a crashed augmentation still exits 0. Verify the row counts, never the exit
code:

```bash
python -c "import json;d=json.load(open('$VLM_DATA_ROOT/datasets/augmented/augment_manifest.json'));print(d)"
# expect total 6308 -> 8198; per-rule 609->871 / 53->689 / 98->696 / 42->714; safe 5528 unchanged
python -c "import json;d=json.load(open('$VLM_DATA_ROOT/datasets/grpo_pool/build_manifest.json'));print(d)"
# expect pool_total 1732, exactly 866/866 safe/violation; by rule 677 / 59 / 109 / 46
```

### 2.3 Cache the LLM judge

Compute nodes have **no internet**, and Llama 3 is a gated repo. Do this on the login node, once.

1. In a browser, accept the license at `https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct` using the
   same account your HF token belongs to. Approval is usually minutes.
2. Then:

```bash
hf auth login                                                            # token with gated-repo read access
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"   # ~16 GB
```

`--exclude "original/*"` is not optional in practice — the repo also ships `original/consolidated.00.pth`, a
second ~16 GB copy in Meta's own format that transformers never reads. If the gate is refused, set
`configs/base.yaml::llm_judge.model_id` to `NousResearch/Meta-Llama-3-8B-Instruct` (same weights, ungated).

**Verify it on a GPU, not with `snapshot_download`.** A `snapshot_download(..., local_files_only=True)` check
raises `IncompleteSnapshotError` because of the `original/*` files you deliberately excluded — that error is a
false alarm. The pipeline uses `from_pretrained`, so test that path:

```bash
srun --partition=gpu-h100 --gres=gpu:h100:1 --mem=64G --time=00:20:00 --pty bash -lc '
  module load gcc/13.3.0 python/3.12.5 && source $HOME/envs/vlm_grpo/bin/activate
  export HF_HOME=$HOME/scratch/hf_cache
  python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
m=\"meta-llama/Meta-Llama-3-8B-Instruct\"
AutoTokenizer.from_pretrained(m, local_files_only=True)
AutoModelForCausalLM.from_pretrained(m, local_files_only=True, dtype=\"bfloat16\", device_map=\"cuda\")
print(\"judge OK\")"'
```

The judge is **fail-soft** (`llm_judge.fail_hard: false`): if it cannot load, the evaluation still completes,
writes `llm_judge_status.json` with `"status": "failed"`, and omits the judge keys. So a cache problem never
costs a 3004-image evaluation — but it also never announces itself in the exit code. **Check that file, not
the job status.**

---

## 3. Pre-flight, before you burn a GPU hour

```bash
# 1. the reward surface — fails loudly if any degenerate policy beats the honest one,
#    if a break-even IoU exceeds 0.75, or if the violation operating point leaves (0.20, 0.50)
python scripts/validate_rewards.py
python scripts/validate_rewards.py --task violations_only --probe --pool-stats

# 2. the test suite, MINUS the job-submitting test
pytest tests/ -k "not submitter_can_override_gres"

# 3. GRPO assembly + real prompt length for the task (does NOT validate the pool itself)
python scripts/preflight_grpo.py --tier 2b --task violations_only

# 4. the GRPO image column really is named `image`
python scripts/verify_grpo_images.py
```

Then confirm the tree is clean and the commit is the one you mean — every run records
`git_commit` / `git_is_dirty` into its manifest, and all nine v2 runs recorded `git_is_dirty: true`, which makes
them non-reproducible from a hash alone:

```bash
git status --short && git log --oneline -1
```

---

## 4. Submitting a version

One submitter for every task. `--version` is required and must match `v<digits>`.

```bash
python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b 8b --version v2
```

That submits **4 jobs per tier** — `baseline` (independent) and `sft → merge → grpo` chained with `afterok`
dependencies — so 12 jobs for three tiers. Any number of tasks/tiers can be in flight at the same `--version`
without colliding; the isolation is a tested invariant.

Useful variants:

| You want | Add |
|---|---|
| skip the login-node model pre-download | `--skip-preload` |
| force every stage onto one card type (debug only) | `--gres gpu:h100:1` |
| a dry run | run it on Windows — no `sbatch`, so it prints the exact commands with `DUMMY_JOB_ID` |

**`--tiers` accepts any string** (no `choices=`). A typo like `--tiers 8B` is accepted, submits four jobs, and
only fails on the compute node after the queue wait. Check spelling by eye.

**Verify the preload actually helped** (it inherits the login shell's environment and has no `HF_HOME`
override of its own):

```bash
ls ~/scratch/hf_cache/hub    # the cache every job reads
ls ~/.cache/huggingface/hub  # where an un-exported preload lands instead
```

Capturing the submission output is worth it — it is the only record of which job id is which stage:

```bash
python scripts/submit_pipeline.py --task violations_only --tiers 2b 4b 8b --version v2 2>&1 \
  | tee ~/submit_vo_v2.log
grep -E "Running: sbatch|Successfully submitted" ~/submit_vo_v2.log
```

---

## 5. Monitoring

```bash
squeue -u $USER                                    # what is queued / running, and why it is waiting
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
sacct -u $USER -S 2026-09-14 --format=JobID%10,JobName%20,State%12,Elapsed,End,NodeList%8 | grep -v '\.'
scontrol show job <jobid>                          # the full record, incl. the GRES actually requested
```

Common `squeue` reasons:

| Reason | Meaning |
|---|---|
| `(Priority)` / `(Resources)` | normal queueing |
| `(Dependency)` | waiting on its `afterok` parent — expected for merge and grpo |
| `(DependencyNeverSatisfied)` | **its parent failed.** The job will wait forever; cancel it |
| `(QOSMaxGRESPerUser)` | at your account's concurrent-GPU limit (observed at 4 GPUs during the v2 submission). Jobs start as earlier ones finish; nothing is wrong |
| `(JobHeldUser)` | you ran `scontrol hold` on it |

Reading progress — **W&B is offline in every phase script**, so nothing appears in the web dashboard:

```bash
tail -f $VLM_DATA_ROOT/logs/sft_vo_<jobid>.out       # SFT logs every step, evals every 25
tail -f $VLM_DATA_ROOT/logs/grpo_vo_<jobid>.out      # GRPO logs every 5 steps
grep -c "Error in reward function" $VLM_DATA_ROOT/logs/grpo_vo_<jobid>.err   # must be 0
wandb sync $HOME/scratch/wandb/offline-run-*         # optional, after the fact
```

A GRPO curve straight out of the log:

```bash
python - "$VLM_DATA_ROOT/logs/grpo_vo_<jobid>.out" <<'EOF'
import ast, sys
rows = [ast.literal_eval(l.strip()) for l in open(sys.argv[1], errors="ignore") if l.startswith("{'loss'")]
for d in rows[::4] + rows[-1:]:
    print(f"step {round(float(d['epoch'])*54):>3}  lr {d['learning_rate']:>9}  kl {d['kl']:>9}  "
          f"reward {d['reward']:>6}  zero_std {d['frac_reward_zero_std']:>5}  "
          f"len {d['completions/mean_length']:>6}  clipped {d['completions/clipped_ratio']}")
EOF
```

(The `*54` converts epoch to step: 54 steps per epoch for the 1732-image pool. Every value in these log dicts
is a **string** — coerce before doing arithmetic.)

What healthy looks like, measured on v2: `reward/mean` climbing ~0.06 over the first 40–70 steps then
plateauing; `kl` 0.003–0.005; `frac_reward_zero_std` 0.40–0.47; `completions/clipped_ratio` 0. What is *not* a
red flag: `kl` well below 0.01, and `reward_format/std` at 0 post-SFT (saturated by design).

---

## 6. Holding and releasing queued jobs

A held job stays in the queue but is not eligible to start. Use it when a later stage is queued behind
something you are about to fix, or when you want to stop a chain without losing its place.

```bash
scontrol hold <jobid> [<jobid> ...]     # -> squeue shows (JobHeldUser)
scontrol release <jobid> [<jobid> ...]  # back to normal queueing
scontrol show job <jobid> | grep -i -E "JobState|Reason|Dependency"
```

Hold does **not** stop a job that is already `RUNNING` — only `scancel` does. And a held job keeps its
`afterok` dependents waiting, which is usually the point.

---

## 7. Failure recovery playbook

### 7.1 A node crashed — `NODE_FAIL`

**What it looks like:** `sacct` shows `NODE_FAIL`, two or more of your jobs on the *same node* end at the *same
second*, and the `.out` log stops mid-line with **no Python traceback**. Observed 2026-09-14: `48501344`
(baseline) and `48501345` (sft) both died on `mgh5` at `03:56:49` after 1:15, the SFT log ending mid-inference
at sample 48/94.

**This is not your code.** Recovery:

```bash
# 1. cancel the now-orphaned dependents (they will read DependencyNeverSatisfied)
scancel <merge_jobid> <grpo_jobid>

# 2. delete every artifact the dead jobs produced -- see section 8
# 3. resubmit only the affected tier
python scripts/submit_pipeline.py --task violations_only --tiers 4b --version v2
```

Do **not** resubmit the whole version; the healthy tiers keep running and their names would collide.

### 7.2 A parent failed — `DependencyNeverSatisfied`

The dependent will never run. Cancel it (`scancel <jobid>`), fix or clean the parent, resubmit the tier.

### 7.3 A GRPO job hit the 24 h wall

**It is not lost.** `models/grpo_trainer.py` auto-resumes from the last checkpoint (`save_steps: 20`, so ≤ 20
steps at risk). **Re-submit the identical GRPO job with the same variant name** — it continues rather than
restarts. Measured v2 walltimes leave real headroom (8B GRPO ≈ 11.5 h), so a wall kill means something else
changed.

### 7.4 A CUDA OOM

Not seen in v2 at any tier. If it happens in GRPO, drop `per_device_train_batch_size` from 16 to 8 in
`configs/grpo.yaml` (previously verified safe) — and note that the change only affects jobs submitted *after*
it, since `sbatch` freezes the batch script at submit time.

### 7.5 You submitted the wrong thing

`scancel <jobid> ...` for specific jobs, `scancel -u $USER` for everything. Then clean the artifacts (§8).
`sacct ... --format=...,NodeList` plus `scontrol show job <id> | grep AllocNode` tells you which login session
submitted a mystery job — that is how the 8 accidental `caption_only` jobs were traced back to a `pytest` run
on `mgh3`.

### 7.6 Only some jobs failed

Nothing about the pipeline is all-or-nothing. Tiers are independent, and within a tier the baseline is
independent of the `sft → merge → grpo` chain. Rescue the smallest unit that failed.

---

## 8. Deleting the artifacts of a failed or wrong run

**Do this before resubmitting.** A half-written checkpoint or a stale `predictions.jsonl` will be picked up by
the next stage or silently indexed by the analysis toolset.

Everything a run writes is namespaced by `<prefix>-<phase>-<tier>-<version>`, so deletion is exact. For tier
`4b`, task `vo`, version `v2`:

```bash
T=4b; P=vo; V=v2

# 1. LOOK FIRST — never pipe find straight into rm
find $VLM_DATA_ROOT/checkpoints/qwen3vl-$T -maxdepth 1 -name "*${P}-*-${T}-${V}*"
find $VLM_DATA_ROOT/results/inference     -maxdepth 1 -name "${P}-*-${T}-${V}*"
find $VLM_DATA_ROOT/datasets/stats        -name "oversample_manifest_${T}_${P}-*-${T}-${V}.json"
ls   $VLM_DATA_ROOT/logs | grep -E "_${P}_(<jobid1>|<jobid2>)\.(out|err)"

# 2. then delete
rm -rf $VLM_DATA_ROOT/checkpoints/qwen3vl-$T/${P}-sft-${T}-${V} \
       $VLM_DATA_ROOT/checkpoints/qwen3vl-$T/${P}-grpo-${T}-${V} \
       $VLM_DATA_ROOT/checkpoints/qwen3vl-$T/merged-${P}-sft-${T}-${V}
rm -rf $VLM_DATA_ROOT/results/inference/${P}-baseline-${T}-${V} \
       $VLM_DATA_ROOT/results/inference/${P}-sft-${T}-${V}_final \
       $VLM_DATA_ROOT/results/inference/${P}-grpo-${T}-${V}_final
rm -f  $VLM_DATA_ROOT/datasets/stats/oversample_manifest_${T}_${P}-sft-${T}-${V}.json

# 3. confirm nothing is left
find $VLM_DATA_ROOT -name "*${P}-*-${T}-${V}*" -not -path "*/logs/*"
```

Logs are the one thing worth **keeping** — they are the only record of a failure, and they carry the training
curves. Delete them only if they would confuse a later `find … -newermt` when packing a dump.

`datasets/` and the HF cache are read-only during training and must never be touched by a cleanup.

---

## 9. Verifying a finished version

```bash
# 9 metrics files for a 3-tier violations_only version
ls $VLM_DATA_ROOT/results/inference/vo-*-v2*/evaluation_results/metrics.json | wc -l   # expect 9

# the judge really ran, with the same rubric, and parsed everything
for f in $VLM_DATA_ROOT/results/inference/vo-*-v2*/evaluation_results/llm_judge_status.json; do
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{sys.argv[1].split('/')[-3]:24} \
{d['status']:8} items={d['n_items']:>4} unparsed={d['n_unparsed']:>3} \
rubric={d['rubric_sha256'][:12]} {d['elapsed_seconds']}s\")" "$f"
done

# no silently-swallowed reward exceptions
grep -l "Error in reward function" $VLM_DATA_ROOT/logs/grpo_vo_*.err

# every inference covered the whole split
grep -h "structural_total_samples_count" $VLM_DATA_ROOT/results/inference/vo-*-v2*/evaluation_results/metrics.json
# expect 3004 everywhere
```

Expected for a good version: `status: ok` in all nine, identical `rubric_sha256`, `unparsed=0`, and
`structural_total_samples_count = 3004`.

---

## 10. Building the results index and comparison tables

Run on the **login node** — no GPU, no SLURM job, pure file I/O, read-only with respect to the pipeline.

```bash
D=$HOME/v2_dump
mkdir -p $D/compare_v2 $D/compare_all_versions

# one index for the new version, one spanning every version of the task
python -m experiments.build_results_index --tasks violations_only --versions v2 --out $D/index_v2.json
python -m experiments.build_results_index --tasks violations_only               --out $D/index_all.json
```

The index carries `violation_per_image_outcomes_b64` — one byte per image (predicted rule mask in the high
nibble, ground truth in the low nibble), ~4 KB per run. That is all the paired bootstrap needs, which is why
the whole analysis travels as one small JSON.

```bash
python -m experiments.compare_all --index $D/index_v2.json  --out $D/compare_v2 \
       --no-charts --bootstrap 0 2>&1 | tee $D/compare_v2/console.txt
python -m experiments.compare_all --index $D/index_all.json --out $D/compare_all_versions \
       --no-charts --bootstrap 0 2>&1 | tee $D/compare_all_versions/console.txt
```

**`--bootstrap 0` on ARC is deliberate.** The default 2000-resample bootstrap is slow on a shared login node
(it was interrupted during v2). Run it locally instead — see §12. `--no-charts` likewise: matplotlib may be
absent from the login env, and charts are better made where you will look at them.

**If `compare_all` raises `FileNotFoundError: .../index_v2.json`, you skipped `build_results_index`.** The two
commands are separate on purpose; the index is the only thing `compare_all` reads.

---

## 11. Packing a dump and downloading it

`index_*.json` alone is enough for the tables and the bootstrap. Pack the rest when you want the raw
predictions, the training curves, the provenance, or qualitative material. The v2 dump came to **12 MB /
215 files**.

```bash
D=$HOME/v2_dump
mkdir -p $D/results/inference $D/logs $D/provenance $D/datasets_stats

# per-run results: metrics + judge + raw and repaired predictions + repair reports
for run in vo-baseline-2b-v2 vo-sft-2b-v2_final vo-grpo-2b-v2_final \
           vo-baseline-4b-v2 vo-sft-4b-v2_final vo-grpo-4b-v2_final \
           vo-baseline-8b-v2 vo-sft-8b-v2_final vo-grpo-8b-v2_final; do
  S=$VLM_DATA_ROOT/results/inference/$run
  mkdir -p $D/results/inference/$run
  cp -r $S/evaluation_results $S/repair_applied $D/results/inference/$run/ 2>/dev/null
  cp    $S/predictions.jsonl  $S/run_manifest.json $D/results/inference/$run/ 2>/dev/null
done

# logs -- the ONLY place the training curves exist, since W&B is offline.
# -newermt filters out the previous version's logs; set it to just before you submitted.
find $VLM_DATA_ROOT/logs -maxdepth 1 -newermt "2026-09-14 02:00" \
     \( -name "*_vo_*" -o -name "*vo-*-v2*" \) -exec cp {} $D/logs/ \;

# provenance: the exact merged config + git commit each trained run used
for t in 2b 4b 8b; do for v in vo-sft-$t-v2 vo-grpo-$t-v2; do
  mkdir -p $D/provenance/$v
  cp $VLM_DATA_ROOT/checkpoints/qwen3vl-$t/$v/run_config.json \
     $VLM_DATA_ROOT/checkpoints/qwen3vl-$t/$v/run_manifest.json \
     $VLM_DATA_ROOT/checkpoints/qwen3vl-$t/$v/training_state.json $D/provenance/$v/ 2>/dev/null
done; done

# dataset denominators
cp $VLM_DATA_ROOT/datasets/stats/*.json $D/datasets_stats/ 2>/dev/null
cp $VLM_DATA_ROOT/datasets/augmented/augment_manifest.json \
   $VLM_DATA_ROOT/datasets/grpo_pool/build_manifest.json $D/datasets_stats/ 2>/dev/null

cd $HOME && tar -czf v2_dump.tar.gz v2_dump && du -sh v2_dump.tar.gz && find v2_dump -type f | wc -l
```

Checkpoints are **not** packed — tens of GB, and no analysis needs them. `2>/dev/null` hides the expected
"no such file" noise for runs that legitimately lack a `run_manifest.json`.

Then, from a terminal **on the laptop** (not inside the SSH session):

```powershell
scp nabeel.shan@arc.ucalgary.ca:/home/nabeel.shan/v2_dump.tar.gz `
    D:\Abrd\Mitacs\UoC\Research\vlm-safety-reasoning\results_index\
cd D:\Abrd\Mitacs\UoC\Research\vlm-safety-reasoning\results_index
tar -xzf v2_dump.tar.gz    # Windows 10+ ships tar
dir v2_dump
```

MobaXterm's SFTP pane does the same by drag-and-drop. `results_index/` is git-ignored, which is where the dump
belongs.

---

## 12. Local analysis

```powershell
.\venv\Scripts\Activate.ps1
python -m experiments.compare_all --index results_index\v2_dump\index_v2.json `
       --out results_index\analysis_v2 --bootstrap 2000
```

Writes `master_wide.csv` (235 metric keys × 9 runs), `delta_vo_v2.csv`, `repair_stats.csv`,
`significance.csv`, and ~120 charts under `charts/`. Add `--tasks/--tiers/--phases/--versions` to narrow,
`--no-charts` to skip plotting, `--seed` to change the bootstrap seed.

Qualitative material needs the flat layout:

```powershell
python scripts/fetch_results.py --task violations_only --version v2 --tiers 2b 4b 8b
python -m experiments.extract_qualitative --task violations_only --tier 8b --version v2
```

Two limits to know before trusting an output: `compare_all` **never pairs across `--version`**, and it
bootstraps only `f1_micro` / `f1_macro`. Anything else — precision, recall, F2, per-rule or image-level
intervals, v1↔v2 deltas — has to be computed separately. See `CLAUDE.md` → *Local analysis*.

---

## 13. Cost reference

Measured on the v2 `violations_only` runs (whole job: training + 3004-image inference + repair + evaluation +
LLM judge), one H100 each.

| tier | baseline | SFT | merge | GRPO | tier total |
|---|---|---|---|---|---|
| 2B | 1:19:02 | 1:09:08 | 0:01:02 | 5:23:31 | ~7 h 53 m |
| 4B | 1:18:36 | 1:45:55 | 0:03:30 | 9:21:57 | ~12 h 30 m |
| 8B | 0:47:37 | 2:01:50 | 0:01:48 | 11:26:53 | ~14 h 18 m |

A full 3-tier `violations_only` version is **≈ 35 H100-hours**, plus whatever the queue costs. Add ~2.5 h for
the 4B node failure and its resubmission to get the ~37 h the v2 run actually consumed. Walltime requests:
baseline/SFT 12 h, merge 1.5 h, GRPO 24 h (the `gpu-h100` partition `MaxTime`); memory: 150 G for
baseline/SFT, 80 G merge, 250 G GRPO.

The LLM judge adds only **147–288 s** per evaluation. Disk: ~16 GB one-time for the judge in
`$HOME/scratch/hf_cache`, plus the per-tier checkpoints (`save_total_limit: 3` rotates numbered checkpoints,
but `best/`, `final/` and `persistent-checkpoint-N` are never rotated — those accumulate).
