#!/usr/bin/env python3
"""
Gate G2 — no-GPU pre-flight validator for the reward surface and the token budget.

Two independent checks, either of which can be run alone:

  1. REWARD LANDSCAPE PROBE (--probe)
     Scores a set of synthetic policies against real ground truth and asserts that
     honest behaviour wins. This is what catches reward-hacking BEFORE a GRPO job
     burns GPU hours, and it is what would have caught the two exploits this script
     was written for:

       B4  object_only: with a flat true-negative constant of 0.15 and the measured
           pool prevalences, the break-even IoU for `rebar` was 1.55 and for
           `worker_with_white_hard_hat` 1.15. Both exceed 1.0, so suppressing those
           classes was strictly dominant no matter how good the detector became.

       B5  violations_only: a contentless {"reason": "", "bounding_box": []} earned
           a perfect F-beta = 1.0 on the 0.40-weighted identification component, and
           unconditionally asserting rule_1 beat honest abstention 0.294 vs 0.168.

  2. TOKEN CENSUS (--census)
     Tokenizes SYSTEM_PROMPT + task prompt + SFT target over the task's real SFT
     split and reports the length distribution, so you know whether any target is
     being silently truncated at the SFT max_seq_length. Requires the dataset and a
     tokenizer; skipped automatically when either is unavailable.

Neither check needs a GPU. The probe needs no dataset either unless --pool-stats
is passed, so it runs anywhere, including a laptop.

Usage:
    # everything, all four tasks (probe runs offline; census needs the dataset)
    python scripts/validate_rewards.py

    # just the exploit checks for one task
    python scripts/validate_rewards.py --task object_only --probe

    # recompute class prevalences from the real GRPO pool, then check break-evens
    python scripts/validate_rewards.py --task object_only --probe --pool-stats

    # token census only (needs datasets + a tokenizer)
    python scripts/validate_rewards.py --task caption_only --census

Exit code is 0 only if every requested check passes. Wire it into CI or run it
before every `submit_pipeline.py`.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import GROUNDING_CLASSES, RULES, VALID_TASKS
from core.tasks import CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS, task_has

# Measured prevalences in datasets/grpo_pool. Override with --pool-stats, which
# recomputes them from the real pool.
DEFAULT_POOL_PREVALENCE = {
    "excavator": 0.361,
    "rebar": 0.088,
    "worker_with_white_hard_hat": 0.115,
}
DEFAULT_RULE_PREVALENCE = {"rule_1": 0.391, "rule_2": 0.034, "rule_3": 0.063, "rule_4": 0.027}
DEFAULT_SAFE_RATE = 0.50

# A class is only worth emitting if a competent detector can clear its break-even
# IoU. Anything above this ceiling means the reward is telling the model to give up
# on that class.
BREAKEVEN_CEILING = 0.75

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


# The reward assembly logs one INFO line per call and this script calls it once
# per policy per ground truth. Quiet it so the tables stay readable.
try:
    from loguru import logger as _loguru
    _loguru.disable("rewards.unified_reward")
except Exception:
    pass


def _ok(msg):
    print(f"  {GREEN}PASS{RESET}  {msg}")


def _fail(msg):
    print(f"  {RED}FAIL{RESET}  {msg}")


def _info(msg):
    print(f"  {DIM}····{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Synthetic completions
# ---------------------------------------------------------------------------

def _fenced(obj):
    return "```json\n" + json.dumps(obj, separators=(",", ":")) + "\n```"


def _gt_for(task):
    """One concrete ground truth carrying every field any task needs."""
    return {
        "caption": "Two workers in white hard hats stand beside a yellow excavator "
                   "near a stack of exposed rebar on a muddy site.",
        "rule_1_violation": {"bounding_box": [[0.1, 0.2, 0.3, 0.6]],
                             "reason": "A worker on foot is not wearing a hard hat."},
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": {"bounding_box": [[0.5, 0.4, 0.8, 0.9]],
                             "reason": "A worker stands inside the excavator operating radius."},
        "excavator": [[0.4, 0.3, 0.9, 0.8]],
        "rebar": [[0.0, 0.7, 0.5, 1.0]],
        "worker_with_white_hard_hat": [],
    }


def _safe_gt():
    gt = _gt_for(None)
    for r in RULES:
        gt[f"{r}_violation"] = None
    for c in GROUNDING_CLASSES:
        gt[c] = []
    return gt


def _to_1000(box):
    return [round(v * 1000) for v in box]


def _policies(task, gt):
    """(label, completion, is_honest) for the given task and ground truth."""
    has_cap, has_obj, has_vio = (task_has(task, c) for c in (CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS))
    out = {}

    if has_cap and not has_obj and not has_vio:  # caption_only: bare prose
        out["honest"] = gt["caption"]
        out["degenerate/generic"] = "A construction site with workers and equipment present."
        out["degenerate/blank"] = "   "
        out["degenerate/fenced"] = '```json\n{"caption": "%s"}\n```' % gt["caption"]
        return out

    def _obj(payload):
        return {c: payload.get(c, []) for c in GROUNDING_CLASSES}

    def _vio(payload):
        return {f"{r}_violation": payload.get(r) for r in RULES}

    body_honest, body_empty, body_hall = {}, {}, {}

    if has_cap:
        body_honest["caption"] = gt["caption"]
        body_empty["caption"] = gt["caption"]
        body_hall["caption"] = gt["caption"]
    if has_obj:
        body_honest.update(_obj({c: [_to_1000(b) for b in gt[c]] for c in GROUNDING_CLASSES}))
        body_empty.update(_obj({}))
        body_hall.update(_obj({c: [[100, 100, 300, 300]] for c in GROUNDING_CLASSES}))
    if has_vio:
        honest_v = {}
        for r in RULES:
            v = gt[f"{r}_violation"]
            honest_v[r] = None if v is None else {
                "bounding_box": [_to_1000(b) for b in v["bounding_box"]], "reason": v["reason"]}
        body_honest.update(_vio(honest_v))
        body_empty.update(_vio({}))
        body_hall.update(_vio({r: {"bounding_box": [[0, 0, 1000, 1000]],
                                   "reason": "A safety violation is visible in this image."}
                               for r in RULES}))

    out["honest"] = _fenced(body_honest)
    out["degenerate/all-empty"] = _fenced(body_empty)
    out["degenerate/hallucinate-all"] = _fenced(body_hall)

    if has_obj:
        # The B4 exploit: emit only the common class, suppress the rare ones.
        suppress = dict(body_honest)
        suppress.update(_obj({"excavator": [_to_1000(b) for b in gt["excavator"]]}))
        out["degenerate/common-class-only"] = _fenced(suppress)
    if has_vio:
        # The B5 exploits.
        contentless = dict(body_honest)
        contentless.update(_vio({r: {"reason": "", "bounding_box": []} for r in RULES}))
        out["degenerate/contentless-assert-all"] = _fenced(contentless)

        always_r1 = dict(body_honest)
        always_r1.update(_vio({"rule_1": {"bounding_box": [[0, 0, 1000, 1000]],
                                          "reason": "A worker is not wearing the required PPE."}}))
        out["degenerate/always-assert-rule1"] = _fenced(always_r1)

    return out


def _score(task, completion, gt):
    from rewards.unified_reward import get_reward_funcs_for_task
    funcs, weights = get_reward_funcs_for_task(task)
    per = {}
    total = 0.0
    for fn, w in zip(funcs, weights):
        v = fn(completions=[completion], ground_truth=[json.dumps(gt)])[0]
        per[fn.__name__] = v
        total += w * v
    return total, per


# ---------------------------------------------------------------------------
# Check 1 — reward landscape probe
# ---------------------------------------------------------------------------

def probe(task, prevalence, rule_prevalence, safe_rate):
    print(f"\n{YELLOW}REWARD LANDSCAPE — {task}{RESET}")
    failures = []

    violation_gt, safe_gt = _gt_for(task), _safe_gt()

    for label_gt, gt in (("violation image", violation_gt), ("safe image", safe_gt)):
        print(f"\n  {DIM}ground truth: {label_gt}{RESET}")
        rows = []
        for label, completion in _policies(task, gt).items():
            total, per = _score(task, completion, gt)
            rows.append((label, total, per))
        honest = next(t for l, t, _ in rows if l == "honest")
        width = max(len(l) for l, _, _ in rows)
        for label, total, per in sorted(rows, key=lambda r: -r[1]):
            mark = "<-- honest" if label == "honest" else ""
            comp = "  ".join(f"{k.replace('reward_', '')}={v:.3f}" for k, v in per.items())
            print(f"      {label:<{width}}  {total:6.4f}   {DIM}{comp}{RESET} {mark}")
        for label, total, _ in rows:
            if label != "honest" and total > honest + 1e-9:
                failures.append(
                    f"{task}/{label_gt}: '{label}' ({total:.4f}) beats honest ({honest:.4f})")

    # Per-class break-even, the B4 check.
    if task_has(task, CAP_OBJECTS):
        from rewards.reward_utils import grounding_tn_constant
        print(f"\n  {DIM}per-class break-even IoU (emit is positive-EV above this){RESET}")
        for cls in GROUNDING_CLASSES:
            p = prevalence[cls]
            c = grounding_tn_constant(task, cls)
            be = c * (1 - p) / p if p > 0 else float("inf")
            flag = "" if be <= BREAKEVEN_CEILING else f"  {RED}<-- above ceiling {BREAKEVEN_CEILING}{RESET}"
            print(f"      {cls:<28} p={p:.3f}  c={c:.3f}  break-even IoU={be:.3f}{flag}")
            if be > BREAKEVEN_CEILING:
                failures.append(
                    f"{task}: class '{cls}' break-even IoU {be:.3f} > {BREAKEVEN_CEILING} — "
                    "suppressing it is dominant; lower grounding_tn_constant for this class")

    # Expected value over the pool for the two zero-vision policies, the B5 check.
    if task_has(task, CAP_VIOLATIONS):
        from rewards.reward_utils import reward_constant
        c = float(reward_constant(task, "violation_tn_constant", 0.15))
        ev_safe = safe_rate * c
        ev_assert = rule_prevalence["rule_1"] * 1.0
        print(f"\n  {DIM}zero-vision policy EV on the pool (identification component only){RESET}")
        print(f"      always-safe            {ev_safe:.4f}   (P(safe)={safe_rate:.2f} x c={c:.2f})")
        print(f"      always-assert-rule_1   {ev_assert:.4f}   (P(rule_1)={rule_prevalence['rule_1']:.3f})")
        if ev_assert > ev_safe:
            failures.append(
                f"{task}: unconditional rule_1 assertion (EV {ev_assert:.4f}) beats honest "
                f"abstention (EV {ev_safe:.4f}) — raise violation_tn_constant above "
                f"{ev_assert / safe_rate:.3f}")

    # Component variance: a component that never varies contributes no gradient.
    print(f"\n  {DIM}per-component spread across the probed policies{RESET}")
    all_per = []
    for gt in (violation_gt, safe_gt):
        for _, completion in _policies(task, gt).items():
            all_per.append(_score(task, completion, gt)[1])
    names = list(all_per[0].keys())
    varying = 0
    for n in names:
        vals = [p[n] for p in all_per]
        sd = statistics.pstdev(vals)
        varying += sd > 1e-6
        print(f"      {n:<32} min={min(vals):.3f} max={max(vals):.3f} sd={sd:.4f}")
    if varying == 0:
        failures.append(f"{task}: no reward component varies at all — GRPO would get zero gradient")
    else:
        _info(f"{varying}/{len(names)} components vary")

    print()
    if failures:
        for f in failures:
            _fail(f)
    else:
        _ok(f"{task}: honest behaviour is optimal under every probed condition")
    return failures


# ---------------------------------------------------------------------------
# Check 2 — token census
# ---------------------------------------------------------------------------

def census(task, tokenizer_name=None, limit=None):
    print(f"\n{YELLOW}TOKEN CENSUS — {task}{RESET}")
    from core.config import load_config
    from data.prompt_templates import SYSTEM_PROMPT, get_prompt_for_task
    from data.preprocessor import build_target_json

    sft_cfg = load_config(task=task, training_kind="sft")
    cap = sft_cfg.get("max_seq_length", 2048)

    try:
        from data.loader import load_processed_dataset
        splits = load_processed_dataset(subdir=sft_cfg.get("sft_dataset_subdir"))
    except Exception as e:
        _info(f"dataset unavailable ({type(e).__name__}) — census skipped")
        return []

    try:
        from transformers import AutoTokenizer
        from models.model_loader import get_model_info
        name = tokenizer_name or get_model_info(sft_cfg.get("active_tier", "2b"))["hf_path"]
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        _info(f"tokenizer unavailable ({type(e).__name__}) — census skipped")
        return []

    prompt = get_prompt_for_task(task)
    base = len(tok(SYSTEM_PROMPT + prompt).input_ids)
    rows = splits["train"]

    # Vision tokens count toward SFTConfig.max_length, which bounds prompt + target as
    # ONE sequence. An earlier version of this census compared TEXT-ONLY length against
    # that cap and merely printed a note telling the reader to add the vision tokens
    # themselves -- so it could print PASS while the real sequence truncated. Measure
    # them instead: push one real image through the processor under the configured pixel
    # bounds and read the grid.
    vision = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(name)
        ip = getattr(proc, "image_processor", None)
        if ip is not None:
            lo = sft_cfg.get("image_min_pixels")
            hi = sft_cfg.get("image_max_pixels")
            if lo and hi:
                ip.size = {"shortest_edge": lo, "longest_edge": hi}
                for attr, val in (("min_pixels", lo), ("max_pixels", hi)):
                    if hasattr(ip, attr):
                        setattr(ip, attr, val)
            out = ip(images=[rows[0]["image"]], return_tensors="pt")
            grid = out["image_grid_thw"][0]
            merge = getattr(ip, "merge_size", 2) or 2
            vision = int(int(grid[0]) * int(grid[1]) * int(grid[2]) // (merge * merge))
    except Exception as e:
        _info(f"could not measure vision tokens ({type(e).__name__}) — falling back to 1270")

    measured = vision is not None
    if vision is None:
        vision = 1270  # documented worst case at the 1.2 MP cap

    # Column-wise access so the image is decoded only for the one probe above.
    cols = [c for c in rows.column_names if c != "image"]
    text_only = []
    n = min(limit or len(rows), len(rows))
    for row in rows.select_columns(cols).select(range(n)):
        text_only.append(base + len(tok(build_target_json(row, task=task)).input_ids))

    text_only.sort()
    totals = [t + vision for t in text_only]
    p99 = totals[int(0.99 * (len(totals) - 1))]
    print(f"      text-only prompt        {base}")
    print(f"      vision tokens           {vision}"
          f"{'  (measured)' if measured else '  (assumed worst case)'}")
    print(f"      text-only min/mean/max  {text_only[0]} / "
          f"{sum(text_only)//len(text_only)} / {text_only[-1]}")
    print(f"      TRUE  min/p99/max       {totals[0]} / {p99} / {totals[-1]}"
          f"   {DIM}(text + vision){RESET}")
    print(f"      SFTConfig.max_length    {cap}")

    failures = []
    if totals[-1] >= cap:
        failures.append(
            f"{task}: longest sequence {totals[-1]} (text {text_only[-1]} + vision {vision}) "
            f">= max_length {cap} — targets TRUNCATE. Raise sft.yaml max_seq_length, or "
            f"shorten the prompt."
        )
        _fail(failures[-1])
    else:
        _ok(f"{task}: longest sequence {totals[-1]} < {cap} "
            f"({cap - totals[-1]} tokens of margin, vision included)")
    return failures


# ---------------------------------------------------------------------------
# Pool statistics
# ---------------------------------------------------------------------------

def pool_stats():
    """Recompute class and rule prevalence from the real GRPO pool."""
    from data.loader import load_grpo_pool
    pool = load_grpo_pool()
    n = len(pool)
    cls_counts = {c: 0 for c in GROUNDING_CLASSES}
    rule_counts = {r: 0 for r in RULES}
    safe = 0
    none_obj = 0
    for row in pool:
        any_obj = False
        for c in GROUNDING_CLASSES:
            if row.get(c):
                cls_counts[c] += 1
                any_obj = True
        none_obj += not any_obj
        is_safe = True
        for r in RULES:
            if row.get(f"{r}_violation") is not None:
                rule_counts[r] += 1
                is_safe = False
        safe += is_safe
    print(f"\n{YELLOW}GRPO POOL COMPOSITION{RESET}   n={n}")
    for c in GROUNDING_CLASSES:
        print(f"      P({c:<28}) = {cls_counts[c]/n:.4f}")
    print(f"      P(no object at all)            = {none_obj/n:.4f}   "
          f"{DIM}<- these produce zero gradient for object_only{RESET}")
    for r in RULES:
        print(f"      P({r:<28}) = {rule_counts[r]/n:.4f}")
    print(f"      P(safe)                        = {safe/n:.4f}")
    return ({c: cls_counts[c] / n for c in GROUNDING_CLASSES},
            {r: rule_counts[r] / n for r in RULES},
            safe / n)


def sft_stats(task: str, batch_size: int = 32):
    """Measures rare-class incidence on the task's REAL SFT split and reports how often a
    batch would contain none of each class.

    This is the number the object_only stratification decision turns on, and it cannot be
    read off the paper: Table 4 counts box *occurrences* (rebar 846, white-hard-hat 680
    across 7009 train images), not images containing at least one, and an image may hold
    several boxes. Reporting only -- nothing here changes training.
    """
    from core.config import load_task_config
    from data.loader import load_processed_dataset
    from data.oversampling import build_rare_mask_for_task, rare_class_incidence

    print("")
    print(f"{YELLOW}SFT RARE-CLASS INCIDENCE — {task}{RESET}")
    cfg = load_task_config(task)
    subdir = cfg.get("sft_dataset_subdir")
    try:
        train = load_processed_dataset(subdir=subdir)["train"]
    except Exception as e:
        _info(f"SFT split unavailable ({type(e).__name__}: {e}) — run this on ARC.")
        return []

    _info(f"split: {subdir or 'default (augmented)'}  rows: {len(train)}")
    info = rare_class_incidence(train, task)
    n = info["n_images"]
    if not info["counts"]:
        _info("this task has no rare-class axis (caption_only) — plain shuffle is correct")
        return []

    print(f"      {'class':32s} {'images':>7s} {'prev':>7s} {'batch starved':>14s} {'steps':>7s}")
    steps = n // batch_size
    for k, c in sorted(info["counts"].items(), key=lambda kv: kv[1]):
        pv = c / max(n, 1)
        starved = (1.0 - pv) ** batch_size
        colour = RED if starved > 0.20 else (YELLOW if starved > 0.05 else GREEN)
        print(f"      {k:32s} {c:>7d} {pv:>6.1%} "
              f"{colour}{starved:>13.1%}{RESET} {int(starved * steps):>7d}")

    mask = build_rare_mask_for_task(train, task)
    if mask is not None:
        r = sum(mask)
        _info(f"rare mask: {r}/{len(mask)} rows ({r / max(len(mask), 1):.1%}) -> "
              f"StratifiedRareClassSampler spreads these evenly across each epoch")
        if r / max(len(mask), 1) > 0.5:
            _info("NOTE: over half the rows are 'rare', so stratification is close to a "
                  "plain shuffle here — that is fine, just not doing much.")
    _info(f"batch_size={batch_size}, ~{steps} steps/epoch. A 'starved' batch contributes "
          "no gradient for that class.")
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=VALID_TASKS,
                    help="Task(s) to check. Repeatable. Default: all four.")
    ap.add_argument("--probe", action="store_true", help="Run the reward landscape probe.")
    ap.add_argument("--census", action="store_true", help="Run the token census.")
    ap.add_argument("--pool-stats", action="store_true",
                    help="Recompute class/rule prevalence from datasets/grpo_pool.")
    ap.add_argument("--tokenizer", default=None, help="Override the tokenizer for the census.")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows in the census.")
    ap.add_argument("--sft-stats", action="store_true",
                    help="Measure rare-class incidence on the real SFT split (needs the "
                         "dataset; run on ARC). Reporting only.")
    args = ap.parse_args()

    tasks = args.task or list(VALID_TASKS)
    _any_mode = args.probe or args.census or args.sft_stats
    run_probe = args.probe or not _any_mode
    run_census = args.census or not _any_mode

    prevalence, rule_prev, safe_rate = DEFAULT_POOL_PREVALENCE, DEFAULT_RULE_PREVALENCE, DEFAULT_SAFE_RATE
    if args.pool_stats:
        try:
            prevalence, rule_prev, safe_rate = pool_stats()
        except Exception as e:
            _info(f"pool unavailable ({type(e).__name__}) — using measured defaults")

    failures = []
    for t in tasks:
        if run_probe:
            failures += probe(t, prevalence, rule_prev, safe_rate)
        if run_census:
            failures += census(t, args.tokenizer, args.limit)
        if args.sft_stats:
            failures += sft_stats(t)

    print("\n" + "=" * 74)
    if failures:
        print(f"{RED}{len(failures)} CHECK(S) FAILED{RESET}")
        for f in failures:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print(f"{GREEN}ALL CHECKS PASSED{RESET} — reward surface and token budgets are sane.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
