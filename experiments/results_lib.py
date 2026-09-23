"""
Shared library for the results-comparison toolset (build_results_index.py /
compare_all.py / results_charts.py).

Pure additive code: nothing in the actual training/inference/eval pipeline
(experiments/run_{sft,grpo,inference,evaluation}.py, models/, preprocessing/,
scripts/hpc_*.sh) imports anything from this module or is affected by it.
This module only ever READS artifacts those scripts already write
(metrics.json, repair_report.json, eval_manifest.json) -- it never writes
into results/inference/ or any pipeline-owned path.

Data model
----------
One consolidated index has three long-format tables:

  runs[]    -- one row per (task, tier, phase, version) run: summary fields.
  metrics[] -- one row per metric key per run: {run_id, task, tier, phase,
               version, metric_family, metric_key, value}. Every key present
               in a real metrics.json becomes a row here automatically --
               metric_family is derived from the key's prefix, never
               hand-maintained, which is what makes it impossible to
               silently drop a whole metric family the way
               experiments/generate_comparison_csv.py's old hand-written
               METRIC_GROUPS dict did for captioning/grounding.
  repair[]  -- one row per repair_report.json entry per run: status counts,
               change-type counts, warning-type counts. This data existed in
               scripts/fetch_results.py's copy step but was read by zero
               comparison tool before this.
"""
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.tasks import TASK_REGISTRY, get_task_spec, task_capabilities

# ---------------------------------------------------------------------------
# Run-name parsing (reverse of core/naming.py::results_dir_names)
# ---------------------------------------------------------------------------

PREFIX_TO_TASK: Dict[str, str] = {spec.prefix: name for name, spec in TASK_REGISTRY.items()}

_PREFIX_ALT = "|".join(
    re.escape(p) for p in sorted(PREFIX_TO_TASK, key=len, reverse=True)
)
_RUN_NAME_RE = re.compile(
    rf"^(?P<prefix>{_PREFIX_ALT})-(?P<phase>baseline|sft|grpo)-(?P<tier>[^-]+)-"
    rf"(?P<version>v\d+)(?:_(?P<suffix>best|final))?$"
)

# core/naming.py::results_dir_names' own suffix convention: baseline has none,
# grpo always '_final', and sft '_final' as of 2026-09-10.
#
# `_best` is still ACCEPTED for sft, and only for sft. Every SFT run produced
# before the switch is named `<variant>_best` on disk, and refusing those would
# silently drop a whole phase out of an index built over existing results -- the
# runs would show up as "skipped as unparseable", which reads like a naming bug.
# Accepting both costs nothing: the suffix is not load-bearing downstream, the
# phase is already pinned by the middle segment of the name.
#
# A name whose suffix disagrees with its phase in any OTHER way (a baseline
# carrying a suffix, a grpo named `_best`) is still NOT one of ours -- treated as
# unparseable rather than force-matched.
_PHASE_SUFFIXES = {
    "baseline": {None},
    "sft": {"final", "best"},   # "best" = legacy, pre-2026-09-10
    "grpo": {"final"},
}

PHASES = ("baseline", "sft", "grpo")
PHASE_ORDER = {p: i for i, p in enumerate(PHASES)}


@dataclass(frozen=True)
class RunKey:
    task: str
    phase: str
    tier: str
    version: str

    @property
    def run_id(self) -> str:
        return f"{self.task}:{self.tier}:{self.phase}:{self.version}"


def parse_run_name(name: str) -> Optional[RunKey]:
    """Reverse of results_dir_names()[phase]. Returns None (never raises) for
    anything that doesn't match -- this is how legacy/hand-named result
    folders (e.g. 'baseline_8b', 'vo_baseline_2b') get skipped instead of
    crashing the whole index build.
    """
    m = _RUN_NAME_RE.match(name)
    if not m:
        return None
    phase = m.group("phase")
    if m.group("suffix") not in _PHASE_SUFFIXES[phase]:
        return None
    task = PREFIX_TO_TASK.get(m.group("prefix"))
    if task is None:
        return None
    return RunKey(task=task, phase=phase, tier=m.group("tier"), version=m.group("version"))


def _tier_sort_key(tier: str) -> Tuple[int, str]:
    """Sorts '2b'/'4b'/'8b' numerically; anything odd falls back to lexical."""
    m = re.match(r"(\d+)", tier)
    return (int(m.group(1)) if m else 999, tier)


def column_label(task: str, tier: str, phase: str, version: str) -> str:
    return f"{get_task_spec(task).prefix}/{tier}/{phase}/{version}"


def sorted_columns(cols) -> List[Tuple[str, str, str, str]]:
    task_order = {t: i for i, t in enumerate(TASK_REGISTRY)}
    return sorted(
        cols,
        key=lambda c: (task_order.get(c[0], 99), _tier_sort_key(c[1]), PHASE_ORDER.get(c[2], 9), c[3]),
    )


# ---------------------------------------------------------------------------
# Metric family derivation -- prefix-based, self-maintaining
# ---------------------------------------------------------------------------

FAMILY_ORDER = ["structural", "captioning", "grounding", "violation", "reasoning", "think", "other"]


def metric_family(key: str) -> str:
    if key.startswith("structural_"):
        return "structural"
    if key.startswith("captioning_"):
        return "captioning"
    if key.startswith("grounding_"):
        return "grounding"
    # Must be checked before the generic "violation_" prefix below.
    if key.startswith("reasoning_text_similarity_"):
        return "reasoning"
    if key.startswith("violation_"):
        return "violation"
    # <think>-block diagnostics (violations_think only). A distinct family rather than
    # a "structural" sub-prefix: structural_* describes the JSON payload's validity,
    # and lumping the two together would put a block-presence rate next to a schema
    # adherence rate as if they measured the same contract.
    if key.startswith("think_"):
        return "think"
    return "other"


# A curated, small subset of metric keys used for the terminal headline view
# (the full set always goes to CSV regardless). Picking a fixed short list
# keeps the terminal output scannable across a task's whole family; not
# exhaustive by design.
#
# Split into BOUNDED (genuinely [0,1] -- precision/recall/F1/IoU/validity
# rates) and UNBOUNDED (CIDEr-D commonly runs 0-3+; there is no natural
# ceiling). This split matters for PLOTTING, not just documentation: sharing
# a fixed 0-1 axis or a 0-1 heatmap color scale between the two is actively
# misleading -- a mediocre 1.8 CIDEr-D score saturates a 0-1 heatmap
# identically to a perfect 1.0, and a bar chart clipped to (0, 1.05) simply
# cuts CIDEr-D bars off outside the frame. results_charts.py never mixes the
# two groups on one axis; every unbounded metric gets its own auto-scaled chart.
BOUNDED_HEADLINE_KEYS = {
    "structural": ["structural_json_validity_rate", "structural_schema_adherence_rate"],
    "captioning": ["captioning_bertscore_f1", "captioning_meteor", "captioning_clipscore"],
    "grounding": ["grounding_mask_iou_all_macro_mean_tn0", "grounding_presence_f1_macro"],
    "violation": [
        # FIRST, deliberately. `violation_pred_positive_rate` is the fraction of
        # images on which the model flagged ANY violation; read against
        # `violation_gt_positive_rate` (0.1368 on the real test split) it is the
        # single most diagnostic number for this task, and it was absent from
        # every table before. A model flagging 64% of images has a recall that
        # looks excellent and means nothing -- see the zero-shot baselines.
        # Older metrics.json files predate these keys; the row is skipped
        # automatically when no selected run has them.
        "violation_gt_positive_rate",
        "violation_pred_positive_rate",
        "violation_identification_precision_micro",
        "violation_identification_recall_micro",
        "violation_identification_f1_micro",
        "violation_identification_precision_macro",
        "violation_identification_recall_macro",
        "violation_identification_f1_macro",
        "violation_identification_recall_rule_0",
        # Bounding-box localisation quality for violations -- a fully separate
        # metric family from object grounding.
        "violation_grounding_mask_iou_macro_tn0",
        "violation_grounding_greedy_iou_macro_tn0",
        "violation_identification_iou_conditioned_f1_micro",
        "violation_identification_iou_conditioned_f1_macro",
    ],
    "reasoning": [
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro",
    ],
}

# Demoted out of the terminal headline table, but still written to
# master_wide.csv (which dumps every key unconditionally) and still charted.
# Demotion is about what competes for attention in a scannable table, never
# about discarding data.
#
# reasoning clipscore: across the nine real vo runs it spans 0.627-0.666 -- a 6%
#   range -- while f1_micro over the same runs spans 0.129-0.487. It is scoring
#   "is this sentence about a construction site?", to which the answer is always
#   yes, for a prompt-regurgitating baseline just as much as for the best model.
# reasoning long_clipscore: bit-identical to clipscore on this task. Violation
#   reasons are one sentence, so they never chunk
#   (long_clipscore_avg_chunks_per_caption == 1.0 in every run and every rule)
#   and the two keys compute the same value.
#
# captioning_clipscore is NOT demoted: for caption_only/unified it scores a
# whole paragraph against the image, which is the benchmark's own metric.
DEMOTED_HEADLINE_KEYS = {
    "reasoning": [
        "reasoning_text_similarity_clipscore_macro",
        "reasoning_text_similarity_long_clipscore_micro",
    ],
}

BOUNDED_HEADLINE_KEYS["think"] = [
    # All three are [0,1] rates. Read them in this order: a low present_rate means the
    # model stopped emitting blocks at all; a low closed_rate means completions are
    # being truncated mid-block; a low agreement_rate is the interesting failure -- the
    # stated reasoning does not describe the answer the model then gave.
    "think_block_present_rate",
    "think_block_closed_rate",
    "think_verdict_json_agreement_rate",
]

UNBOUNDED_HEADLINE_KEYS = {
    "captioning": ["captioning_ciderd"],
    "reasoning": [
        "reasoning_text_similarity_ciderd_macro",
        # LLM judge total, 0-6 (the dataset paper's Table 8 units). Not [0,1], so it
        # must not share the bounded 0-1.05 axis -- that is exactly how CIDEr-D bars
        # were once silently clipped off the top of the frame. results_charts.py gives
        # it a fixed 0-6 axis via METRIC_Y_CEILING instead of auto-scaling.
        "reasoning_llm_judge_total_macro",
    ],
}

# Terminal/CSV consumers don't have a scale problem (they print raw numbers),
# so they read the union of both -- only the chart code needs the split above.
HEADLINE_KEYS = {
    fam: BOUNDED_HEADLINE_KEYS.get(fam, []) + UNBOUNDED_HEADLINE_KEYS.get(fam, [])
    for fam in set(BOUNDED_HEADLINE_KEYS) | set(UNBOUNDED_HEADLINE_KEYS)
}

# What results_charts.py plots: the headline set PLUS the demoted keys, so
# demoting a metric from the terminal table never silently deletes a chart.
CHARTABLE_BOUNDED_KEYS = {
    fam: BOUNDED_HEADLINE_KEYS.get(fam, []) + DEMOTED_HEADLINE_KEYS.get(fam, [])
    for fam in set(BOUNDED_HEADLINE_KEYS) | set(DEMOTED_HEADLINE_KEYS)
}

# Sample-size / support keys shown alongside the scores they underpin.
# Every reasoning metric is conditioned on the model's OWN true positives, so
# two runs are literally scored on different image populations: on the real vo
# runs, 4b-grpo's rule_3 reasoning is an average over 22 images and 8b-grpo's
# over 8. Printing the count beside the score is what makes that visible.
SUPPORT_KEYS = {
    "reasoning": (
        ["reasoning_text_similarity_scored_count_micro"]
        + [f"reasoning_text_similarity_scored_count_rule_{i}" for i in (1, 2, 3, 4)]
        # The judge scores the SAME true-positive list (metrics_reasoning.
        # collect_tp_reason_pairs), so these equal the rows above whenever the judge
        # ran. The unparsed count is shown because unparseable replies score 0/0/0 and
        # pull the judge means down.
        + ["reasoning_llm_judge_scored_count_micro", "reasoning_llm_judge_unparsed_count_micro"]
        + [f"reasoning_llm_judge_scored_count_rule_{i}" for i in (1, 2, 3, 4)]
    ),
    "think": (
        ["think_block_present_count", "think_verdict_json_comparable_count",
         "think_total_samples_count"]
        + [f"think_verdict_json_comparable_count_rule_{i}" for i in (1, 2, 3, 4)]
    ),
    "violation": (
        ["violation_gt_positive_image_count", "violation_pred_positive_image_count"]
        + [f"violation_gt_count_rule_{i}" for i in (1, 2, 3, 4)]
        + [f"violation_pred_count_rule_{i}" for i in (1, 2, 3, 4)]
        + [f"violation_tp_count_rule_{i}" for i in (1, 2, 3, 4)]
        + [f"violation_grounding_scored_count_rule_{i}" for i in (1, 2, 3, 4)]
    ),
}

# metric key -> the key holding the divisor its macro was taken over. Rendered
# inline in the headline table as "0.7817 (n=2)", so a macro over 2 rules can
# never be read as a macro over 4.
MACRO_N_RULES_KEY = {
    "reasoning_text_similarity_bertscore_f1_macro": "reasoning_text_similarity_bertscore_f1_macro_n_rules",
    "reasoning_text_similarity_meteor_macro": "reasoning_text_similarity_meteor_macro_n_rules",
    "reasoning_text_similarity_ciderd_macro": "reasoning_text_similarity_ciderd_macro_n_rules",
    "reasoning_text_similarity_clipscore_macro": "reasoning_text_similarity_clipscore_macro_n_rules",
    "violation_grounding_mask_iou_macro_tn0": "violation_grounding_mask_iou_macro_tn0_n_rules",
    "violation_grounding_greedy_iou_macro_tn0": "violation_grounding_greedy_iou_macro_tn0_n_rules",
    "reasoning_llm_judge_total_macro": "reasoning_llm_judge_total_macro_n_rules",
    "reasoning_llm_judge_relevance_macro": "reasoning_llm_judge_relevance_macro_n_rules",
    "reasoning_llm_judge_equivalence_macro": "reasoning_llm_judge_equivalence_macro_n_rules",
    "reasoning_llm_judge_specificity_macro": "reasoning_llm_judge_specificity_macro_n_rules",
}

GROUNDING_CLASSES = ["excavator", "rebar", "worker_with_white_hard_hat"]
VIOLATION_RULES = ["rule_0", "rule_1", "rule_2", "rule_3", "rule_4"]


# ---------------------------------------------------------------------------
# On-disk layout resolution -- mirrors scripts/fetch_results.py::WANTED,
# the already-proven dual-candidate-path pattern for the real ARC tree.
# ---------------------------------------------------------------------------

_ARC_CANDIDATES = {
    "metrics.json": ["evaluation_results/metrics.json", "repair_applied/evaluation_results/metrics.json"],
    "repair_report.json": ["repair_applied/repair_report.json", "repair_report.json"],
    "eval_manifest.json": ["evaluation_results/eval_manifest.json", "repair_applied/evaluation_results/eval_manifest.json"],
}
_FLAT_CANDIDATES = {
    "metrics.json": ["metrics.json"],
    "repair_report.json": ["repair_report.json"],
    "eval_manifest.json": ["eval_manifest.json"],
}


def candidates_for(layout: str) -> Dict[str, List[str]]:
    if layout == "arc":
        return _ARC_CANDIDATES
    if layout == "flat":
        return _FLAT_CANDIDATES
    if layout == "auto":
        return {k: _ARC_CANDIDATES[k] + _FLAT_CANDIDATES[k] for k in _ARC_CANDIDATES}
    raise ValueError(f"Unknown layout: {layout!r}. Expected 'arc', 'flat', or 'auto'.")


def _load_json_candidates(run_dir: Path, candidates: List[str]) -> Tuple[Optional[dict], Optional[str]]:
    for rel in candidates:
        p = run_dir / rel
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), rel
            except (json.JSONDecodeError, OSError) as e:
                # A run mid-write (job still running) can leave a truncated
                # file. Treat as absent rather than aborting the whole index.
                return None, f"{rel} (unreadable: {e})"
    return None, None


def find_run_dirs(source: Path) -> List[str]:
    if not source.is_dir():
        return []
    return sorted(p.name for p in source.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Per-run indexing
# ---------------------------------------------------------------------------

def index_one_run(run_dir: Path, key: RunKey, layout: str) -> dict:
    cands = candidates_for(layout)
    metrics_data, metrics_src = _load_json_candidates(run_dir, cands["metrics.json"])
    repair_data, repair_src = _load_json_candidates(run_dir, cands["repair_report.json"])
    manifest_data, _ = _load_json_candidates(run_dir, cands["eval_manifest.json"])

    run_id = key.run_id
    metrics_rows = []
    if metrics_data:
        for mkey, val in metrics_data.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            metrics_rows.append({
                "run_id": run_id, "task": key.task, "tier": key.tier,
                "phase": key.phase, "version": key.version,
                "metric_family": metric_family(mkey), "metric_key": mkey, "value": val,
            })

    repair_rows = []
    if repair_data:
        status_pct = repair_data.get("status_percentages", {})
        for name, count in repair_data.get("status_counts", {}).items():
            repair_rows.append({
                "run_id": run_id, "task": key.task, "tier": key.tier,
                "phase": key.phase, "version": key.version,
                "kind": "status", "name": name,
                "count": count, "pct_of_total": status_pct.get(name),
                "total_occurrences": None, "records_affected": None,
            })
        for kind, group_key in (("change_type", "change_type_summary"), ("warning_type", "warning_type_summary")):
            for name, d in repair_data.get(group_key, {}).items():
                repair_rows.append({
                    "run_id": run_id, "task": key.task, "tier": key.tier,
                    "phase": key.phase, "version": key.version,
                    "kind": kind, "name": name,
                    "count": None, "pct_of_total": None,
                    "total_occurrences": d.get("total_occurrences"),
                    "records_affected": d.get("records_affected"),
                })

    status_pct = (repair_data or {}).get("status_percentages", {})
    eval_config = (manifest_data or {}).get("configuration", {})
    run_summary = {
        "run_id": run_id, "task": key.task, "tier": key.tier,
        "phase": key.phase, "version": key.version,
        "capabilities": sorted(task_capabilities(key.task)),
        "has_metrics": metrics_data is not None,
        "has_repair_report": repair_data is not None,
        "has_eval_manifest": manifest_data is not None,
        "metrics_source": metrics_src,
        "repair_source": repair_src,
        "total_samples": (metrics_data or {}).get("structural_total_samples_count"),
        "json_validity_rate": (metrics_data or {}).get("structural_json_validity_rate"),
        "schema_adherence_rate": (metrics_data or {}).get("structural_schema_adherence_rate"),
        "repair_valid_raw_pct": status_pct.get("valid_raw"),
        "repair_fixed_pct": status_pct.get("fixed_valid"),
        "repair_invalid_json_pct": status_pct.get("invalid_json"),
        "repair_invalid_schema_pct": status_pct.get("invalid_schema"),
        "eval_experiment": eval_config.get("experiment"),
        # Packed per-image violation outcomes (base64; see
        # evaluation/metrics_violations.py). A STRING, so it is skipped by the
        # numeric filter that builds metrics_rows above and never becomes a
        # metric row -- it is carried on the run summary instead, which is what
        # lets compare_all.py run a paired bootstrap from index.json alone
        # instead of needing the multi-hundred-MB predictions files. ~4 KB per
        # run. None for runs evaluated before this key existed; the bootstrap
        # is then skipped with a message rather than failing.
        "violation_per_image_outcomes_b64": (metrics_data or {}).get("violation_per_image_outcomes_b64"),
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {"run": run_summary, "metrics": metrics_rows, "repair": repair_rows}


def discover_and_index(source: Path, layout: str, tasks=None, tiers=None, versions=None):
    """Walks every subdirectory of `source`, parses whatever matches the
    <prefix>-<phase>-<tier>-<version>[_best|_final] convention, indexes it,
    and skips (never crashes on) anything that doesn't match or filters out.
    """
    names = find_run_dirs(source)
    runs, metrics, repair, skipped = [], [], [], []
    counted = Counter()
    for name in names:
        key = parse_run_name(name)
        if key is None:
            skipped.append(name)
            continue
        if tasks and key.task not in tasks:
            continue
        if tiers and key.tier not in tiers:
            continue
        if versions and key.version not in versions:
            continue
        result = index_one_run(source / name, key, layout)
        runs.append(result["run"])
        metrics.extend(result["metrics"])
        repair.extend(result["repair"])
        counted[(key.task, key.tier, key.phase, key.version)] += 1
    return runs, metrics, repair, skipped, counted


# ---------------------------------------------------------------------------
# Index I/O -- atomic write, matching the tmp+os.replace pattern already used
# by core/run_manifest.py elsewhere in this repo.
# ---------------------------------------------------------------------------

def write_index(out_path: Path, runs: list, metrics: list, repair: list,
                 skipped: list, layout: str, source: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "layout": layout,
        "skipped_names": skipped,
        "runs": runs,
        "metrics": metrics,
        "repair": repair,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def load_index(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting helpers shared by compare_all.py's terminal + CSV output
# (ported from experiments/generate_comparison_csv.py, generalized).
# ---------------------------------------------------------------------------

def fmt_val(val):
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) >= 10 else f"{val:.6f}"
    return str(val)


def fmt_delta(delta):
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    if isinstance(delta, int):
        return f"{sign}{delta}"
    return f"{sign}{delta:.6f}"


def fmt_pct(baseline_val, new_val):
    if baseline_val is None or new_val is None:
        return "N/A"
    if baseline_val == 0:
        if new_val == 0:
            return "0.00%"
        return "+inf%" if new_val > 0 else "-inf%"
    pct = ((new_val - baseline_val) / abs(baseline_val)) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def best_phase(baseline_val, sft_val, grpo_val):
    vals = {"BASELINE": baseline_val, "SFT": sft_val, "GRPO": grpo_val}
    valid = {k: v for k, v in vals.items() if v is not None}
    if not valid:
        return "N/A"
    return max(valid, key=valid.get)


def is_non_comparable_key(key: str) -> bool:
    """Count/threshold/word-length/sample-size keys aren't 'higher is better'
    metrics -- excluded from win-tallying, same exclusion generate_comparison_csv.py
    used, generalized to any family instead of a hand-picked list."""
    lowered = key.lower()
    # Beyond counts/thresholds/lengths, four more shapes are not "higher is better":
    #   n_rules        -- a macro's divisor, not a score
    #   positive_rate  -- violation_{gt,pred}_positive_rate: the GT rate is constant
    #                     across phases (every tie "won" by BASELINE) and a higher
    #                     predicted flag rate is over-flagging, not an improvement
    #   unparsed       -- LLM-judge failures, where higher is worse
    #   failure        -- violation_prediction_failure_rate, where higher is worse
    # Tallying any of these skews the delta CSV's SUMMARY win counts.
    return any(tok in lowered for tok in (
        "count", "threshold", "words", "samples",
        "n_rules", "positive_rate", "unparsed", "failure",
    ))


# ---------------------------------------------------------------------------
# Paired bootstrap over per-image outcomes
#
# WHY THIS EXISTS. Three of the four safety rules have 25, 63 and 24 positive
# examples in the entire 3004-image test split. A per-rule F1 there carries a
# 95% interval up to 0.33 wide, and violation_identification_f1_macro averages
# four such numbers -- which is why phase and tier rankings appear to flip
# between runs that are statistically indistinguishable. Point estimates alone
# cannot tell "GRPO helped" from "GRPO did nothing"; on the real vo runs the
# SFT->GRPO delta is +0.018 (p=0.076) at 4b and +0.012 (p=0.149) at 8b, both
# inside the noise, while baseline->SFT is +0.24 (p<0.001).
#
# Resampling is over IMAGES (the independent unit), paired: every run is scored
# on the same resampled index set, so the between-run correlation is preserved
# and the interval is on the DIFFERENCE rather than on two separate estimates.
# ---------------------------------------------------------------------------

N_RULE_BITS = 4


def decode_outcomes(b64: Optional[str]):
    """Unpacks metrics_violations.py's per-image outcome vector.

    Returns (pred_masks, gt_masks) as two equal-length lists of small ints, or
    None when the run predates the key.
    """
    if not b64:
        return None
    import base64
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    return [b >> 4 for b in raw], [b & 0x0F for b in raw]


def contribution_matrices(pred, gt):
    """Per-image, per-rule TP / FP / FN indicators, each shaped (n_images, 4).

    This is the whole trick that makes the bootstrap affordable. Every metric
    below is a ratio of SUMS of these indicators, and resampling images only
    changes which rows are summed -- never the row values. So the per-image
    contributions are computed once, and each resample is a sum, not a rescoring.
    """
    n = len(pred)
    tp = [[0] * N_RULE_BITS for _ in range(n)]
    fp = [[0] * N_RULE_BITS for _ in range(n)]
    fn = [[0] * N_RULE_BITS for _ in range(n)]
    for i in range(n):
        p, g = pred[i], gt[i]
        for b in range(N_RULE_BITS):
            pb, gb = (p >> b) & 1, (g >> b) & 1
            if pb and gb:
                tp[i][b] = 1
            elif pb:
                fp[i][b] = 1
            elif gb:
                fn[i][b] = 1
    return tp, fp, fn


def _f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else 0.0


def _metric_from_totals(tp_r, fp_r, fn_r, metric):
    """tp_r/fp_r/fn_r are per-rule totals (length 4)."""
    if metric == "f1_micro":
        return _f1(sum(tp_r), sum(fp_r), sum(fn_r))
    if metric == "f1_macro":
        return sum(_f1(tp_r[b], fp_r[b], fn_r[b]) for b in range(N_RULE_BITS)) / N_RULE_BITS
    raise ValueError(f"Unknown metric {metric!r}")


BOOTSTRAP_METRICS = ("f1_micro", "f1_macro")


def bootstrap_indices(n: int, n_boot: int, seed: int = 0):
    """Paired resample index sets, as a plain list of lists.

    Deterministic: the same index.json with the same --bootstrap/--seed always
    reproduces the same intervals. Every run is scored on these SAME index sets,
    which is what makes the comparisons paired -- the between-run correlation is
    preserved, so the interval lands on the difference rather than on two
    independently wandering estimates.
    """
    import random
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]


def bootstrap_series(pred, gt, boot_idx, metrics=BOOTSTRAP_METRICS):
    """{metric: (point_estimate, [value per resample])} for ONE run.

    Computed once per run and reused for every comparison that run takes part
    in -- a paired delta is then an elementwise subtraction of two series, not a
    second pass over the data.

    Uses numpy when it is importable (the ARC training env and the local venv
    both have it) and falls back to pure Python otherwise, so the tool keeps
    working on a bare interpreter -- just more slowly.
    """
    tp, fp, fn = contribution_matrices(pred, gt)
    out = {}
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        TP = np.asarray(tp, dtype=np.int32)
        FP = np.asarray(fp, dtype=np.int32)
        FN = np.asarray(fn, dtype=np.int32)
        IDX = np.asarray(boot_idx, dtype=np.int64) if boot_idx else np.empty((0, 0), dtype=np.int64)
        # (B, 4) per-rule totals for each resample, in one pass per array.
        if IDX.size:
            tp_b = TP[IDX].sum(axis=1)
            fp_b = FP[IDX].sum(axis=1)
            fn_b = FN[IDX].sum(axis=1)
        else:
            tp_b = fp_b = fn_b = np.empty((0, N_RULE_BITS), dtype=np.int64)
        tp_pt, fp_pt, fn_pt = TP.sum(axis=0), FP.sum(axis=0), FN.sum(axis=0)
        for m in metrics:
            point = _metric_from_totals(list(tp_pt), list(fp_pt), list(fn_pt), m)
            series = [
                _metric_from_totals(list(tp_b[k]), list(fp_b[k]), list(fn_b[k]), m)
                for k in range(len(tp_b))
            ]
            out[m] = (point, series)
        return out

    def totals(idx):
        tp_r = [0] * N_RULE_BITS
        fp_r = [0] * N_RULE_BITS
        fn_r = [0] * N_RULE_BITS
        for i in idx:
            ti, fi, ni = tp[i], fp[i], fn[i]
            for b in range(N_RULE_BITS):
                tp_r[b] += ti[b]
                fp_r[b] += fi[b]
                fn_r[b] += ni[b]
        return tp_r, fp_r, fn_r

    all_totals = [totals(idx) for idx in boot_idx]
    pt = totals(range(len(pred)))
    for m in metrics:
        out[m] = (_metric_from_totals(*pt, m), [_metric_from_totals(*t, m) for t in all_totals])
    return out


def ci_from_series(series, alpha=0.05):
    """(lo, hi) percentile interval, or (None, None) when there are no resamples."""
    if not series:
        return None, None
    vals = sorted(series)
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


def paired_delta(series_a, series_b, alpha=0.05):
    """(lo, hi, p) for B minus A. Both series must come from the same boot_idx.

    p is one-sided: the fraction of resamples in which the observed improvement
    did NOT hold (delta <= 0).
    """
    if not series_a or not series_b:
        return None, None, None
    diffs = [b - a for a, b in zip(series_a, series_b)]
    lo, hi = ci_from_series(diffs, alpha)
    return lo, hi, sum(1 for d in diffs if d <= 0) / len(diffs)


def significance_marker(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def print_table(headers, rows, title=None):
    """Minimal stdlib-only aligned text table -- no pandas/tabulate dependency,
    so this runs identically on the ARC login node or locally with nothing
    extra to install."""
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        srow = []
        for v in row:
            if v is None:
                srow.append("")
            elif isinstance(v, float):
                srow.append(f"{v:.4f}")
            else:
                srow.append(str(v))
        str_rows.append(srow)
        for i, cell in enumerate(srow):
            widths[i] = max(widths[i], len(cell))
    if title:
        print(f"\n=== {title} ===")
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for srow in str_rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(srow)))
