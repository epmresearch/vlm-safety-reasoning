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
# sft always '_best', grpo always '_final'. A name whose suffix disagrees with
# its phase is NOT one of ours -- treated as unparseable, not force-matched.
_PHASE_SUFFIX = {"baseline": None, "sft": "best", "grpo": "final"}

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
    if m.group("suffix") != _PHASE_SUFFIX[phase]:
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

FAMILY_ORDER = ["structural", "captioning", "grounding", "violation", "reasoning", "other"]


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
    return "other"


# A curated, small subset of metric keys used for the terminal headline view
# (the full set always goes to CSV regardless). Picking a fixed short list
# keeps the terminal output scannable across a task's whole family; not
# exhaustive by design.
HEADLINE_KEYS = {
    "structural": ["structural_json_validity_rate", "structural_schema_adherence_rate"],
    "captioning": ["captioning_bertscore_f1", "captioning_meteor", "captioning_ciderd", "captioning_clipscore"],
    "grounding": ["grounding_mask_iou_all_macro_mean_tn0", "grounding_presence_f1_macro"],
    "violation": [
        "violation_identification_precision_micro",
        "violation_identification_recall_micro",
        "violation_identification_f1_micro",
        "violation_identification_precision_macro",
        "violation_identification_recall_macro",
        "violation_identification_f1_macro",
        "violation_identification_recall_rule_0",
        # Bounding-box localisation quality for violations -- a fully separate
        # metric family from object grounding, previously absent from every
        # chart (phase progression, tier scaling, master heatmap all read this
        # same list).
        "violation_grounding_mask_iou_macro_tn0",
        "violation_grounding_greedy_iou_macro_tn0",
    ],
    "reasoning": [
        "reasoning_text_similarity_bertscore_f1_macro",
        "reasoning_text_similarity_meteor_macro",
        "reasoning_text_similarity_ciderd_macro",
        "reasoning_text_similarity_clipscore_macro",
    ],
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
    return any(tok in lowered for tok in ("count", "threshold", "words", "samples"))


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
