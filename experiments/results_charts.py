"""
Chart generation for compare_all.py. Pure additive, read-only against the
results tree (see results_lib.py's module docstring for the same guarantee).

Design rules (rewritten 2026-09-08 after real-data feedback on the first
version):

1. ONE METRIC PER CHART, not several unrelated metrics crammed onto one axis.
   Different metric families have genuinely different natural ranges (F1/IoU
   are [0,1]; CIDEr-D commonly runs 0-3+) -- sharing an axis between them
   either clips the larger one off the frame (a hardcoded 0-1 ylim) or makes
   the smaller ones invisible next to it. results_lib.py's
   BOUNDED_HEADLINE_KEYS / UNBOUNDED_HEADLINE_KEYS split is what this module
   uses to keep them apart; every chart here is either fixed to (0, 1.05)
   because its metric is genuinely bounded there, or left auto-scaled because
   it isn't -- never a chart mixing both kinds of series.
2. EVERY CHART IS A REAL COMPARISON, not a single-run snapshot. A chart whose
   only content is "here are 5 metrics for this one run" doesn't let you see
   whether GRPO helped, or whether 8b beats 2b -- the two questions this
   toolset exists to answer. So every chart's bar-grouping dimension is tier
   or phase (or both, via one chart per phase showing all tiers, and a
   companion chart per tier showing all phases) -- never "no comparison axis
   at all."
3. Defensive by design: comparisons here are routinely run over a PARTIAL set
   of tasks/tiers/phases (one tier just finished, another is mid-training),
   so every chart function skips itself -- printing one line explaining why
   -- rather than crashing, whenever it doesn't have enough data to be
   meaningful.

matplotlib is optional. If it isn't installed (most likely scenario: running
build_results_index.py + compare_all.py directly on the ARC login node in a
minimal env), MATPLOTLIB_AVAILABLE is False and generate_all_charts() prints
one message and returns immediately -- tables and CSVs from compare_all.py
are entirely unaffected, since they need no plotting dependency at all.
"""
from collections import defaultdict
from pathlib import Path

from experiments.results_lib import (
    BOUNDED_HEADLINE_KEYS, FAMILY_ORDER, GROUNDING_CLASSES, PHASE_ORDER,
    UNBOUNDED_HEADLINE_KEYS, VIOLATION_RULES, _tier_sort_key, column_label,
    sorted_columns,
)
from core.tasks import get_task_spec, task_has, CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- must work over SSH/MobaXterm with no display
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

DPI = 150


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _lut(metrics_rows):
    """(task,tier,phase,version,key) -> value"""
    d = {}
    for r in metrics_rows:
        d[(r["task"], r["tier"], r["phase"], r["version"], r["metric_key"])] = r["value"]
    return d


def _bar(x_labels, series, title, ylabel, path, y_bounded=True):
    """Generic grouped-bar renderer used by every chart below.

    series = [(series_label, [value_or_None per x_label]), ...].

    y_bounded=True fixes the axis to (0, 1.05) -- ONLY correct for a metric
    that is genuinely capped at 1 (precision/recall/F1/IoU/validity rates).
    y_bounded=False leaves the axis auto-scaled with headroom for the value
    labels -- required for CIDEr-D and anything else without a natural
    ceiling; forcing those onto a 0-1 axis silently clips the bars off the
    top of the chart instead of erroring, which is exactly the bug this
    rewrite fixes.
    """
    fig, ax = plt.subplots(figsize=(max(7, len(x_labels) * 1.6), 6))
    n = len(series)
    width = 0.8 / max(n, 1)
    x = range(len(x_labels))
    colors = cm.get_cmap("Set1", max(n, 3))
    for i, (label, vals) in enumerate(series):
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        plot_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(offsets, plot_vals, width=width * 0.9, label=label, color=colors(i))
        for b, v in zip(bars, vals):
            if v is not None:
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    if y_bounded:
        ax.set_ylim(0, 1.05)
    else:
        ax.margins(y=0.15)  # headroom for value labels on an auto-scaled axis
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 1. Per-metric scaling: ONE metric, x=tier, bars=phase. Replaces the old
#    multi-metric phase_progression/tier_scaling charts -- every tier x phase
#    comparison now lives on its own correctly-scaled figure.
# ---------------------------------------------------------------------------

def _metric_scaling(lut, columns, out_dir, key_source, y_bounded, subdir) -> list:
    written = []
    by_task_version = defaultdict(set)
    for c in columns:
        by_task_version[(c[0], c[3])].add(c)

    for (task, version), cols in by_task_version.items():
        prefix = get_task_spec(task).prefix
        tiers = sorted({c[1] for c in cols}, key=_tier_sort_key)
        phases = sorted({c[2] for c in cols}, key=lambda p: PHASE_ORDER.get(p, 9))
        keys = []
        for fam in FAMILY_ORDER:
            keys.extend(key_source.get(fam, []))
        for key in keys:
            series = []
            for phase in phases:
                vals = [lut.get((task, tier, phase, version, key)) for tier in tiers]
                if all(v is None for v in vals):
                    continue
                series.append((phase.upper(), vals))
            if not series:
                continue
            path = out_dir / prefix / subdir / f"{key}_{prefix}_{version}.png"
            _bar(tiers, series, f"{prefix} / {version} -- {key}", key, path, y_bounded=y_bounded)
            written.append(path)
    return written


def chart_metric_scaling(lut, columns, out_dir) -> list:
    """One chart per bounded headline metric (structural/captioning/grounding/
    violation/reasoning): x=tier, bars=phase."""
    return _metric_scaling(lut, columns, out_dir, BOUNDED_HEADLINE_KEYS, y_bounded=True, subdir="scaling")


def chart_metric_scaling_unbounded(lut, columns, out_dir) -> list:
    """Same, but for CIDEr-D (and anything else without a [0,1] ceiling) --
    kept structurally separate so it never shares an axis with a bounded
    metric."""
    return _metric_scaling(lut, columns, out_dir, UNBOUNDED_HEADLINE_KEYS, y_bounded=False, subdir="scaling")


# ---------------------------------------------------------------------------
# 2. Per-rule, single metric, both comparison directions: one chart per
#    PHASE showing all tiers, plus a companion chart per TIER showing all
#    phases. Covers violation identification (plain + strict/IoU-conditioned),
#    violation bounding-box grounding, and per-rule reasoning quality --
#    every one of these was either absent entirely or crammed into a
#    multi-metric single-run snapshot before this rewrite.
# ---------------------------------------------------------------------------

def _rule_values(lut, task, tier, phase, version, rules, key_template):
    vals = [lut.get((task, tier, phase, version, key_template.format(rule=r))) for r in rules]
    return None if all(v is None for v in vals) else vals


def _by_rule_single_metric(lut, columns, out_dir, cap, rules, key_template,
                            slug, ylabel, subdir, y_bounded) -> list:
    written = []
    by_task_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], cap):
            by_task_version[(c[0], c[3])].add(c)

    for (task, version), cols in by_task_version.items():
        prefix = get_task_spec(task).prefix
        tiers = sorted({c[1] for c in cols}, key=_tier_sort_key)
        phases = sorted({c[2] for c in cols}, key=lambda p: PHASE_ORDER.get(p, 9))

        for phase in phases:
            series = []
            for tier in tiers:
                vals = _rule_values(lut, task, tier, phase, version, rules, key_template)
                if vals is not None:
                    series.append((tier, vals))
            if series:
                path = out_dir / prefix / subdir / f"{slug}_across_tiers_{prefix}_{phase}_{version}.png"
                _bar(rules, series, f"{prefix} / {phase} / {version} -- {ylabel} by rule, across tiers",
                     ylabel, path, y_bounded)
                written.append(path)

        for tier in tiers:
            series = []
            for phase in phases:
                vals = _rule_values(lut, task, tier, phase, version, rules, key_template)
                if vals is not None:
                    series.append((phase.upper(), vals))
            if series:
                path = out_dir / prefix / subdir / f"{slug}_across_phases_{prefix}_{tier}_{version}.png"
                _bar(rules, series, f"{prefix} / {tier} / {version} -- {ylabel} by rule, across phases",
                     ylabel, path, y_bounded)
                written.append(path)
    return written


_VG_RULES = VIOLATION_RULES[1:]  # rule_1..4 only -- no box/reasoning to ground on a safe image

# (capability, rules, key_template, file_slug, y_axis_label, output_subdir, y_bounded)
PER_RULE_METRIC_SPECS = [
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_precision_{rule}", "precision", "Precision", "violation_id", True),
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_recall_{rule}", "recall", "Recall", "violation_id", True),
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_f1_{rule}", "f1", "F1", "violation_id", True),
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_iou_conditioned_precision_{rule}",
     "strict_precision", "Precision (IoU-conditioned)", "violation_id_strict", True),
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_iou_conditioned_recall_{rule}",
     "strict_recall", "Recall (IoU-conditioned)", "violation_id_strict", True),
    (CAP_VIOLATIONS, VIOLATION_RULES, "violation_identification_iou_conditioned_f1_{rule}",
     "strict_f1", "F1 (IoU-conditioned)", "violation_id_strict", True),
    (CAP_VIOLATIONS, _VG_RULES, "violation_grounding_mask_iou_{rule}_tn0", "mask_iou", "Mask IoU", "violation_grounding", True),
    (CAP_VIOLATIONS, _VG_RULES, "violation_grounding_greedy_iou_{rule}_tn0", "greedy_iou", "Greedy IoU", "violation_grounding", True),
    (CAP_VIOLATIONS, _VG_RULES, "reasoning_text_similarity_bertscore_f1_{rule}", "bertscore_f1", "BERTScore F1", "reasoning", True),
    (CAP_VIOLATIONS, _VG_RULES, "reasoning_text_similarity_meteor_{rule}", "meteor", "METEOR", "reasoning", True),
    (CAP_VIOLATIONS, _VG_RULES, "reasoning_text_similarity_clipscore_{rule}", "clipscore", "CLIPScore", "reasoning", True),
    # Unbounded -- must never share a chart (or axis) with the bounded metrics above.
    (CAP_VIOLATIONS, _VG_RULES, "reasoning_text_similarity_ciderd_{rule}", "ciderd", "CIDEr-D", "reasoning", False),
]


def chart_per_rule_metrics(lut, columns, out_dir) -> list:
    written = []
    for cap, rules, key_template, slug, ylabel, subdir, y_bounded in PER_RULE_METRIC_SPECS:
        written.extend(_by_rule_single_metric(lut, columns, out_dir, cap, rules, key_template,
                                               slug, ylabel, subdir, y_bounded))
    return written


# ---------------------------------------------------------------------------
# 3. Overview charts -- birds-eye views across every run at once. Kept
#    separate from the per-metric charts above; useful for spotting outliers
#    quickly, not for reading exact tier/phase deltas.
# ---------------------------------------------------------------------------

def chart_repair_status(runs, out_dir) -> list:
    """One stacked bar chart per version: x = run (task/tier/phase), stacked
    bars = valid_raw / fixed_valid / invalid_json / invalid_schema %
    (percentages -- always [0,100], no scale-mixing risk)."""
    written = []
    by_version = defaultdict(list)
    for r in runs:
        if r.get("repair_valid_raw_pct") is None and r.get("repair_fixed_pct") is None:
            continue
        by_version[r["version"]].append(r)

    for version, rows in by_version.items():
        rows = sorted(rows, key=lambda r: (get_task_spec(r["task"]).prefix, _tier_sort_key(r["tier"]), PHASE_ORDER.get(r["phase"], 9)))
        labels = [column_label(r["task"], r["tier"], r["phase"], r["version"]) for r in rows]
        valid_raw = [r.get("repair_valid_raw_pct") or 0 for r in rows]
        fixed = [r.get("repair_fixed_pct") or 0 for r in rows]
        inv_json = [r.get("repair_invalid_json_pct") or 0 for r in rows]
        inv_schema = [r.get("repair_invalid_schema_pct") or 0 for r in rows]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 6))
        x = range(len(labels))
        ax.bar(x, valid_raw, label="valid_raw (no repair needed)", color="#2ca02c")
        ax.bar(x, fixed, bottom=valid_raw, label="fixed_valid (repaired)", color="#1f77b4")
        bottom2 = [a + b for a, b in zip(valid_raw, fixed)]
        ax.bar(x, inv_json, bottom=bottom2, label="invalid_json (unparseable)", color="#ff7f0e")
        bottom3 = [a + b for a, b in zip(bottom2, inv_json)]
        ax.bar(x, inv_schema, bottom=bottom3, label="invalid_schema (still broken)", color="#d62728")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("% of predictions")
        ax.set_title(f"Structural repair outcome breakdown -- {version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / f"repair_status_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_violation_heatmap(lut, columns, out_dir) -> list:
    """Per-rule F1 heatmap, rows = rule_0..4, columns = every violation-capable
    run present (any task with CAP_VIOLATIONS). F1 is bounded [0,1] so a
    fixed-scale heatmap is valid here -- a birds-eye view, not a replacement
    for the focused per-rule/per-phase-or-tier charts above."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], CAP_VIOLATIONS):
            by_version[c[3]].add(c)

    for version, cols in by_version.items():
        cols = sorted_columns(cols)
        if not cols:
            continue
        data = []
        for rule in VIOLATION_RULES:
            key = f"violation_identification_f1_{rule}"
            data.append([lut.get((c[0], c[1], c[2], c[3], key)) for c in cols])

        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), 5))
        display = [[v if v is not None else float("nan") for v in row] for row in data]
        im = ax.imshow(display, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(VIOLATION_RULES)))
        ax.set_yticklabels(VIOLATION_RULES)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([column_label(*c) for c in cols], rotation=60, ha="right", fontsize=8)
        for i in range(len(VIOLATION_RULES)):
            for j in range(len(cols)):
                v = data[i][j]
                if v is not None:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="F1")
        ax.set_title(f"Violation identification F1 by rule (overview) -- {version}")
        path = out_dir / f"violation_f1_heatmap_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_grounding_per_class(lut, columns, out_dir) -> list:
    """Per-class grounding IoU bar chart, grouped by run, for any object-
    capable task (unified AND object_only together)."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], CAP_OBJECTS):
            by_version[c[3]].add(c)

    for version, cols in by_version.items():
        cols = sorted_columns(cols)
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(max(9, len(cols) * 0.9), 6))
        n_classes = len(GROUNDING_CLASSES)
        width = 0.8 / n_classes
        x = range(len(cols))
        colors = cm.get_cmap("Set2", n_classes)
        for i, cls in enumerate(GROUNDING_CLASSES):
            key = f"grounding_mask_iou_all_macro_{cls}_tn0"
            vals = [lut.get((c[0], c[1], c[2], c[3], key)) for c in cols]
            offsets = [xi + (i - (n_classes - 1) / 2) * width for xi in x]
            plot_vals = [v if v is not None else 0 for v in vals]
            ax.bar(offsets, plot_vals, width=width * 0.9, label=cls, color=colors(i))
        ax.set_xticks(list(x))
        ax.set_xticklabels([column_label(*c) for c in cols], rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mask IoU (macro, tn0)")
        ax.set_title(f"Per-class grounding IoU (overview) -- {version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / f"grounding_per_class_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_captioning_quality(lut, columns, out_dir) -> list:
    """Grouped bar chart of BOUNDED caption-quality metrics only (BERTScore/
    METEOR/CLIPScore) -- CIDEr-D is deliberately excluded here (see
    chart_metric_scaling_unbounded) since it is not [0,1]-bounded and would
    either dominate or get clipped by this chart's fixed axis."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], CAP_CAPTION):
            by_version[c[3]].add(c)

    metrics_ = BOUNDED_HEADLINE_KEYS["captioning"]
    for version, cols in by_version.items():
        cols = sorted_columns(cols)
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(max(9, len(cols) * 0.9), 6))
        n = len(metrics_)
        width = 0.8 / n
        x = range(len(cols))
        colors = cm.get_cmap("Dark2", n)
        for i, key in enumerate(metrics_):
            vals = [lut.get((c[0], c[1], c[2], c[3], key)) for c in cols]
            offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
            plot_vals = [v if v is not None else 0 for v in vals]
            ax.bar(offsets, plot_vals, width=width * 0.9, label=key.replace("captioning_", ""), color=colors(i))
        ax.set_xticks(list(x))
        ax.set_xticklabels([column_label(*c) for c in cols], rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Captioning quality, bounded metrics (overview) -- {version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / f"captioning_quality_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_master_heatmap(lut, columns, out_dir) -> list:
    """One big at-a-glance heatmap per version: rows = every BOUNDED headline
    metric across every family, columns = every run present. Restricted to
    BOUNDED_HEADLINE_KEYS -- an unbounded metric (CIDEr-D) on a fixed 0-1
    color scale would saturate at "perfect green" for any score above 1,
    indistinguishable from an actually perfect score."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        by_version[c[3]].add(c)

    all_bounded_keys = []
    for fam in FAMILY_ORDER:
        all_bounded_keys.extend(BOUNDED_HEADLINE_KEYS.get(fam, []))

    for version, cols in by_version.items():
        cols = sorted_columns(cols)
        if not cols:
            continue
        keys = [k for k in all_bounded_keys if any(lut.get((c[0], c[1], c[2], c[3], k)) is not None for c in cols)]
        if not keys:
            continue
        data = [[lut.get((c[0], c[1], c[2], c[3], k)) for c in cols] for k in keys]
        display = [[v if v is not None else float("nan") for v in row] for row in data]

        fig, ax = plt.subplots(figsize=(max(9, len(cols) * 0.8), max(6, len(keys) * 0.4)))
        im = ax.imshow(display, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([column_label(*c) for c in cols], rotation=60, ha="right", fontsize=8)
        for i in range(len(keys)):
            for j in range(len(cols)):
                v = data[i][j]
                if v is not None:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="Score")
        ax.set_title(f"Master headline-metric heatmap, bounded metrics only -- {version}")
        path = out_dir / f"master_heatmap_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def generate_all_charts(runs, metrics_rows, columns, out_dir: Path) -> list:
    if not MATPLOTLIB_AVAILABLE:
        print("\n[charts] matplotlib is not installed in this Python environment -- "
              "skipping chart generation. Tables and CSVs above are unaffected. "
              "Install with: pip install matplotlib  (no GPU needed), or run "
              "compare_all.py again later from an environment that has it "
              "(e.g. locally) against the same index.json.")
        return []

    lut = _lut(metrics_rows)
    written = []
    chart_fns = [
        ("repair status", lambda: chart_repair_status(runs, out_dir)),
        ("violation F1 heatmap (overview)", lambda: chart_violation_heatmap(lut, columns, out_dir)),
        ("metric scaling, bounded (tier x phase, one metric per chart)", lambda: chart_metric_scaling(lut, columns, out_dir)),
        ("metric scaling, unbounded (CIDEr-D etc.)", lambda: chart_metric_scaling_unbounded(lut, columns, out_dir)),
        ("per-rule detail (precision/recall/F1/IoU/reasoning, both directions)", lambda: chart_per_rule_metrics(lut, columns, out_dir)),
        ("grounding per-class (overview)", lambda: chart_grounding_per_class(lut, columns, out_dir)),
        ("captioning quality, bounded (overview)", lambda: chart_captioning_quality(lut, columns, out_dir)),
        ("master heatmap, bounded (overview)", lambda: chart_master_heatmap(lut, columns, out_dir)),
    ]
    print("\n[charts] generating...")
    for label, fn in chart_fns:
        try:
            paths = fn()
            written.extend(paths)
            print(f"  {label:<55s}: {len(paths)} chart(s)" if paths else f"  {label:<55s}: skipped (not enough data)")
        except Exception as e:  # noqa: BLE001 -- one bad chart must never abort the rest
            print(f"  {label:<55s}: FAILED ({e})")
    return written
