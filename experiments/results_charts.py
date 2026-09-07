"""
Chart generation for compare_all.py. Pure additive, read-only against the
results tree (see results_lib.py's module docstring for the same guarantee).

Every function here is defensive by design: comparisons in this repo are
routinely run over a PARTIAL set of tasks/tiers/phases (one tier just
finished, another is still training, a task was never run at this version),
so every chart function skips itself -- printing one line explaining why --
rather than crashing, whenever it doesn't have enough data to be meaningful
(e.g. a tier-scaling line needs >= 2 tiers; a phase-progression bar chart
needs >= 1 phase but is more useful with 2-3).

matplotlib is optional. If it isn't installed (most likely scenario: running
build_results_index.py + compare_all.py directly on the ARC login node in a
minimal env), MATPLOTLIB_AVAILABLE is False and generate_all_charts() prints
one message and returns immediately -- tables and CSVs from compare_all.py
are entirely unaffected, since they need no plotting dependency at all.
"""
from collections import defaultdict
from pathlib import Path

from experiments.results_lib import (
    FAMILY_ORDER, GROUNDING_CLASSES, HEADLINE_KEYS, PHASE_ORDER, VIOLATION_RULES,
    _tier_sort_key, column_label,
)
from core.tasks import TASK_REGISTRY, get_task_spec, task_has, CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- must work over SSH/MobaXterm with no display
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

DPI = 150
FIGSIZE_WIDE = (14, 7)
FIGSIZE_TALL = (10, 9)


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


def chart_phase_progression(lut, columns, out_dir) -> list:
    """One grouped bar chart per (task, tier, version): bars = whatever phases
    are present, groups = the task's headline metrics."""
    written = []
    groups = defaultdict(set)  # (task,tier,version) -> {phase}
    for (task, tier, phase, version) in columns:
        groups[(task, tier, version)].add(phase)

    for (task, tier, version), phases_present in groups.items():
        phases = [p for p in PHASE_ORDER if p in phases_present]
        if not phases:
            continue
        caps = get_task_spec(task).capabilities
        keys = []
        for fam in ("structural", "captioning", "grounding", "violation", "reasoning"):
            for k in HEADLINE_KEYS.get(fam, []):
                keys.append(k)
        # Only keep keys that actually have at least one value for this run set
        keys = [k for k in keys if any(lut.get((task, tier, p, version, k)) is not None for p in phases)]
        if not keys:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        n_phases = len(phases)
        width = 0.8 / max(n_phases, 1)
        x = range(len(keys))
        colors = cm.get_cmap("viridis", max(n_phases, 3))
        for i, phase in enumerate(phases):
            vals = [lut.get((task, tier, phase, version, k)) for k in keys]
            offsets = [xi + (i - (n_phases - 1) / 2) * width for xi in x]
            plot_vals = [v if v is not None else 0 for v in vals]
            bars = ax.bar(offsets, plot_vals, width=width * 0.9, label=phase.upper(), color=colors(i))
            for b, v in zip(bars, vals):
                if v is not None:
                    ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(list(x))
        ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
        ax.set_ylabel("Score")
        prefix = get_task_spec(task).prefix
        ax.set_title(f"{prefix} / {tier} / {version} -- headline metrics by phase")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / prefix / f"phase_progression_{prefix}_{tier}_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_tier_scaling(lut, columns, out_dir) -> list:
    """One line chart per (task, phase, version) with >= 2 tiers present:
    x = tier, y = value, one line per headline metric."""
    written = []
    groups = defaultdict(set)  # (task,phase,version) -> {tier}
    for (task, tier, phase, version) in columns:
        groups[(task, phase, version)].add(tier)

    for (task, phase, version), tiers_present in groups.items():
        tiers = sorted(tiers_present, key=_tier_sort_key)
        if len(tiers) < 2:
            continue
        keys = []
        for fam in ("structural", "captioning", "grounding", "violation", "reasoning"):
            keys.extend(HEADLINE_KEYS.get(fam, []))
        keys = [k for k in keys if any(lut.get((task, t, phase, version, k)) is not None for t in tiers)]
        if not keys:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = cm.get_cmap("tab10", max(len(keys), 3))
        for i, key in enumerate(keys):
            vals = [lut.get((task, t, phase, version, key)) for t in tiers]
            xs = [t for t, v in zip(tiers, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, marker="o", label=key, color=colors(i))
        prefix = get_task_spec(task).prefix
        ax.set_xlabel("Tier")
        ax.set_ylabel("Score")
        ax.set_title(f"{prefix} / {phase} / {version} -- scaling across tiers")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        path = out_dir / prefix / f"tier_scaling_{prefix}_{phase}_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_repair_status(runs, out_dir) -> list:
    """One stacked bar chart per version: x = run (task/tier/phase), stacked
    bars = valid_raw / fixed_valid / invalid_json / invalid_schema %."""
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
    run present (any task with CAP_VIOLATIONS, not just violations_only)."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        task = c[0]
        if task_has(task, CAP_VIOLATIONS):
            by_version[c[3]].add(c)

    for version, cols in by_version.items():
        from experiments.results_lib import sorted_columns
        cols = sorted_columns(cols)
        if not cols:
            continue
        data = []
        for rule in VIOLATION_RULES:
            row = []
            key = f"violation_identification_f1_{rule}"
            for c in cols:
                row.append(lut.get((c[0], c[1], c[2], c[3], key)))
            data.append(row)

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
        ax.set_title(f"Violation identification F1 by rule -- {version}")
        path = out_dir / f"violation_f1_heatmap_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_grounding_per_class(lut, columns, out_dir) -> list:
    """Per-class grounding IoU bar chart, grouped by run, for any object-
    capable task (unified AND object_only together -- a comparison no
    existing tool made, since they were always plotted separately)."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], CAP_OBJECTS):
            by_version[c[3]].add(c)

    for version, cols in by_version.items():
        from experiments.results_lib import sorted_columns
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
        ax.set_ylabel("Mask IoU (macro, tn0)")
        ax.set_title(f"Per-class grounding IoU -- {version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / f"grounding_per_class_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_captioning_quality(lut, columns, out_dir) -> list:
    """Grouped bar chart of caption-quality metrics, for any caption-capable
    task, one figure per version."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        if task_has(c[0], CAP_CAPTION):
            by_version[c[3]].add(c)

    metrics_ = HEADLINE_KEYS["captioning"]
    for version, cols in by_version.items():
        from experiments.results_lib import sorted_columns
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
        ax.set_ylabel("Score")
        ax.set_title(f"Captioning quality -- {version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = out_dir / f"captioning_quality_{version}.png"
        _save(fig, path)
        written.append(path)
    return written


def chart_master_heatmap(lut, columns, out_dir) -> list:
    """One big at-a-glance heatmap per version: rows = every headline metric
    across every family, columns = every run present (any task, tier, phase)
    -- the single-figure summary view."""
    written = []
    by_version = defaultdict(set)
    for c in columns:
        by_version[c[3]].add(c)

    all_headline_keys = []
    for fam in FAMILY_ORDER:
        all_headline_keys.extend(HEADLINE_KEYS.get(fam, []))

    for version, cols in by_version.items():
        from experiments.results_lib import sorted_columns
        cols = sorted_columns(cols)
        if not cols:
            continue
        keys = [k for k in all_headline_keys if any(lut.get((c[0], c[1], c[2], c[3], k)) is not None for c in cols)]
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
        ax.set_title(f"Master headline-metric heatmap -- {version}")
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
        ("phase progression", lambda: chart_phase_progression(lut, columns, out_dir)),
        ("tier scaling", lambda: chart_tier_scaling(lut, columns, out_dir)),
        ("repair status", lambda: chart_repair_status(runs, out_dir)),
        ("violation F1 heatmap", lambda: chart_violation_heatmap(lut, columns, out_dir)),
        ("grounding per-class", lambda: chart_grounding_per_class(lut, columns, out_dir)),
        ("captioning quality", lambda: chart_captioning_quality(lut, columns, out_dir)),
        ("master heatmap", lambda: chart_master_heatmap(lut, columns, out_dir)),
    ]
    print("\n[charts] generating...")
    for label, fn in chart_fns:
        try:
            paths = fn()
            written.extend(paths)
            print(f"  {label:<22s}: {len(paths)} chart(s)" if paths else f"  {label:<22s}: skipped (not enough data)")
        except Exception as e:  # noqa: BLE001 -- one bad chart must never abort the rest
            print(f"  {label:<22s}: FAILED ({e})")
    return written
