#!/usr/bin/env python3
"""
Reads a results_index/index.json (built by build_results_index.py) and
produces terminal-printed comparison tables, CSVs, and PNG charts -- for
however many tasks/tiers/phases/versions actually have data. Works fine with
one task and one phase, or all four tasks across all three tiers and all
three phases; nothing here assumes a complete 4x3x3 grid.

This is the entire replacement for the old compare_results.py / plot_metrics.py /
plot_metrics_vo.py / generate_comparison_csv.py -- one command instead of
picking the right one of five scripts per situation.

Usage
-----
    # Everything in the index
    python -m experiments.compare_all --index results_index/index.json --out results_index/

    # Narrow to a subset -- any combination, any number of values
    python -m experiments.compare_all --index results_index/index.json \\
        --tasks violations_only unified --tiers 2b 4b --phases baseline grpo

    # Tables + CSVs only, skip chart generation (e.g. no matplotlib available)
    python -m experiments.compare_all --index results_index/index.json --no-charts

Output (under --out, default results_index/):
    master_wide.csv       -- every metric, every selected run, one file
    repair_stats.csv      -- structural repair status/change/warning counts, per run
    delta_<prefix>_<version>.csv  -- one per (task, version): baseline/sft/grpo
                                      deltas, %-change, best-phase, per tier
    charts/                -- PNG charts (see results_charts.py), unless --no-charts
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import VALID_TASKS
from core.tasks import get_task_spec, TASK_REGISTRY
from experiments.results_lib import (
    BOOTSTRAP_METRICS, FAMILY_ORDER, HEADLINE_KEYS, MACRO_N_RULES_KEY, PHASES,
    PHASE_ORDER, SUPPORT_KEYS, best_phase, bootstrap_indices, bootstrap_series,
    ci_from_series, column_label, decode_outcomes, fmt_delta, fmt_pct, fmt_val,
    is_non_comparable_key, load_index, paired_delta, print_table,
    significance_marker, sorted_columns, _tier_sort_key,
)


def filter_runs(runs, tasks, tiers, phases, versions):
    out = []
    for r in runs:
        if tasks and r["task"] not in tasks:
            continue
        if tiers and r["tier"] not in tiers:
            continue
        if phases and r["phase"] not in phases:
            continue
        if versions and r["version"] not in versions:
            continue
        out.append(r)
    return out


def col_of(row):
    return (row["task"], row["tier"], row["phase"], row["version"])


def build_lut(metrics_rows, columns):
    lut, fam_of = {}, {}
    for r in metrics_rows:
        c = col_of(r)
        if c not in columns:
            continue
        lut[(c[0], c[1], c[2], c[3], r["metric_key"])] = r["value"]
        fam_of[r["metric_key"]] = r["metric_family"]
    return lut, fam_of


# ---------------------------------------------------------------------------
# Terminal headline view
# ---------------------------------------------------------------------------

def print_headline_tables(lut, columns):
    by_task = defaultdict(set)
    for c in columns:
        by_task[c[0]].add(c)

    for task in TASK_REGISTRY:
        if task not in by_task:
            continue
        cols = sorted_columns(by_task[task])
        caps = get_task_spec(task).capabilities
        rows = []
        headers = ["metric"] + [column_label(*c) for c in cols]
        for fam in ("structural", "captioning", "grounding", "violation", "reasoning"):
            for key in HEADLINE_KEYS.get(fam, []):
                vals = [lut.get((c[0], c[1], c[2], c[3], key)) for c in cols]
                if all(v is None for v in vals):
                    continue
                # Macro metrics carry their divisor inline: "0.7817 (n=2)".
                # Reasoning and violation-grounding macros average only over the
                # rules that had a true positive to be measured on, so a run that
                # detected nothing for two rules produces a macro over 2, not 4.
                # Rendering the value without n invites comparing those directly.
                nkey = MACRO_N_RULES_KEY.get(key)
                if nkey:
                    ns = [lut.get((c[0], c[1], c[2], c[3], nkey)) for c in cols]
                    if any(n is not None for n in ns):
                        vals = [
                            (f"{v:.4f}" + (f" (n={int(n)})" if n is not None else ""))
                            if v is not None else None
                            for v, n in zip(vals, ns)
                        ]
                rows.append([key] + vals)
        if rows:
            print_table(headers, rows, title=f"{get_task_spec(task).prefix} ({task}) -- headline metrics")


def print_support_tables(lut, columns):
    """Sample sizes behind the scores, printed as their own table.

    Every reasoning metric is conditioned on the model's OWN true positives, so
    two runs are scored on different image populations. On the real vo runs
    4b-grpo's rule_3 reasoning averages over 22 images and 8b-grpo's over 8 --
    8b scores higher partly because it only had to explain its 8 easiest cases.
    The same conditioning applies to violation grounding IoU.

    These numbers are shown, never used to suppress a comparison: the deltas are
    always printed in full and judging them is the reader's call.
    """
    by_task = defaultdict(set)
    for c in columns:
        by_task[c[0]].add(c)
    for task in TASK_REGISTRY:
        if task not in by_task:
            continue
        cols = sorted_columns(by_task[task])
        for fam in ("violation", "reasoning"):
            rows = []
            for key in SUPPORT_KEYS.get(fam, []):
                vals = [lut.get((c[0], c[1], c[2], c[3], key)) for c in cols]
                if all(v is None for v in vals):
                    continue
                rows.append([key] + [None if v is None else int(v) for v in vals])
            if rows:
                print_table(
                    ["support / sample size"] + [column_label(*c) for c in cols],
                    rows,
                    title=f"{get_task_spec(task).prefix} ({task}) -- {fam} support counts "
                          f"(the denominators behind the scores above)",
                )


def print_significance(runs, columns, n_boot, seed, out_dir):
    """Bootstrap 95% CIs per run, and a paired test for every adjacent comparison.

    Reads the packed per-image outcome vector carried on each run row, so this
    needs nothing beyond index.json -- no predictions_repaired.jsonl, no second
    download. Runs evaluated before that key existed are reported as
    unavailable rather than silently skipped.

    WHY IT MATTERS. rule_2/rule_3/rule_4 have 25/63/24 positives in the whole
    3004-image test split, so their per-rule F1 carries a 95% interval up to
    0.33 wide and f1_macro averages four such numbers. Point estimates alone
    cannot separate "GRPO helped" from "GRPO did nothing", which is exactly why
    phase and tier rankings appear to flip between runs.
    """
    outcomes, missing = {}, []
    for r in runs:
        c = col_of(r)
        if c not in columns:
            continue
        dec = decode_outcomes(r.get("violation_per_image_outcomes_b64"))
        if dec is None:
            missing.append(column_label(*c))
        else:
            outcomes[c] = dec

    if not outcomes:
        print()
        print("=== Significance ===")
        print("  No run in this index carries violation_per_image_outcomes_b64, so no")
        print("  bootstrap can run. That key is written by evaluation/metrics_violations.py;")
        print("  re-run run_evaluation.py on these predictions to enable confidence intervals.")
        return
    if missing:
        print()
        print(f"[significance] no per-image outcomes for: {', '.join(missing)} "
              f"(evaluated before that key existed) -- excluded from the tests below.")

    lengths = {len(p) for p, _ in outcomes.values()}
    if len(lengths) > 1:
        print()
        print(f"[significance] runs disagree on image count {sorted(lengths)}; "
              "a paired test needs one common split -- skipping.")
        return
    n = lengths.pop()
    boot_idx = bootstrap_indices(n, n_boot, seed)

    # One pass per run. Every comparison below reuses these series, so a paired
    # delta is an elementwise subtraction rather than a second bootstrap.
    series = {c: bootstrap_series(pred, gt, boot_idx) for c, (pred, gt) in outcomes.items()}

    cols = sorted_columns(outcomes)
    rows = []
    for c in cols:
        row = [column_label(*c)]
        for metric in BOOTSTRAP_METRICS:
            point, ser = series[c][metric]
            lo, hi = ci_from_series(ser)
            row += [f"{point:.4f}",
                    "" if lo is None else f"[{lo:.4f}, {hi:.4f}]"]
        rows.append(row)
    print_table(
        ["run", "F1_micro", "95% CI", "F1_macro", "95% CI"], rows,
        title=f"Violation identification with 95% bootstrap CI over {n} images "
              f"(B={n_boot}, seed={seed})",
    )

    # Adjacent comparisons: phase progression within a tier, then tier scaling
    # within a phase. Those are the two axes this pipeline exists to compare.
    pairs = []
    by_tier = defaultdict(dict)
    for c in cols:
        by_tier[(c[0], c[1], c[3])][c[2]] = c
    for _, phases in sorted(by_tier.items()):
        for a, b in (("baseline", "sft"), ("sft", "grpo"), ("baseline", "grpo")):
            if a in phases and b in phases:
                pairs.append((phases[a], phases[b]))
    by_phase = defaultdict(dict)
    for c in cols:
        by_phase[(c[0], c[2], c[3])][c[1]] = c
    for _, tiers in sorted(by_phase.items()):
        ordered = sorted(tiers, key=_tier_sort_key)
        for a, b in zip(ordered, ordered[1:]):
            pairs.append((tiers[a], tiers[b]))

    if not pairs:
        return

    def delta_row(ca, cb):
        cells = []
        for metric in BOOTSTRAP_METRICS:
            pa, sa = series[ca][metric]
            pb, sb = series[cb][metric]
            lo, hi, p = paired_delta(sa, sb)
            cells += [f"{pb - pa:+.4f}",
                      "" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]",
                      "" if p is None else f"{p:.3f} {significance_marker(p)}"]
        return cells

    print_table(
        ["comparison", "d F1_micro", "95% CI", "p", "d F1_macro", "95% CI", "p"],
        [[f"{column_label(*ca)}  ->  {column_label(*cb)}"] + delta_row(ca, cb) for ca, cb in pairs],
        title="Paired bootstrap -- does the right-hand run beat the left-hand one?",
    )
    print("  p is one-sided: the fraction of resamples where the improvement did NOT hold.")
    print("  *** p<0.001   ** p<0.01   * p<0.05   ns = not significant (the CI contains 0).")

    path = out_dir / "significance.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "value_a", "value_b", "delta", "ci_lo", "ci_hi",
                    "p_one_sided", "marker"])
        for ca, cb in pairs:
            for metric in BOOTSTRAP_METRICS:
                pa, sa = series[ca][metric]
                pb, sb = series[cb][metric]
                lo, hi, p = paired_delta(sa, sb)
                w.writerow([f"{column_label(*ca)} -> {column_label(*cb)}", metric,
                            f"{pa:.6f}", f"{pb:.6f}", f"{pb - pa:.6f}",
                            "" if lo is None else f"{lo:.6f}",
                            "" if hi is None else f"{hi:.6f}",
                            "" if p is None else f"{p:.4f}", significance_marker(p)])
    print()
    print(f"Wrote {path}")


def print_repair_summary(runs, columns):
    rows_by_col = {col_of(r): r for r in runs}
    present = [rows_by_col[c] for c in sorted_columns(columns) if c in rows_by_col]
    present = [r for r in present if r.get("repair_valid_raw_pct") is not None or r.get("repair_fixed_pct") is not None]
    if not present:
        return
    headers = ["run", "valid_raw_%", "fixed_%", "invalid_json_%", "invalid_schema_%"]
    rows = [
        [column_label(r["task"], r["tier"], r["phase"], r["version"]),
         r.get("repair_valid_raw_pct"), r.get("repair_fixed_pct"),
         r.get("repair_invalid_json_pct"), r.get("repair_invalid_schema_pct")]
        for r in present
    ]
    print_table(headers, rows, title="Structural repair summary")


# ---------------------------------------------------------------------------
# CSVs
# ---------------------------------------------------------------------------

def write_master_wide_csv(path, lut, fam_of, columns):
    cols = sorted_columns(columns)
    headers = ["metric_family", "metric_key"] + [column_label(*c) for c in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        by_fam = defaultdict(set)
        for key, fam in fam_of.items():
            by_fam[fam].add(key)
        for fam in FAMILY_ORDER:
            for key in sorted(by_fam.get(fam, [])):
                row = [fam, key]
                for c in cols:
                    v = lut.get((c[0], c[1], c[2], c[3], key))
                    row.append("" if v is None else v)
                w.writerow(row)


def write_repair_csv(path, repair_rows, columns):
    cols = sorted_columns(columns)
    data = defaultdict(dict)
    row_order = []
    for r in repair_rows:
        c = col_of(r)
        if c not in columns:
            continue
        if r["kind"] == "status":
            for suffix, val in (("count", r.get("count")), ("pct", r.get("pct_of_total"))):
                rk = f"status:{r['name']}:{suffix}"
                if rk not in data:
                    row_order.append(rk)
                data[rk][c] = val
        else:
            for suffix, val in (("occurrences", r.get("total_occurrences")), ("records_affected", r.get("records_affected"))):
                rk = f"{r['kind']}:{r['name']}:{suffix}"
                if rk not in data:
                    row_order.append(rk)
                data[rk][c] = val
    if not row_order:
        return False
    headers = ["repair_row"] + [column_label(*c) for c in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for rk in sorted(row_order):
            row = [rk] + ["" if data[rk].get(c) is None else data[rk][c] for c in cols]
            w.writerow(row)
    return True


def write_delta_csvs(lut, fam_of, columns, out_dir):
    written = []
    task_versions = sorted({(c[0], c[3]) for c in columns})
    for task, version in task_versions:
        tiers_present = sorted({c[1] for c in columns if c[0] == task and c[3] == version}, key=_tier_sort_key)
        keys_present = sorted(
            k for k in fam_of
            if any((task, t, p, version, k) in lut for t in tiers_present for p in PHASES)
        )
        if not keys_present:
            continue
        prefix = get_task_spec(task).prefix
        path = out_dir / f"delta_{prefix}_{version}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for tier in tiers_present:
                w.writerow([f"=== {prefix} / {tier} / {version} ==="])
                w.writerow(["metric_family", "metric", "BASELINE", "SFT", "GRPO",
                             "d_SFT-Base", "pct_SFT_vs_Base", "d_GRPO-Base", "pct_GRPO_vs_Base",
                             "d_GRPO-SFT", "pct_GRPO_vs_SFT", "best_phase"])
                wins, total = defaultdict(int), 0
                for fam in FAMILY_ORDER:
                    fam_keys = sorted(k for k in keys_present if fam_of.get(k) == fam)
                    for key in fam_keys:
                        b = lut.get((task, tier, "baseline", version, key))
                        s = lut.get((task, tier, "sft", version, key))
                        g = lut.get((task, tier, "grpo", version, key))
                        if b is None and s is None and g is None:
                            continue
                        d_sb = (s - b) if (s is not None and b is not None) else None
                        d_gb = (g - b) if (g is not None and b is not None) else None
                        d_gs = (g - s) if (g is not None and s is not None) else None
                        winner = best_phase(b, s, g)
                        if b is not None and s is not None and g is not None and not is_non_comparable_key(key):
                            total += 1
                            wins[winner] += 1
                        w.writerow([fam, key, fmt_val(b), fmt_val(s), fmt_val(g),
                                    fmt_delta(d_sb), fmt_pct(b, s), fmt_delta(d_gb), fmt_pct(b, g),
                                    fmt_delta(d_gs), fmt_pct(s, g), winner])
                w.writerow([])
                w.writerow([f"SUMMARY {tier}", "total_fully_scored_metrics", total])
                for ph in ("BASELINE", "SFT", "GRPO"):
                    n = wins.get(ph, 0)
                    pct = f"{n / total * 100:.1f}%" if total else ""
                    w.writerow([f"SUMMARY {tier}", f"{ph}_wins", n, pct])
                w.writerow([])
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="Path to index.json from build_results_index.py")
    ap.add_argument("--out", default="results_index", help="Output directory (default: results_index/)")
    ap.add_argument("--tasks", nargs="+", default=None, choices=VALID_TASKS)
    ap.add_argument("--tiers", nargs="+", default=None)
    ap.add_argument("--phases", nargs="+", default=None, choices=list(PHASES))
    ap.add_argument("--versions", nargs="+", default=None)
    ap.add_argument("--no-charts", action="store_true", help="Skip PNG chart generation (tables/CSVs still run)")
    ap.add_argument("--bootstrap", type=int, default=2000, metavar="B",
                    help="Bootstrap resamples for the confidence intervals and paired "
                         "significance tests (default: 2000; 0 disables).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Bootstrap RNG seed (default: 0). Same index + same seed "
                         "reproduces identical intervals.")
    args = ap.parse_args()

    index = load_index(Path(args.index))
    runs = filter_runs(index["runs"], args.tasks, args.tiers, args.phases, args.versions)
    if not runs:
        raise SystemExit(
            "No runs matched your filters (or the index is empty). "
            "Check --tasks/--tiers/--phases/--versions, or re-run build_results_index.py."
        )
    columns = {col_of(r) for r in runs}
    metrics_rows = [r for r in index["metrics"] if col_of(r) in columns]
    repair_rows = [r for r in index["repair"] if col_of(r) in columns]
    lut, fam_of = build_lut(metrics_rows, columns)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(runs)} run(s) across "
          f"{len({c[0] for c in columns})} task(s), {len({c[1] for c in columns})} tier(s), "
          f"{len({c[2] for c in columns})} phase(s), {len({c[3] for c in columns})} version(s).\n")

    print_headline_tables(lut, columns)
    print_support_tables(lut, columns)
    print_repair_summary(runs, columns)
    if args.bootstrap > 0:
        print_significance(runs, columns, args.bootstrap, args.seed, out_dir)
    else:
        print()
        print("[significance] --bootstrap 0 passed, skipping confidence intervals.")

    wide_path = out_dir / "master_wide.csv"
    write_master_wide_csv(wide_path, lut, fam_of, columns)
    print(f"\nWrote {wide_path}  ({len(fam_of)} metric keys x {len(columns)} runs)")

    repair_path = out_dir / "repair_stats.csv"
    if write_repair_csv(repair_path, repair_rows, columns):
        print(f"Wrote {repair_path}")
    else:
        print("No repair_report.json data found for the selected runs -- repair_stats.csv skipped.")

    delta_paths = write_delta_csvs(lut, fam_of, columns, out_dir)
    for p in delta_paths:
        print(f"Wrote {p}")
    if not delta_paths:
        print("No task had enough phase data for a delta CSV (need at least one phase per tier).")

    if args.no_charts:
        print("\n[charts] --no-charts passed, skipping chart generation.")
    else:
        from experiments.results_charts import generate_all_charts
        generate_all_charts(runs, metrics_rows, columns, out_dir / "charts")

    print(f"\nAll output under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
