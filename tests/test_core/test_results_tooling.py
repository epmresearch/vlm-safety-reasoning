"""
Regression tests for the results-comparison toolset (experiments/results_lib.py,
build_results_index.py, compare_all.py) that replaced compare_results.py /
plot_metrics.py / plot_metrics_vo.py / generate_comparison_csv.py on 2026-09-07.

These tests are entirely offline (no GPU, no ARC, no real results tree needed)
and never touch anything the actual training/inference/eval pipeline reads or
writes -- they only exercise the new, purely-additive read-only tooling.
"""
import json

import pytest

from core.constants import VALID_TASKS
from core.naming import results_dir_names
from experiments.results_lib import (
    RunKey, discover_and_index, index_one_run, metric_family, parse_run_name,
)


# ---------------------------------------------------------------------------
# parse_run_name must be the exact reverse of results_dir_names -- if these
# two ever drift apart, the index builder silently stops discovering runs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", VALID_TASKS)
@pytest.mark.parametrize("tier", ["2b", "4b", "8b"])
def test_parse_run_name_round_trips_every_real_generated_name(task, tier):
    version = "v1"
    names = results_dir_names(task, tier, version)
    for phase, name in names.items():
        key = parse_run_name(name)
        assert key is not None, f"failed to parse a real generated name: {name!r}"
        assert key == RunKey(task=task, phase=phase, tier=tier, version=version)


@pytest.mark.parametrize("bad_name", [
    "baseline",              # pre-normalization legacy unified baseline
    "baseline_8b",           # legacy, no prefix/version
    "vo_baseline_2b",        # underscore convention, no version
    "vo-sft-2b-v4",          # missing the required suffix
    "vo-grpo-2b-v4",         # missing the required _final suffix
    "vo-grpo-2b-v4_best",    # grpo has no eval set and therefore never a best/
    "vo-baseline-2b-v1_best",   # baseline must NOT carry a suffix
    "vo-baseline-2b-v1_final",  # ...of either kind
    "notatask-baseline-2b-v1",  # unregistered prefix
    "vo-training-2b-v1",     # unregistered phase
    "plots_vo_v4",           # not a run directory at all
    "logs",
])
def test_parse_run_name_rejects_legacy_and_malformed_names(bad_name):
    assert parse_run_name(bad_name) is None


def test_sft_run_names_accept_both_final_and_the_legacy_best_suffix():
    """SFT results are named `_final` as of 2026-09-10 (the merge handoff moved
    from best/ to final/). Every SFT run produced BEFORE that is on disk as
    `<variant>_best`, and refusing those would silently drop a whole phase out of
    an index built over existing results -- reported as "skipped as unparseable",
    which reads like a naming bug rather than a convention change.

    The suffix is not load-bearing downstream: the phase is already pinned by the
    middle segment, so accepting both costs nothing.
    """
    for suffix in ("final", "best"):
        key = parse_run_name(f"vo-sft-2b-v1_{suffix}")
        assert key == RunKey(task="violations_only", phase="sft", tier="2b", version="v1"), suffix

    # ...but the leniency is scoped to sft alone.
    assert parse_run_name("vo-grpo-2b-v1_best") is None


def test_results_dir_names_emit_final_for_both_trained_phases():
    """core/naming.py is the single source of truth the shell scripts follow:
    hpc_sft.sh runs inference with --checkpoint final and writes
    ${VARIANT}_final, and hpc_merge_sft.sh merges from final/. If this drifts,
    the index looks for directories the pipeline never created."""
    names = results_dir_names("violations_only", "8b", "v1")
    assert names["sft"] == "vo-sft-8b-v1_final"
    assert names["grpo"] == "vo-grpo-8b-v1_final"
    assert names["baseline"] == "vo-baseline-8b-v1"


def test_parse_run_name_prefixes_are_unambiguous():
    """No registered task prefix may be a leading substring of another --
    would make the regex alternation order-sensitive and fragile."""
    from core.tasks import TASK_REGISTRY
    prefixes = [spec.prefix for spec in TASK_REGISTRY.values()]
    for a in prefixes:
        for b in prefixes:
            if a != b:
                assert not b.startswith(a), f"{a!r} is a prefix of {b!r}"


# ---------------------------------------------------------------------------
# metric_family -- derived purely from key prefix, must classify every real
# family correctly and never crash on something unexpected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("structural_json_validity_rate", "structural"),
    ("captioning_bertscore_f1", "captioning"),
    ("grounding_mask_iou_all_macro_excavator_tn0", "grounding"),
    ("violation_identification_f1_rule_3", "violation"),
    ("violation_grounding_mask_iou_rule_1_tn0", "violation"),
    ("violation_prediction_failure_rate", "violation"),
    ("reasoning_text_similarity_bertscore_f1_macro", "reasoning"),
    ("some_totally_unknown_future_key", "other"),
])
def test_metric_family_derivation(key, expected):
    assert metric_family(key) == expected


# ---------------------------------------------------------------------------
# index_one_run against a synthetic on-disk run, both layouts.
# ---------------------------------------------------------------------------

def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


SAMPLE_METRICS = {
    "structural_json_validity_rate": 0.99,
    "structural_schema_adherence_rate": 0.98,
    "structural_total_samples_count": 3004,
    "violation_identification_f1_micro": 0.24,
    "violation_identification_f1_rule_1": 0.15,
}

SAMPLE_REPAIR = {
    "total_records": 3004,
    "status_counts": {"valid_raw": 460, "fixed_valid": 2525, "invalid_json": 15, "invalid_schema": 4},
    "status_percentages": {"valid_raw": 15.31, "fixed_valid": 84.05, "invalid_json": 0.5, "invalid_schema": 0.13},
    "change_type_summary": {"box_flat_list_reshaped": {"total_occurrences": 312, "records_affected": 198}},
    "warning_type_summary": {"likely_truncated_output": {"total_occurrences": 7, "records_affected": 7}},
}


def test_index_one_run_arc_layout(tmp_path):
    run_dir = tmp_path / "vo-baseline-2b-v1"
    _write_json(run_dir / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    _write_json(run_dir / "repair_applied" / "repair_report.json", SAMPLE_REPAIR)

    key = parse_run_name("vo-baseline-2b-v1")
    result = index_one_run(run_dir, key, "arc")

    assert result["run"]["has_metrics"] is True
    assert result["run"]["has_repair_report"] is True
    assert result["run"]["total_samples"] == 3004
    assert result["run"]["repair_fixed_pct"] == 84.05

    metric_keys = {m["metric_key"] for m in result["metrics"]}
    assert metric_keys == set(SAMPLE_METRICS)
    assert all(m["metric_family"] in ("structural", "violation") for m in result["metrics"])

    repair_kinds = {(r["kind"], r["name"]) for r in result["repair"]}
    assert ("status", "valid_raw") in repair_kinds
    assert ("change_type", "box_flat_list_reshaped") in repair_kinds
    assert ("warning_type", "likely_truncated_output") in repair_kinds


def test_index_one_run_flat_layout(tmp_path):
    run_dir = tmp_path / "oo-grpo-4b-v1_final"
    _write_json(run_dir / "metrics.json", SAMPLE_METRICS)

    key = parse_run_name("oo-grpo-4b-v1_final")
    assert key == RunKey(task="object_only", phase="grpo", tier="4b", version="v1")

    result = index_one_run(run_dir, key, "flat")
    assert result["run"]["has_metrics"] is True
    assert result["run"]["has_repair_report"] is False  # never written for this run
    assert len(result["metrics"]) == len(SAMPLE_METRICS)


def test_index_one_run_missing_files_does_not_crash(tmp_path):
    run_dir = tmp_path / "co-sft-2b-v1_best"
    run_dir.mkdir()  # empty dir -- job submitted but nothing written yet

    key = parse_run_name("co-sft-2b-v1_best")
    result = index_one_run(run_dir, key, "auto")

    assert result["run"]["has_metrics"] is False
    assert result["run"]["has_repair_report"] is False
    assert result["metrics"] == []
    assert result["repair"] == []


def test_index_one_run_corrupted_json_treated_as_missing_not_fatal(tmp_path):
    run_dir = tmp_path / "unified-baseline-8b-v1"
    p = run_dir / "evaluation_results" / "metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")  # e.g. job killed mid-write

    key = parse_run_name("unified-baseline-8b-v1")
    result = index_one_run(run_dir, key, "arc")  # must not raise

    assert result["run"]["has_metrics"] is False
    assert result["metrics"] == []


# ---------------------------------------------------------------------------
# discover_and_index end-to-end over a small synthetic tree: mixed valid/
# legacy/partial runs, filters, and a full write_index/load_index round trip.
# ---------------------------------------------------------------------------

def test_discover_and_index_mixed_tree(tmp_path):
    # A real, complete run.
    _write_json(tmp_path / "vo-baseline-2b-v1" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    # A run that's been submitted but not evaluated yet.
    (tmp_path / "vo-sft-2b-v1_best").mkdir()
    # A legacy, unparseable directory that must be skipped, not crash anything.
    (tmp_path / "baseline_2b").mkdir()
    # An unrelated directory (e.g. a stray plots/ folder) -- also skipped.
    (tmp_path / "plots_vo_v4").mkdir()

    runs, metrics, repair, skipped, counted = discover_and_index(tmp_path, "arc")

    run_ids = {r["run_id"] for r in runs}
    assert "violations_only:2b:baseline:v1" in run_ids
    assert "violations_only:2b:sft:v1" in run_ids
    assert len(runs) == 2
    assert len(metrics) == len(SAMPLE_METRICS)  # only the baseline run had data
    assert set(skipped) == {"baseline_2b", "plots_vo_v4"}


def test_discover_and_index_filters_by_task_tier_version(tmp_path):
    _write_json(tmp_path / "vo-baseline-2b-v1" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    _write_json(tmp_path / "unified-baseline-2b-v1" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    _write_json(tmp_path / "vo-baseline-4b-v1" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    _write_json(tmp_path / "vo-baseline-2b-v2" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)

    runs, *_ = discover_and_index(tmp_path, "arc", tasks=["violations_only"], tiers=["2b"], versions=["v1"])
    assert len(runs) == 1
    assert runs[0]["run_id"] == "violations_only:2b:baseline:v1"


def test_write_and_load_index_round_trip(tmp_path):
    from experiments.results_lib import load_index, write_index

    _write_json(tmp_path / "src" / "vo-baseline-2b-v1" / "evaluation_results" / "metrics.json", SAMPLE_METRICS)
    runs, metrics, repair, skipped, _ = discover_and_index(tmp_path / "src", "arc")

    out_path = tmp_path / "out" / "index.json"
    write_index(out_path, runs, metrics, repair, skipped, "arc", str(tmp_path / "src"))

    loaded = load_index(out_path)
    assert loaded["schema_version"] == 1
    assert len(loaded["runs"]) == 1
    assert len(loaded["metrics"]) == len(SAMPLE_METRICS)
    assert loaded["runs"][0]["run_id"] == "violations_only:2b:baseline:v1"


# ===========================================================================
# Bootstrap / significance helpers
#
# These exist because three of the four safety rules have 25, 63 and 24
# positives in the whole test split, so a per-rule F1 carries a 95% interval up
# to 0.33 wide and f1_macro averages four such numbers. Point estimates alone
# cannot separate "GRPO helped" from "GRPO did nothing", which is why phase and
# tier rankings appear to flip between statistically indistinguishable runs.
# ===========================================================================

import base64

from experiments.results_lib import (
    BOOTSTRAP_METRICS, CHARTABLE_BOUNDED_KEYS, DEMOTED_HEADLINE_KEYS,
    MACRO_N_RULES_KEY, SUPPORT_KEYS, bootstrap_indices, bootstrap_series,
    ci_from_series, contribution_matrices, decode_outcomes, paired_delta,
    significance_marker,
)


def _pack(pairs):
    """pairs: list of (pred_mask, gt_mask) -> the base64 vector metrics_violations emits."""
    return base64.b64encode(bytes((p << 4) | g for p, g in pairs)).decode("ascii")


def test_decode_outcomes_round_trips():
    pairs = [(0b0001, 0b0001), (0b0000, 0b0100), (0b1010, 0b0010), (0b0000, 0b0000)]
    pred, gt = decode_outcomes(_pack(pairs))
    assert pred == [p for p, _ in pairs]
    assert gt == [g for _, g in pairs]


def test_decode_outcomes_is_tolerant_of_absence_and_garbage():
    # A run evaluated before the key existed must disable the bootstrap, never crash.
    assert decode_outcomes(None) is None
    assert decode_outcomes("") is None
    assert decode_outcomes("!!!not base64!!!") is None


def test_contribution_matrices_match_a_hand_confusion():
    #            pred      gt
    pred, gt = decode_outcomes(_pack([
        (0b0001, 0b0001),   # rule_1 TP
        (0b0010, 0b0000),   # rule_2 FP
        (0b0000, 0b0100),   # rule_3 FN
    ]))
    tp, fp, fn = contribution_matrices(pred, gt)
    assert tp[0] == [1, 0, 0, 0]
    assert fp[1] == [0, 1, 0, 0]
    assert fn[2] == [0, 0, 1, 0]


def test_bootstrap_point_estimate_equals_the_direct_computation():
    pairs = [(0b0001, 0b0001)] * 30 + [(0b0010, 0b0000)] * 10 + [(0b0000, 0b0100)] * 10
    pred, gt = decode_outcomes(_pack(pairs))
    series = bootstrap_series(pred, gt, bootstrap_indices(len(pred), 50, seed=1))
    # 30 TP, 10 FP, 10 FN pooled -> P = R = 0.75, F1 = 0.75
    assert abs(series["f1_micro"][0] - 0.75) < 1e-12
    for m in BOOTSTRAP_METRICS:
        point, vals = series[m]
        assert len(vals) == 50
        lo, hi = ci_from_series(vals)
        assert lo <= point <= hi


def test_bootstrap_is_deterministic_for_a_given_seed():
    pairs = [(0b0001, 0b0001)] * 20 + [(0b0010, 0b0000)] * 20
    pred, gt = decode_outcomes(_pack(pairs))
    a = bootstrap_series(pred, gt, bootstrap_indices(len(pred), 40, seed=7))
    b = bootstrap_series(pred, gt, bootstrap_indices(len(pred), 40, seed=7))
    c = bootstrap_series(pred, gt, bootstrap_indices(len(pred), 40, seed=8))
    assert a["f1_micro"][1] == b["f1_micro"][1]
    assert a["f1_micro"][1] != c["f1_micro"][1]


def test_paired_delta_detects_a_real_difference_and_a_null_one():
    n = 200
    gt_pairs = [(0b0001, 0b0001)] * 100 + [(0b0000, 0b0001)] * 100
    weak, gt = decode_outcomes(_pack([(0b0000, g) for _, g in gt_pairs]))   # detects nothing
    strong, _ = decode_outcomes(_pack([(0b0001, g) for _, g in gt_pairs]))  # detects everything
    boot = bootstrap_indices(n, 400, seed=0)

    sw = bootstrap_series(weak, gt, boot)
    ss = bootstrap_series(strong, gt, boot)

    lo, hi, p = paired_delta(sw["f1_micro"][1], ss["f1_micro"][1])
    assert p < 0.001 and lo > 0, "a large real improvement must be significant"

    # Identical runs: delta is exactly 0 everywhere, so p == 1.0 (never > 0).
    lo, hi, p = paired_delta(ss["f1_micro"][1], ss["f1_micro"][1])
    assert lo == hi == 0.0 and p == 1.0


def test_paired_delta_handles_an_empty_series():
    assert paired_delta([], [], ) == (None, None, None)


def test_significance_marker_thresholds():
    assert significance_marker(0.0005) == "***"
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.02) == "*"
    assert significance_marker(0.5) == "ns"
    assert significance_marker(None) == ""


def test_index_carries_the_per_image_outcome_vector(tmp_path):
    """The vector is a STRING, so the numeric filter must keep it out of the
    long-format metrics table while the run summary still carries it -- that is
    what lets compare_all.py bootstrap from index.json alone."""
    from experiments.results_lib import RunKey, index_one_run

    vec = _pack([(0b0001, 0b0001), (0b0000, 0b0000)])
    run_dir = tmp_path / "vo-sft-2b-v1_best" / "evaluation_results"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({
        "violation_identification_f1_micro": 0.5,
        "violation_per_image_outcomes_b64": vec,
    }), encoding="utf-8")

    key = RunKey(task="violations_only", phase="sft", tier="2b", version="v1")
    result = index_one_run(tmp_path / "vo-sft-2b-v1_best", key, "auto")

    assert result["run"]["violation_per_image_outcomes_b64"] == vec
    keys = {m["metric_key"] for m in result["metrics"]}
    assert "violation_identification_f1_micro" in keys
    assert "violation_per_image_outcomes_b64" not in keys, "a string must not become a metric row"


def test_demoted_keys_stay_out_of_the_headline_but_keep_their_charts():
    """Demotion governs what competes for attention in a scannable table, never
    what is kept: master_wide.csv dumps every key, and charts read the chartable
    set. Reasoning CLIPScore spans 0.627-0.666 across the nine real vo runs
    while f1_micro spans 0.129-0.487 -- no discriminative power, but still data."""
    from experiments.results_lib import BOUNDED_HEADLINE_KEYS

    demoted = DEMOTED_HEADLINE_KEYS["reasoning"]
    assert "reasoning_text_similarity_clipscore_macro" in demoted
    for k in demoted:
        assert k not in BOUNDED_HEADLINE_KEYS["reasoning"]
        assert k in CHARTABLE_BOUNDED_KEYS["reasoning"]
    # captioning CLIPScore is the benchmark's own metric for the caption tasks
    # and is deliberately NOT demoted.
    assert "captioning_clipscore" in BOUNDED_HEADLINE_KEYS["captioning"]


def test_flag_rate_leads_the_violation_headline():
    from experiments.results_lib import BOUNDED_HEADLINE_KEYS

    head = BOUNDED_HEADLINE_KEYS["violation"][:2]
    assert head == ["violation_gt_positive_rate", "violation_pred_positive_rate"], (
        "the predicted-vs-true flag rate is the first thing to read; a model "
        "flagging 64% of images against a 13.7% base rate has a meaningless recall"
    )


def test_every_annotated_macro_names_an_n_rules_key():
    for metric, nkey in MACRO_N_RULES_KEY.items():
        assert nkey.endswith("_n_rules")
        assert nkey.startswith(metric.rsplit("_macro", 1)[0]), (metric, nkey)


def test_support_keys_cover_both_conditioned_families():
    assert any("scored_count" in k for k in SUPPORT_KEYS["reasoning"])
    assert "violation_gt_positive_image_count" in SUPPORT_KEYS["violation"]
    assert any("grounding_scored_count" in k for k in SUPPORT_KEYS["violation"])
