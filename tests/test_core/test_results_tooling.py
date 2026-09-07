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
    "vo-sft-2b-v4",          # missing the required _best suffix
    "vo-grpo-2b-v4",         # missing the required _final suffix
    "vo-sft-2b-v4_final",    # wrong suffix for this phase
    "vo-baseline-2b-v1_best",  # baseline must NOT carry a suffix
    "notatask-baseline-2b-v1",  # unregistered prefix
    "vo-training-2b-v1",     # unregistered phase
    "plots_vo_v4",           # not a run directory at all
    "logs",
])
def test_parse_run_name_rejects_legacy_and_malformed_names(bad_name):
    assert parse_run_name(bad_name) is None


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
