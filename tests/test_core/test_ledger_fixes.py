"""Regression tests for the bug-ledger fixes (BUG-08 .. BUG-29).

BUG-01..06 are the blockers, pinned in test_blocker_fixes.py.
BUG-07 (auto-resume / duplicate records) is pinned in
tests/test_grpo/test_inference_no_resume.py.
BUG-13..15 (SFT max_length, Mode-2 deletion, repetition penalty) are pinned here and
in tests/test_rewards/test_unified_reward.py.
"""
import inspect
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


# ---------------------------------------------------------------------------
# BUG-08 — eval log / manifest name must be run-scoped
# ---------------------------------------------------------------------------

def test_bug08_eval_log_name_is_run_scoped():
    """output_dir.name is the literal "evaluation_results" for every task and
    phase, so all twelve eval steps of a 4-task run wrote the same filename."""
    src = (REPO / "experiments" / "run_evaluation.py").read_text(encoding="utf-8")
    assert "run_label" in src
    assert 'f"evaluation_{output_dir.name}_{timestamp}.txt"' not in src
    assert 'f"evaluation_{run_label}_{timestamp}.txt"' in src
    # the manifest inherited the same defect
    assert 'f"evaluation_{output_dir.name}"' not in src


# ---------------------------------------------------------------------------
# BUG-09 / BUG-10 — analysis outputs must be task+version namespaced
# ---------------------------------------------------------------------------

def test_bug09_comparison_table_is_namespaced():
    src = (REPO / "experiments" / "compare_results.py").read_text(encoding="utf-8")
    assert '"comparison_table.csv"' not in src, "the bare filename is back"
    assert "comparison_table_" in src and "task_prefix" in src


def test_bug10_plots_dir_is_namespaced():
    src = (REPO / "experiments" / "plot_metrics.py").read_text(encoding="utf-8")
    assert 'f"plots_{args.tier}"' not in src, "tier-only PLOTS_DIR is back"
    assert "task_prefix(args.task)" in src
    assert "args.version" in src


def test_bug21_plot_metrics_probes_repair_applied():
    src = (REPO / "experiments" / "plot_metrics.py").read_text(encoding="utf-8")
    assert '"repair_applied" / "repair_report.json"' in src


# ---------------------------------------------------------------------------
# BUG-11 — image_id must reach the ground-truth dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["unified", "violations_only", "object_only", "caption_only"])
def test_bug11_gt_carries_image_id(task):
    """Without this, evaluator.py labels every failure "unknown_0", "unknown_1", …
    and json_parse_failures.json cannot be traced to a source image."""
    from data.preprocessor import build_gt_dict
    raw = {
        "image_id": "IMG_0042", "image_caption": "c",
        "illumination": "normal lighting", "camera_distance": "mid distance",
        "view": "ground view", "quality_of_info": "rich",
        "rule_1_violation": None, "rule_2_violation": None,
        "rule_3_violation": None, "rule_4_violation": None,
        "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
    }
    assert build_gt_dict(raw, task=task)["image_id"] == "IMG_0042"


def test_bug11_failures_report_the_real_image_id():
    from unittest.mock import patch
    from PIL import Image
    from evaluation.evaluator import run_full_evaluation

    refs = [{"image_id": "IMG_7", "excavator": [], "rebar": [],
             "worker_with_white_hard_hat": []}]
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True):
        res = run_full_evaluation(["total garbage"], refs,
                                  images=[Image.new("RGB", (8, 8))], task="object_only")
    assert res["failures"][0]["image_id"] == "IMG_7"
    assert not res["failures"][0]["image_id"].startswith("unknown")


# ---------------------------------------------------------------------------
# BUG-12 / BUG-13 — token ceilings must be the ENFORCING values
# ---------------------------------------------------------------------------

def test_bug12_generate_batch_accepts_and_uses_a_prompt_cap():
    from models.inference import generate_batch
    sig = inspect.signature(generate_batch)
    assert "max_prompt_length" in sig.parameters
    src = inspect.getsource(generate_batch)
    assert 'tokenizer_kwargs["max_length"] = max_prompt_length' in src


def test_bug12_run_inference_derives_the_cap_from_the_window():
    src = (REPO / "experiments" / "run_inference.py").read_text(encoding="utf-8")
    assert "max_prompt_length = max(1, inference_window - max_new_tokens)" in src
    assert "max_prompt_length=max_prompt_length" in src


def test_bug13_sft_sets_max_length_not_max_seq_length():
    """Confirmed against trl==0.23.0 on ARC: SFTTrainer.__init__ has no
    max_seq_length parameter and no **kwargs; SFTConfig exposes max_length."""
    src = (REPO / "models" / "sft_trainer.py").read_text(encoding="utf-8")
    assert 'max_length=sft_cfg.get("max_seq_length", 2048)' in src, \
        "SFTConfig.max_length is not being set"
    assert '"max_seq_length": sft_cfg.get' not in src, \
        "max_seq_length is being passed as a trainer kwarg again (silently dropped)"


# ---------------------------------------------------------------------------
# BUG-16 / BUG-17 — evaluation must not do pointless work
# ---------------------------------------------------------------------------

def test_bug16_spice_cache_is_gated():
    src = (REPO / "experiments" / "run_evaluation.py").read_text(encoding="utf-8")
    idx = src.index("restore_spice_cache(SPICE_CACHE_DIR)")
    window = src[max(0, idx - 400):idx]
    assert "if not args.skip_spice" in window, \
        "restore_spice_cache still runs unconditionally (it imports pycocoevalcap)"


def test_bug17_image_map_is_capability_gated():
    src = (REPO / "experiments" / "run_evaluation.py").read_text(encoding="utf-8")
    assert "needs_images" in src
    assert "task_has(args.task, CAP_CAPTION) or task_has(args.task, CAP_VIOLATIONS)" in src


# ---------------------------------------------------------------------------
# BUG-18 — blank predictions are excluded and reported, never faked
# ---------------------------------------------------------------------------

def test_bug18_blanks_never_reach_the_graders():
    from unittest.mock import patch
    import evaluation.metrics_captioning as m

    with patch.object(m, "compute_bertscore", return_value={"bertscore_f1": 0.9}) as bs, \
         patch.object(m, "compute_meteor", return_value={}), \
         patch.object(m, "compute_cider", return_value={}), \
         patch.object(m, "compute_clipscore", return_value={}), \
         patch.object(m, "compute_long_clipscore", return_value={}):
        res = m.compute_all_caption_metrics(
            ["real caption", "", "   "], ["ref a", "ref b", "ref c"],
            images=["i1", "i2", "i3"], include_spice=False)

    passed_preds = bs.call_args[0][0]
    assert passed_preds == ["real caption"]
    assert "empty" not in passed_preds, "the fake 'empty' token is back"
    assert res["blank_prediction_count"] == 2
    assert res["blank_prediction_rate"] == pytest.approx(2 / 3)
    assert res["scored_count"] == 1
    assert res["total_count"] == 3


def test_bug18_images_stay_aligned_with_kept_predictions():
    from unittest.mock import patch
    import evaluation.metrics_captioning as m
    with patch.object(m, "compute_bertscore", return_value={}), \
         patch.object(m, "compute_meteor", return_value={}), \
         patch.object(m, "compute_cider", return_value={}), \
         patch.object(m, "compute_clipscore", return_value={}) as clip, \
         patch.object(m, "compute_long_clipscore", return_value={}):
        m.compute_all_caption_metrics(
            ["", "kept", ""], ["a", "b", "c"], images=["i0", "i1", "i2"],
            include_spice=False)
    preds, imgs = clip.call_args[0][0], clip.call_args[0][1]
    assert preds == ["kept"] and imgs == ["i1"], "text/image pairing desynchronised"


# ---------------------------------------------------------------------------
# BUG-19 — long-context CLIPScore, and it must never crash an eval
# ---------------------------------------------------------------------------

def test_bug19_long_clipscore_exists():
    import evaluation.metrics_captioning as m
    assert hasattr(m, "compute_long_clipscore")
    assert hasattr(m, "_chunk_by_token_budget")


def test_bug19_no_third_party_remote_code():
    """jinaai/jina-clip-v1 was tried first and is incompatible with transformers 5.x:
    its remote eva_model.py calls .item() on a tensor created under transformers'
    meta-device init, raising "Tensor.item() cannot be called on meta tensors".
    The chunked implementation reuses the already-cached CLIP weights instead, so
    there is no remote code and nothing extra to download."""
    import inspect
    import evaluation.metrics_captioning as m
    src = inspect.getsource(m.compute_long_clipscore)
    assert "trust_remote_code" not in src
    assert "_get_clip_model()" in src, "should reuse the standard CLIP model"
    setup = (SCRIPTS / "setup_arc.sh").read_text(encoding="utf-8")
    assert "jinaai/jina-clip-v1" not in setup.split("# Why not")[0]


class TestChunking:
    """The chunker must be lossless and must respect CLIP's 75-content-token budget."""

    @staticmethod
    def _proc():
        from unittest.mock import MagicMock

        class _Tok:  # 1 token per whitespace-separated word, deterministic
            def __call__(self, s, add_special_tokens=True):
                return {"input_ids": list(range(len(s.split())))}

        proc = MagicMock()
        proc.tokenizer = _Tok()
        return proc

    def test_short_caption_is_a_single_chunk(self):
        import evaluation.metrics_captioning as m
        text = " ".join(["word"] * 40)
        assert m._chunk_by_token_budget(text, self._proc()) == [text]

    def test_long_caption_splits_within_budget(self):
        import evaluation.metrics_captioning as m
        text = " ".join(f"w{i}" for i in range(200))
        chunks = m._chunk_by_token_budget(text, self._proc())
        assert len(chunks) == 3
        assert all(len(c.split()) <= 75 for c in chunks)

    def test_chunking_is_lossless_and_ordered(self):
        """No word may be dropped, duplicated, or reordered."""
        import evaluation.metrics_captioning as m
        text = " ".join(f"w{i}" for i in range(200))
        assert " ".join(m._chunk_by_token_budget(text, self._proc())) == text

    def test_blank_caption_yields_no_chunks(self):
        import evaluation.metrics_captioning as m
        assert m._chunk_by_token_budget("   ", self._proc()) == []


def test_bug19_long_clipscore_omits_itself_when_disabled(monkeypatch):
    """Omitted (no key), not 0.0 — an absent metric must not read as a bad score."""
    import evaluation.metrics_captioning as m
    monkeypatch.setenv("VLM_DISABLE_LONG_CLIP", "1")
    assert m.compute_long_clipscore(["a caption"], ["img"]) == {}


def test_bug19_long_clipscore_omits_itself_on_failure(monkeypatch):
    import evaluation.metrics_captioning as m
    monkeypatch.delenv("VLM_DISABLE_LONG_CLIP", raising=False)
    monkeypatch.setattr(m, "_get_clip_model", lambda: (_ for _ in ()).throw(OSError("no weights")))
    assert m.compute_long_clipscore(["a caption"], ["img"]) == {}


# ---------------------------------------------------------------------------
# BUG-20 — a dependency must never be silently dropped
# ---------------------------------------------------------------------------

def test_bug20_unparseable_job_id_exits():
    src = (SCRIPTS / "submit_pipeline.py").read_text(encoding="utf-8")
    assert 'return "UNKNOWN"' not in src, "the sentinel is back"
    assert 'dependencies=[sft_job] if sft_job != "UNKNOWN"' not in src
    assert "dependencies=[sft_job]," in src
    assert "dependencies=[merge_job]," in src


# ---------------------------------------------------------------------------
# BUG-22 / BUG-27 — stale scripts are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "scripts/test_grpo.sh",
    "scripts/test_eval_grpo.sh",
    "analyze_metrics.py",
])
def test_bug22_27_stale_entrypoints_removed(path):
    assert not (REPO / path).exists(), f"{path} is back; it was task-blind/hardcoded"


# ---------------------------------------------------------------------------
# BUG-23 — the local analysis layout has a documented producer
# ---------------------------------------------------------------------------

def test_bug23_fetch_results_exists_and_uses_the_shared_namer():
    p = SCRIPTS / "fetch_results.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    assert "results_dir_names" in src


# ---------------------------------------------------------------------------
# BUG-24 / BUG-26 — SLURM script hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hpc_baseline.sh", "hpc_sft.sh",
                                  "hpc_merge_sft.sh", "hpc_grpo.sh"])
def test_bug24_every_script_declares_mem(name):
    """A hand-run `sbatch scripts/hpc_grpo.sh ...` used to get the partition
    default, because --mem came only from the submitter's CLI."""
    src = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" in src


@pytest.mark.parametrize("name", ["hpc_baseline.sh", "hpc_sft.sh",
                                  "hpc_merge_sft.sh", "hpc_grpo.sh"])
def test_bug26_nvidia_smi_cannot_abort_the_job(name):
    src = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "nvidia-smi || true" in src, \
        "bare nvidia-smi under `set -e` can abort the job before anything is logged"


# ---------------------------------------------------------------------------
# BUG-25 / BUG-29 — provenance
# ---------------------------------------------------------------------------

def test_bug25_grpo_manifest_records_the_kl_base():
    """base_model_override is what makes the KL reference the SFT policy rather
    than the raw pretrained base (invariant #2). It was recorded nowhere."""
    src = (REPO / "models" / "grpo_trainer.py").read_text(encoding="utf-8")
    manifest = src[src.index("manifest = {"):src.index("manifest = {") + 1200]
    for key in ('"base_model_override"', '"git_commit"', '"reward_components"',
                '"reward_weights"', '"prompts"', '"task"'):
        assert key in manifest, f"GRPO manifest is missing {key}"


def test_bug29_wandb_configs_are_complete():
    grpo = (REPO / "models" / "grpo_trainer.py").read_text(encoding="utf-8")
    assert '"sft_cfg": sft_cfg' in grpo, "GRPO W&B config omits what model it loaded"
    sft = (REPO / "models" / "sft_trainer.py").read_text(encoding="utf-8")
    assert '"task_cfg": task_cfg' in sft, "SFT W&B config omits the task config"


# ---------------------------------------------------------------------------
# BUG-28 — unified pins its own GRPO budgets
# ---------------------------------------------------------------------------

def test_bug28_unified_declares_its_grpo_budgets():
    from core.config import load_task_config
    cfg = load_task_config("unified")
    assert cfg["max_completion_length"] == cfg["max_new_tokens"], \
        "unified's GRPO completion budget and inference budget must match"
    assert "max_prompt_length" in cfg


@pytest.mark.parametrize("task", ["unified", "violations_only", "object_only", "caption_only"])
def test_all_tasks_pin_their_own_token_budgets(task):
    from core.config import load_task_config
    cfg = load_task_config(task)
    for key in ("max_new_tokens", "max_completion_length",
                "max_prompt_length", "inference_max_seq_length"):
        assert key in cfg, f"{task} does not pin {key}"
    assert cfg["max_prompt_length"] + cfg["max_completion_length"] <= 3250, \
        f"{task} exceeds the GRPO max_seq_length of 3250"
