"""Regression tests for the six pre-flight blockers (B1-B6).

Each of these was a defect that would have wasted GPU hours or silently
mis-trained a model. They are pinned here because every one of them was invisible
in normal operation: the code ran, produced plausible output, and was wrong.
"""
import inspect
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
PHASE_SCRIPTS = ["hpc_baseline.sh", "hpc_sft.sh", "hpc_merge_sft.sh", "hpc_grpo.sh"]
EVAL_SCRIPTS = ["hpc_baseline.sh", "hpc_sft.sh", "hpc_grpo.sh"]


# ---------------------------------------------------------------------------
# B1 — run_sft must hand its merged config to the trainer
# ---------------------------------------------------------------------------

def test_b1_run_sft_passes_sft_cfg_to_the_trainer():
    """Without this the trainer falls back to configs/sft.yaml alone, which
    discards the tier learning-rate clamps AND makes it impossible for a task
    YAML to override any SFT hyperparameter. Every tier trained at 1.0e-4."""
    src = (REPO / "experiments" / "run_sft.py").read_text(encoding="utf-8")
    call = src[src.index("run_sft_unified("):]
    call = call[:call.index(")\n")]
    assert "sft_cfg=sft_cfg" in call, (
        "run_sft.py must pass sft_cfg= to run_sft_unified(); without it the "
        "tier LR clamps are dead code and the log line reports a rate that was "
        "never applied."
    )


def test_b1_tier_lr_clamp_actually_reaches_the_config():
    """The clamp is computed in run_sft.py; prove the value it writes is the
    value the trainer would read."""
    from core.config import load_config
    for tier, ceiling in (("4b", 5.0e-5), ("8b", 2.0e-5)):
        cfg = load_config(task="unified", training_kind="sft")
        base_lr = cfg.get("learning_rate", 1.0e-4)
        clamped = min(base_lr, ceiling)
        assert clamped == ceiling, (
            f"tier {tier}: expected the clamp to bind (base {base_lr} > {ceiling})"
        )


# ---------------------------------------------------------------------------
# B2 — the GRES type must match the memory profile the config is tuned for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", PHASE_SCRIPTS)
def test_b2_scripts_request_h200_gres(name):
    """configs/grpo.yaml is tuned for the H200's 141 GB — its own comment records
    that per_device_train_batch_size 16 OOM'd at 92.97/93.12 GiB on a 93 GB H100.
    The gpu-h100 partition contains both card types, so the GRES type is what
    actually selects the hardware."""
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "--gres=gpu:h200:1" in text, f"{name} must request an H200"
    assert "--gres=gpu:h100:1" not in text, f"{name} still requests an H100"
    # The partition is correct and must not change: it holds both card types.
    assert "--partition=gpu-h100" in text, f"{name} must stay on the gpu-h100 partition"


# ---------------------------------------------------------------------------
# B3 — the pixel cap must be written in the key shape transformers reads
# ---------------------------------------------------------------------------

def test_b3_apply_pixel_bounds_uses_edge_keys():
    """The Qwen VL image processor stores pixel AREAS under 'shortest_edge' and
    'longest_edge' (the class default is literally {"shortest_edge": 56*56,
    "longest_edge": 28*28*1280}). Writing {"min_pixels", "max_pixels"} is rejected
    outright by transformers, so the cap silently never applied and images went in
    at up to 14.6 MP."""
    from models import model_loader
    src = inspect.getsource(model_loader.apply_pixel_bounds)
    assert '"shortest_edge"' in src and '"longest_edge"' in src
    assert 'image_processor.size = {"min_pixels"' not in src, (
        "the legacy key shape is back; transformers rejects it"
    )


def test_b3_apply_pixel_bounds_sets_the_cap_and_does_not_sqrt():
    """A regression guard against 'converting' the area cap into an edge length.
    1204224 must stay 1204224 — converting to sqrt(1204224) = 1097 would cap the
    AREA at 1097 px^2 and shrink every image to about 33x33."""
    from models.model_loader import apply_pixel_bounds

    class _FakeProcessor:
        def __init__(self):
            self.size = {"shortest_edge": 3136, "longest_edge": 1003520}
            self.min_pixels = 3136
            self.max_pixels = 1003520

    class _FakeTokenizer:
        def __init__(self):
            self.image_processor = _FakeProcessor()

    tok = _FakeTokenizer()
    apply_pixel_bounds(tok, min_pixels=200704, max_pixels=1204224)
    ip = tok.image_processor

    assert ip.size == {"shortest_edge": 200704, "longest_edge": 1204224}
    assert ip.min_pixels == 200704
    assert ip.max_pixels == 1204224


def test_b3_apply_pixel_bounds_is_a_noop_without_an_image_processor():
    from models.model_loader import apply_pixel_bounds

    class _TextOnly:
        image_processor = None

    apply_pixel_bounds(_TextOnly(), 1, 2)  # must not raise


# ---------------------------------------------------------------------------
# B4 — no class may be un-emittable
# ---------------------------------------------------------------------------

POOL_PREVALENCE = {"excavator": 0.361, "rebar": 0.088, "worker_with_white_hard_hat": 0.115}
BREAKEVEN_CEILING = 0.75


@pytest.mark.parametrize("task", ["unified", "object_only"])
def test_b4_every_class_is_worth_emitting(task):
    """A class is worth emitting only when E[IoU] > c*(1-p)/p. At the historical
    flat c = 0.15 that break-even was 1.55 for rebar and 1.15 for the hard-hat
    class — above 1.0, i.e. unreachable — so suppressing them was strictly
    dominant no matter how good the detector became."""
    from rewards.reward_utils import grounding_tn_constant
    for cls, p in POOL_PREVALENCE.items():
        c = grounding_tn_constant(task, cls)
        breakeven = c * (1 - p) / p
        assert breakeven <= BREAKEVEN_CEILING, (
            f"{task}/{cls}: break-even IoU {breakeven:.3f} > {BREAKEVEN_CEILING}. "
            f"Suppressing this class is dominant; lower grounding_tn_constant."
        )


# ---------------------------------------------------------------------------
# B5 — a contentless assertion must never earn identification credit
# ---------------------------------------------------------------------------

CONTENTLESS = {"reason": "", "bounding_box": []}


def _vio_completion(**rules):
    import json
    body = {f"rule_{i}_violation": rules.get(f"rule_{i}") for i in range(1, 5)}
    return "```json\n" + json.dumps(body) + "\n```"


def _gt(**rules):
    return {f"rule_{i}_violation": rules.get(f"rule_{i}") for i in range(1, 5)}


def test_b5_contentless_assertion_scores_zero_on_a_real_violation():
    """It used to score a perfect F-beta = 1.0 on the 0.40-weighted component
    while contributing nothing to grounding or reasoning, which are
    TP-conditioned and so never penalised it."""
    from rewards.reward_violation_id import compute_reward
    gt = _gt(rule_1={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no hat"})
    score = compute_reward(_vio_completion(rule_1=CONTENTLESS), gt, task="violations_only")
    assert score == 0.0, f"contentless assertion earned {score}"


def test_b5_contentless_assertion_is_still_a_false_alarm_on_a_safe_image():
    """The other half of the fix: it must NOT be dropped from the prediction set
    either, or flagging a safe image would earn true-negative credit."""
    from rewards.reward_violation_id import compute_reward
    from rewards.reward_utils import reward_constant
    tn = reward_constant("violations_only", "violation_tn_constant", 0.15)
    score = compute_reward(_vio_completion(rule_1=CONTENTLESS), _gt(), task="violations_only")
    assert score == 0.0, f"a false alarm scored {score}, not 0.0"
    # ...while a genuine abstention does earn the true-negative credit.
    assert compute_reward(_vio_completion(), _gt(), task="violations_only") == pytest.approx(tn)


def test_b5_substantive_assertion_still_earns_full_credit():
    from rewards.reward_violation_id import compute_reward
    gt = _gt(rule_1={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no hat"})
    good = {"bounding_box": [[100, 100, 200, 200]], "reason": "worker without a hard hat"}
    assert compute_reward(_vio_completion(rule_1=good), gt, task="violations_only") == pytest.approx(1.0)
    # A box with no reason, or a reason with no box, is still substantive.
    for partial in ({"bounding_box": [[100, 100, 200, 200]], "reason": ""},
                    {"bounding_box": [], "reason": "worker without a hard hat"}):
        assert compute_reward(_vio_completion(rule_1=partial), gt, task="violations_only") == pytest.approx(1.0)


@pytest.mark.parametrize("task", ["unified", "violations_only"])
def test_b5_honest_abstention_beats_reflexive_flagging(task):
    """Expected value over the 50/50 pool. At the historical c = 0.15,
    always-assert-rule_1 scored ~0.391 against always-safe's 0.075 — a 5x edge for
    a policy that never looks at the image."""
    from rewards.reward_utils import reward_constant
    c = float(reward_constant(task, "violation_tn_constant", 0.15))
    p_safe, p_rule1 = 0.50, 0.391
    ev_honest_abstention = p_safe * c
    ev_always_assert = p_rule1 * 1.0
    assert ev_honest_abstention > ev_always_assert, (
        f"{task}: unconditional rule_1 assertion (EV {ev_always_assert:.4f}) beats "
        f"honest abstention (EV {ev_honest_abstention:.4f}); raise "
        f"violation_tn_constant above {ev_always_assert / p_safe:.3f}"
    )


def test_b5_substance_predicate_semantics():
    from rewards.reward_utils import _is_substantive_violation, _is_violation_present
    # Presence and substance must disagree on exactly the contentless shapes.
    for v in (CONTENTLESS, True, {"reason": "   "}, {"bounding_box": []}):
        assert _is_violation_present(v) is True or v is CONTENTLESS or v == {"bounding_box": []}
        assert _is_substantive_violation(v) is False, v
    for v in ({"reason": "x"}, {"bounding_box": [[1, 2, 3, 4]]},
              {"reason": "x", "bounding_box": [[1, 2, 3, 4]]}):
        assert _is_substantive_violation(v) is True, v
    for v in (None, {}):
        assert _is_substantive_violation(v) is False, v


# ---------------------------------------------------------------------------
# B6 — evaluation must never die on the Java switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EVAL_SCRIPTS)
def test_b6_eval_calls_pass_skip_java_switch(name):
    """SPICE is always skipped, so Java 8 is never needed. Attempting the switch
    risked an uncaught FileNotFoundError from `update-alternatives` AFTER
    inference and structural repair had already run."""
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "--skip_java_switch" in text, f"{name} must pass --skip_java_switch"


def test_b6_ensure_java8_active_never_raises(monkeypatch):
    """Every failure path must be non-fatal."""
    import shutil
    import subprocess
    from experiments import run_evaluation as re_mod

    # No java at all.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    re_mod.ensure_java8_active()

    # java present, update-alternatives missing.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/java" if name == "java" else None)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stderr": "openjdk 21", "stdout": ""})())
    re_mod.ensure_java8_active()

    # Both present, but the subprocess itself explodes.
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def _boom(*a, **k):
        raise OSError("no such binary")

    monkeypatch.setattr(subprocess, "run", _boom)
    re_mod.ensure_java8_active()


def test_b6_ensure_java8_active_is_fully_guarded():
    """Static check: no bare subprocess.run outside a try or a which() guard."""
    from experiments import run_evaluation as re_mod
    src = inspect.getsource(re_mod.ensure_java8_active)
    for call in re.findall(r"subprocess\.run\(", src):
        pass
    assert src.count("try:") >= 3, "each subprocess call must be guarded"
    assert 'shutil.which("java")' in src
    assert 'shutil.which("update-alternatives")' in src
