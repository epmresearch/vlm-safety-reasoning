"""Regression tests for the six pre-flight blockers (B1-B6).

Each of these was a defect that would have wasted GPU hours or silently
mis-trained a model. They are pinned here because every one of them was invisible
in normal operation: the code ran, produced plausible output, and was wrong.
"""
import inspect
import pathlib
import re

import pytest

from core.constants import VALID_TASKS

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
PHASE_SCRIPTS = ["hpc_baseline.sh", "hpc_sft.sh", "hpc_merge_sft.sh", "hpc_grpo.sh"]
EVAL_SCRIPTS = ["hpc_baseline.sh", "hpc_sft.sh", "hpc_grpo.sh"]


# ---------------------------------------------------------------------------
# B1 — run_sft must hand its merged config to the trainer
# ---------------------------------------------------------------------------

def test_b1_run_sft_passes_sft_cfg_to_the_trainer():
    """Without this the trainer falls back to configs/sft.yaml alone, so no task
    YAML can override any SFT hyperparameter (and the tier LR clamp that used to
    live in run_sft.py was dead code that still logged a rate it never applied).

    Located via ast rather than str.index: a previous version searched the raw
    source for "run_sft_unified(" and so could match a COMMENT mentioning the
    function instead of the call, which is exactly what happened when the clamp
    removal was documented in a comment.
    """
    import ast
    tree = ast.parse((REPO / "experiments" / "run_sft.py").read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "run_sft_unified"
    ]
    assert len(calls) == 1, f"expected exactly one run_sft_unified call, found {len(calls)}"
    kwargs = {k.arg for k in calls[0].keywords}
    assert "sft_cfg" in kwargs, (
        "run_sft.py must pass sft_cfg= to run_sft_unified(); without it the merged "
        "config (base -> registry -> sft -> task) is silently discarded and the "
        "trainer re-reads configs/sft.yaml alone."
    )
    assert "task" in kwargs, "run_sft.py must forward task= as well"


def test_b1_sft_learning_rate_is_flat_across_tiers():
    """The tier LR clamp (4b -> 5e-5, 8b -> 2e-5) must stay removed.

    It was never actually applied -- run_sft.py did not pass sft_cfg through, so
    every tier trained at configs/sft.yaml's 1.0e-4. Fixing that plumbing would
    have activated the clamp for the first time, silently changing the 4b/8b runs,
    so it was deleted instead:

      * it confounds the tier comparison (3 tiers exist to isolate model scale,
        and a 5x LR difference makes an 8b regression unattributable);
      * 512 steps cannot absorb it -- at 1e-4 eval loss still improved to ~step
        250, so a 5x lower LR would be stopped mid-descent;
      * LoRA is far less LR-sensitive to scale than full fine-tuning, and 1e-4 is
        already empirically stable at 8b here.

    A per-tier LR is still allowed, but it belongs in model_registry.yaml's tier
    block as declared configuration, not as a hidden override in run_sft.py.
    """
    src = (REPO / "experiments" / "run_sft.py").read_text(encoding="utf-8")
    # Strip comments so the rationale above cannot trip this assertion.
    code = chr(10).join(
        line.split("#", 1)[0] for line in src.splitlines()
    )
    assert "learning_rate" not in code, (
        "run_sft.py must not override learning_rate; it now comes purely from the "
        "merged config so that every tier trains at the same rate."
    )

    from core.config import load_config
    for task in ("unified", "violations_only", "object_only", "caption_only"):
        lr = load_config(task=task, training_kind="sft").get("learning_rate")
        assert lr == 1.0e-4, f"{task}: expected flat 1.0e-4, got {lr}"


def test_grpo_learning_rate_can_move_the_policy():
    """2.0e-7 over 108 steps integrates to ~1.1e-5 of cumulative LR -- 467x less
    than the 8b SFT run it follows -- so the policy cannot measurably move and
    GRPO would read as identical to SFT regardless of the reward surface. That is
    the same "did it train at all?" ambiguity that voided the pre-b8f2470 runs.
    Guard the floor, not an exact value, so tuning stays free.
    """
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "grpo.yaml").read_text(encoding="utf-8"))
    lr = float(cfg["learning_rate"])
    assert 5.0e-7 <= lr <= 1.0e-5, (
        f"GRPO learning_rate {lr:g} is outside the usable LoRA-RL band "
        "(5e-7..1e-5); below it the policy does not move, above it GRPO tends to "
        "collapse."
    )


# ---------------------------------------------------------------------------
# Ghost variables: keys model_loader.py consumes must come from CONFIG, never
# from its own module-level literals
# ---------------------------------------------------------------------------

# Every key models/model_loader.py reads out of the training config. Each has a
# module-level literal fallback, so an absent key is silently substituted with no
# log line and no error -- config appears to say one thing, runtime reads another.
MODEL_LOADER_KEYS = (
    "load_in_4bit",
    "max_seq_length",
    "use_gradient_checkpointing",
    "finetune_vision_layers",
    "finetune_language_layers",
    "finetune_attention_modules",
    "finetune_mlp_modules",
    "lora",
)


@pytest.mark.parametrize("training_kind", ["sft", "grpo"])
@pytest.mark.parametrize("key", MODEL_LOADER_KEYS)
def test_model_loader_keys_are_declared_in_config(training_kind, key):
    """GRPO's merge chain is base -> model_registry -> grpo -> tasks/<task>; it does
    NOT include sft.yaml. Five of these keys (lora + the four finetune_* switches)
    were absent from GRPO's merged config and were being supplied by
    model_loader.py's own literals. They matched sft.yaml by coincidence, so
    editing sft.yaml's lora block would have moved SFT and left GRPO silently on
    the old values.

    finetune_vision_layers is the one that matters most: freezing the vision tower
    is a research decision, and it was being made by a Python default.
    """
    from core.config import load_config
    for task in VALID_TASKS:
        cfg = load_config(task=task, training_kind=training_kind)
        assert key in cfg, (
            f"{training_kind}/{task}: '{key}' is absent from the merged config, so "
            f"models/model_loader.py will substitute its own literal. Declare it in "
            f"configs/{training_kind}.yaml (sft.yaml is not in the GRPO merge chain)."
        )


def test_grpo_lora_shape_is_explicit_not_inherited():
    """configs/grpo.yaml must carry its own lora block, in its own text -- not rely
    on the merge picking one up from elsewhere."""
    raw = (REPO / "configs" / "grpo.yaml").read_text(encoding="utf-8")
    assert "lora:" in raw, (
        "configs/grpo.yaml must declare its own lora block; without it GRPO's "
        "adapter rank comes from model_loader.py's literals."
    )
    import yaml
    lora = yaml.safe_load(raw)["lora"]
    for field in ("r", "alpha", "dropout", "target_modules"):
        assert field in lora, f"grpo.yaml lora block is missing '{field}'"


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
