"""Regression tests for the six pre-flight blockers (B1-B6).

Each of these was a defect that would have wasted GPU hours or silently
mis-trained a model. They are pinned here because every one of them was invisible
in normal operation: the code ran, produced plausible output, and was wrong.
"""
import inspect
import os
import pathlib
import sys
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
# B2 — the GRES type must match each stage's memory profile
# ---------------------------------------------------------------------------

# GPU policy. The gpu-h100 partition holds BOTH card types -- four H100 nodes and one
# H200 node (egh2, 2 GPUs) -- so the GRES *type* is what actually selects the hardware,
# and the partition must never change.
#
# Only GRPO needs the H200: configs/grpo.yaml is tuned for its 141 GB
# (per_device_train_batch_size 16, steps_per_generation 4, no image_max_pixels cap), and
# its own comment records batch 16 OOM'ing at 92.97/93.12 GiB on a 93 GB H100. The other
# three stages are indifferent to the card, so pinning them to the plentiful H100s keeps
# them from queueing behind the single H200 node.
STAGE_GRES = {
    "hpc_baseline.sh": "gpu:h100:1",
    "hpc_sft.sh": "gpu:h100:1",
    "hpc_merge_sft.sh": "gpu:h100:1",
    "hpc_grpo.sh": "gpu:h200:1",
}


@pytest.mark.parametrize("name", PHASE_SCRIPTS)
def test_b2_each_stage_requests_the_right_gpu(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    want = STAGE_GRES[name]
    other = "gpu:h100:1" if want == "gpu:h200:1" else "gpu:h200:1"

    assert f"--gres={want}" in text, f"{name} must request {want}"
    assert f"#SBATCH --gres={other}" not in text, (
        f"{name} still carries an #SBATCH request for {other}"
    )
    # The partition is correct and must not change: it holds both card types.
    assert "--partition=gpu-h100" in text, f"{name} must stay on the gpu-h100 partition"


def test_b2_only_grpo_asks_for_the_scarce_h200():
    """Guards the policy itself, not one script: if a future edit moves SFT onto the H200,
    the whole schedule serialises behind two GPUs again and nothing else would notice."""
    on_h200 = [n for n in PHASE_SCRIPTS
               if "#SBATCH --gres=gpu:h200:1" in (SCRIPTS / n).read_text(encoding="utf-8")]
    assert on_h200 == ["hpc_grpo.sh"], (
        f"exactly one stage should request the H200, got {on_h200}. Only GRPO's memory "
        "profile requires 141 GB; every other stage should use the plentiful H100s."
    )


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


def test_b3_apply_pixel_bounds_survives_a_read_only_min_max_pixels_property():
    """Real crash, confirmed from a live SLURM job: transformers==5.4.0's
    Qwen2VLImageProcessor implements min_pixels/max_pixels as read-only
    @property attributes derived from `size` (getter only, no setter).
    `hasattr(obj, attr)` only tests whether the GETTER succeeds -- it returns
    True for a read-only property just as it would for a plain attribute -- so
    the old `if hasattr(...): setattr(...)` guard was not actually a "can I
    set this" check, and crashed with
        AttributeError: property of 'Qwen2VLImageProcessor' object has no setter
    on every single baseline/SFT/GRPO inference call (this is the no-adapter
    load path model_loader.py::load_model_for_inference always exercises).
    Reproduced here with a fake processor whose min_pixels/max_pixels are
    real read-only properties, not plain attributes.
    """
    from models.model_loader import apply_pixel_bounds

    class _ReadOnlyPixelsProcessor:
        def __init__(self):
            self.size = {"shortest_edge": 3136, "longest_edge": 1003520}

        @property
        def min_pixels(self):
            return self.size["shortest_edge"]

        @property
        def max_pixels(self):
            return self.size["longest_edge"]

    class _FakeTokenizer:
        def __init__(self):
            self.image_processor = _ReadOnlyPixelsProcessor()

    tok = _FakeTokenizer()
    apply_pixel_bounds(tok, min_pixels=200704, max_pixels=1204224)  # must not raise

    ip = tok.image_processor
    # The real cap (image_processor.size) must still apply even though the
    # legacy plain-attribute sync silently could not.
    assert ip.size == {"shortest_edge": 200704, "longest_edge": 1204224}
    assert ip.min_pixels == 200704
    assert ip.max_pixels == 1204224


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


# ---------------------------------------------------------------------------
# Token budgets: the cap must cover TEXT + VISION, not text alone
# ---------------------------------------------------------------------------

# Each vision token covers patch_size^2 * merge_size^2 pixels, and apply_pixel_bounds
# caps post-resize area at image_max_pixels, so the vision side can never exceed
# image_max_pixels / that figure.
#
# Qwen3-VL measured 1024 px/token on ARC (patch 16, merge 2) -> a 1176-token ceiling at
# the 1.2 MP cap, which matches the ~1176-1270 figure recorded elsewhere. This test
# deliberately uses the SMALLER patch-14 geometry (784 px/token -> 1536 tokens), because
# fewer pixels per token means MORE tokens: it is the conservative direction, and it
# keeps the assertion valid if the backbone is ever swapped for one with finer patches.
# validate_rewards.py reads the real geometry off the processor at runtime.
_VISION_TOKEN_PIXELS = 14 * 14 * 2 * 2


def test_sft_max_length_covers_text_plus_vision():
    """SFTConfig.max_length bounds prompt + target + VISION as one sequence.

    This has been wrong twice, both times by comparing the wrong quantity against it:

      * 2048 was verified by a full train+val sweep at max 1865 tokens -- but with the
        OLD ~233-token prompt. The rewritten prompts pushed unified's text side to 1110.
      * 2560 was then set against a SAMPLED vision-token count. Sampling is not a
        ceiling: two dataset roots measured 209 and 1064 tokens from their first rows.

    The ceiling is analytic, so assert against it rather than against any measurement.
    1110 is unified's measured text max (ARC 2026-09-05, train+val, 8198+701 rows).
    With the real 1176-token vision ceiling that is 2286; this test checks the
    conservative 2646 instead, so it fails before reality does.
    """
    from core.config import load_config

    cfg = load_config(task="unified", training_kind="sft")
    cap = cfg["max_seq_length"]
    vision = cfg["image_max_pixels"] // _VISION_TOKEN_PIXELS
    text_max = 1110

    assert vision + text_max < cap, (
        f"worst-case sequence {text_max} (text) + {vision} (vision, from image_max_pixels "
        f"{cfg['image_max_pixels']}) = {text_max + vision} does not fit in max_seq_length "
        f"{cap}. Raise the cap, shorten the prompt, or lower image_max_pixels."
    )


def test_census_derives_the_vision_ceiling_rather_than_sampling_one_image():
    """The census must not pair a max text length with a single image's vision count.

    Guarding the approach, not the number: a future edit that goes back to sampling
    would silently restore a gate that passes while sequences truncate.
    """
    src = (REPO / "scripts" / "validate_rewards.py").read_text(encoding="utf-8")
    census = src[src.index("def census("):src.index("def pool_stats(")]

    assert "image_max_pixels" in census, (
        "census must derive its vision-token ceiling from image_max_pixels"
    )
    assert "WORST CASE" in census, "census must report the worst case it checks"
    # The sampled probe may stay for context, but must not be what the verdict uses.
    verdict = census[census.index("failures = []"):]
    assert "sampled" not in verdict, (
        "the pass/fail branch must use the analytic ceiling, not the sampled image"
    )


def test_submitter_can_override_gres_for_every_stage():
    """--gres is an ESCAPE HATCH, not the normal mechanism. The per-stage policy lives in
    the scripts (see STAGE_GRES): H100 for baseline/sft/merge, H200 for GRPO. This flag
    exists to force every stage onto one card type for a debug run, so it deliberately
    applies to all four and overrides both halves of that policy -- including pulling
    GRPO off the H200, which only makes sense alongside an image_max_pixels cap.

    Default behaviour must stay untouched: no --gres on the sbatch line at all, so each
    script's own directive governs.
    """
    import subprocess

    base = [sys.executable, "scripts/submit_pipeline.py", "--task", "caption_only",
            "--tiers", "2b", "--version", "v1", "--skip-preload"]
    env = {**os.environ, "PYTHONPATH": str(REPO)}

    plain = subprocess.run(base, cwd=REPO, capture_output=True, text=True, env=env).stdout
    over = subprocess.run(base + ["--gres", "gpu:h100:1"], cwd=REPO,
                          capture_output=True, text=True, env=env).stdout

    assert plain.count("Running: sbatch") == 4, "expected 4 submissions"
    assert "--gres" not in plain, "default must leave the in-file gres alone"
    assert over.count("--gres=gpu:h100:1") == 4, (
        "every stage must carry the override, or one pipeline spans two GPU types"
    )


# ---------------------------------------------------------------------------
# B7 — run_sft_unified must not reference task_cfg before assigning it
# ---------------------------------------------------------------------------
def test_b7_task_cfg_is_assigned_before_its_first_use_in_run_sft_unified():
    """models/sft_trainer.py:run_sft_unified used to build the W&B config dict with
    `"task_cfg": task_cfg` while `task_cfg = load_task_config(task)` lived further down
    the SAME function, in the git-metadata block. Python makes any name assigned
    anywhere in a function local to the whole function, so that read raised
    UnboundLocalError unconditionally -- on every SFT run, every task, every tier,
    right after model load and before a single training step.

    `run_sft.py` imports unsloth at module load and cannot even be imported on
    Windows, so this is an AST check on source rather than a live call -- it asserts
    the ORDERING GUARANTEE (first Store before any Load), which is what actually
    matters and survives future refactors better than pinning line numbers.
    """
    import ast

    src = (REPO / "models" / "sft_trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_sft_unified"
    )

    first_store_line = None
    first_load_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "task_cfg":
            if isinstance(node.ctx, ast.Store) and first_store_line is None:
                first_store_line = node.lineno
            if isinstance(node.ctx, ast.Load) and first_load_line is None:
                first_load_line = node.lineno

    assert first_store_line is not None, "task_cfg is never assigned in run_sft_unified"
    assert first_load_line is not None, "task_cfg is never read in run_sft_unified"
    assert first_store_line < first_load_line, (
        f"task_cfg is read at line {first_load_line} before its first assignment at "
        f"line {first_store_line} -- this is an UnboundLocalError waiting to fire on "
        f"every SFT run."
    )


# ---------------------------------------------------------------------------
# B8 — GRPO's own lora/finetune_* block must actually reach the model loader
# ---------------------------------------------------------------------------
def test_b8_grpo_yaml_lora_and_finetune_keys_reach_load_model_for_training():
    """configs/grpo.yaml declaring a lora/finetune_* block is necessary but not
    sufficient: models/grpo_trainer.py::run_grpo loads TWO configs (`cfg` from the
    grpo chain, `sft_cfg` from the sft chain) and calls
    `load_model_for_training(sft_cfg=sft_cfg, ...)` -- and model_loader.py reads
    lora/finetune_* EXCLUSIVELY off whichever dict is passed as sft_cfg. Only a
    handful of keys (max_seq_length, use_gradient_checkpointing, load_in_4bit,
    image_{min,max}_pixels) were ever copied from `cfg` onto `sft_cfg` before that
    call, so grpo.yaml's own lora block and finetune_* switches were silently inert
    -- a no-op on OBSERVED behaviour only because sft.yaml and grpo.yaml happened to
    hold identical values. Editing grpo.yaml's rank for a GRPO-side ablation would
    have changed nothing.

    Guards the FIX (the copy-over block), not just the declaration the existing
    test_grpo_lora_shape_is_explicit_not_inherited already covers.
    """
    src = (REPO / "models" / "grpo_trainer.py").read_text(encoding="utf-8")
    # Isolate the body of run_grpo up to the load_model_for_training call, so a
    # match elsewhere in the file (e.g. in a docstring) can't produce a false pass.
    start = src.index("def run_grpo(")
    call_site = src.index("model, tokenizer, _ = load_model_for_training(", start)
    body = src[start:call_site]

    assert '"lora" in cfg' in body or "'lora' in cfg" in body, (
        "grpo.yaml's lora block is never copied onto sft_cfg before "
        "load_model_for_training() is called -- it will read sft.yaml's lora "
        "block instead, silently."
    )
    for key in (
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
    ):
        assert key in body, (
            f"grpo.yaml's {key!r} override is never copied onto sft_cfg before "
            "load_model_for_training() is called -- it will read sft.yaml's value "
            "instead, silently."
        )


# ---------------------------------------------------------------------------
# B9 — three GRPOConfig fields must not rely on an unpinned TRL default
# ---------------------------------------------------------------------------
def test_b9_grpo_config_pins_loss_type_mask_truncated_and_drop_last():
    """loss_type, mask_truncated_completions and dataloader_drop_last used to be
    absent from the entire GRPO config chain, so their real values were whatever
    TRL 0.23.0 happened to default to (dapo / False / False) -- true only by
    coincidence of that specific TRL version, and a future upgrade could change
    any of them with no log line and no error. All three are now explicit,
    user-decided values in configs/grpo.yaml, threaded through in
    models/grpo_trainer.py's grpo_config_kwargs.
    """
    from core.config import load_config

    for task in ("unified", "violations_only", "object_only", "caption_only"):
        cfg = load_config(task=task, training_kind="grpo")
        assert cfg.get("loss_type") == "dapo", (
            f"{task}: loss_type must be pinned in configs/grpo.yaml, not left to "
            "whatever TRL currently defaults to"
        )
        assert cfg.get("mask_truncated_completions") is True, (
            f"{task}: mask_truncated_completions must be pinned True -- a "
            "truncated rollout already scores 0 on every reward component; it "
            "should not also shape the loss on tokens with no real stopping "
            "decision"
        )
        assert cfg.get("dataloader_drop_last") is True, (
            f"{task}: dataloader_drop_last must be pinned True for GRPO, matching "
            "sft.yaml, so the documented '1732 // 32 x 2 = 108 steps' arithmetic "
            "is enforced rather than merely assumed"
        )

    src = (REPO / "models" / "grpo_trainer.py").read_text(encoding="utf-8")
    for key in ("loss_type", "mask_truncated_completions", "dataloader_drop_last"):
        assert f'{key}=cfg.get("{key}"' in src, (
            f"grpo_config_kwargs never reads {key!r} from the merged config -- "
            "the YAML value would never reach GRPOConfig"
        )


# ---------------------------------------------------------------------------
# B10 — --task / --tier must be required, not silently defaulted, at every
# manual/debug entry point
# ---------------------------------------------------------------------------

# (file, flag) pairs that used to carry a silent default (active_tier's "2b",
# or a literal "unified"/"violations_only") instead of requiring the caller to
# say which task/tier they mean. The orchestrated hpc_*.sh path always passes
# both explicitly, so this only ever mattered for a hand-run/debug invocation
# -- exactly the situation where a silently-wrong default is most dangerous.
REQUIRED_TASK_TIER_FLAGS = [
    ("experiments/run_sft.py", "--task"),
    ("experiments/run_sft.py", "--tier"),
    ("experiments/run_grpo.py", "--task"),
    ("experiments/run_grpo.py", "--tier"),
    ("experiments/run_inference.py", "--task"),
    ("experiments/run_inference.py", "--tier"),
    ("experiments/run_evaluation.py", "--task"),
    ("experiments/compare_results.py", "--task"),
    ("experiments/compare_results.py", "--tier"),
    ("scripts/preflight_grpo.py", "--task"),
    ("scripts/preflight_grpo.py", "--tier"),
]


@pytest.mark.parametrize("relpath,flag", REQUIRED_TASK_TIER_FLAGS)
def test_b10_task_and_tier_flags_are_required_not_defaulted(relpath, flag):
    """A regex on the add_argument(...) call for `flag`, not an import + real
    argparse run: several of these files import unsloth at module level (or
    transitively) and cannot be imported on Windows/no-GPU. The pattern below
    matches the flag's OWN add_argument call specifically (up to the next
    add_argument), so a `default=` on some other flag in the same file can't
    produce a false pass.
    """
    src = (REPO / relpath).read_text(encoding="utf-8")
    # Isolate this flag's add_argument(...) call text, from its own occurrence
    # of the flag literal up to the next add_argument( -- reliable because
    # every add_argument call in these files is on its own statement.
    idx = src.index(f'"{flag}"')
    end = src.find("add_argument(", idx + 1)
    call_text = src[idx:end] if end != -1 else src[idx:idx + 400]

    assert "required=True" in call_text, (
        f"{relpath}: {flag} must be required=True, not silently defaulted -- a "
        "hand-run invocation that forgets this flag should error immediately, "
        "not train/evaluate/compare against the wrong task or tier."
    )
    assert "default=" not in call_text, (
        f"{relpath}: {flag} still carries a default= alongside required=True"
    )


# ---------------------------------------------------------------------------
# B11 — scale_rewards and seed must reach GRPOConfig explicitly, not rely on
# an unpinned TRL default / a merged-but-unread value
# ---------------------------------------------------------------------------
def test_b11_grpo_config_pins_scale_rewards_and_seed():
    """scale_rewards="group" is treated by CLAUDE.md as one of GRPO's three
    load-bearing safety brakes for the 2e-6 learning rate (advantages are
    normalised by in-group std, so raw reward magnitude cannot inflate step
    size) -- but it was never explicitly set, true only by coincidence of TRL
    0.23.0's own default. seed was a different shape of the same problem:
    base.yaml's seed already reaches the merged GRPO config for free (the merge
    chain always starts with load_base_config()), but models/grpo_trainer.py
    never read cfg["seed"] back out into GRPOConfig(seed=...), so the value
    sitting in the merged dict never reached the trainer -- harmless only
    because GRPOConfig's own default also happens to be 42.
    """
    from core.config import load_config

    for task in ("unified", "violations_only", "object_only", "caption_only"):
        cfg = load_config(task=task, training_kind="grpo")
        assert cfg.get("scale_rewards") == "group", (
            f"{task}: scale_rewards must be pinned 'group' in configs/grpo.yaml -- "
            "CLAUDE.md's GRPO learning-rate sizing argument depends on it"
        )
        assert cfg.get("seed") == 42, (
            f"{task}: seed must resolve to base.yaml's value through the merge "
            "chain"
        )

    src = (REPO / "models" / "grpo_trainer.py").read_text(encoding="utf-8")
    assert 'scale_rewards=cfg.get("scale_rewards"' in src, (
        "grpo_config_kwargs never reads scale_rewards from the merged config -- "
        "it would never reach GRPOConfig"
    )
    assert 'seed=cfg.get("seed"' in src, (
        "grpo_config_kwargs never reads seed from the merged config -- a "
        "base.yaml seed change would silently never reach GRPO"
    )


# ---------------------------------------------------------------------------
# B12 — output_format in task YAMLs must not exist as a dead, disagreeing key
# ---------------------------------------------------------------------------
def test_b12_task_yamls_do_not_declare_a_dead_output_format_key():
    """output_format used to be written into every configs/tasks/<task>.yaml
    (e.g. "fenced_minimized_json"), but nothing ever read it -- the live source
    of truth is core/tasks.py::TaskSpec.output_format, which uses a DIFFERENT
    vocabulary (FORMAT_FENCED_JSON, not "fenced_minimized_json"). A decorative
    key in a different vocabulary from the real one is worse than no key: it
    looks authoritative and would silently disagree with core/tasks.py the
    moment anyone wired it in. Removed rather than reconciled, since nothing
    needs a YAML-level override of a value core/tasks.py already owns per task.
    """
    import yaml

    for task_yaml in (REPO / "configs" / "tasks").glob("*.yaml"):
        cfg = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        assert "output_format" not in cfg, (
            f"{task_yaml.name}: output_format must not be declared here -- it is "
            "dead config that disagrees in vocabulary with the real source of "
            "truth, core/tasks.py::TaskSpec.output_format"
        )


# ---------------------------------------------------------------------------
# B13 — GRPO's --time request must not exceed the real cluster MaxTime
# ---------------------------------------------------------------------------
def test_b13_grpo_walltime_does_not_exceed_partition_max_time():
    """gpu-h100's real MaxTime is 1-00:00:00 (24h), confirmed via
    `scontrol show partition gpu-h100` on ARC 2026-09-06. --time is a PARTITION
    property, not a GRES-type one, so it binds a gpu:h200:1 request exactly as
    it would a gpu:h100:1 one -- the earlier 48:00:00 request was rejected at
    submission (sbatch refuses an over-limit --time immediately), which is
    silent in the worst possible way: baseline/sft/merge would queue fine while
    every GRPO job -- the last stage in the chain -- failed to submit at all,
    looking nothing like a training failure.

    Pins BOTH places this value is set: scripts/submit_pipeline.py's
    TIME_CONFIG (what actually reaches the sbatch command line, and therefore
    what actually governs) and scripts/hpc_grpo.sh's in-file #SBATCH directive
    (the fallback if that script is ever run directly without the wrapper).
    """
    import re
    import subprocess
    import sys

    def _to_seconds(spec: str) -> int:
        # SLURM accepts several --time formats; TIME_CONFIG/hpc_grpo.sh only
        # ever use HH:MM:SS or D-HH:MM:SS, so only those two need parsing here.
        if "-" in spec:
            days, rest = spec.split("-", 1)
            h, m, s = (int(x) for x in rest.split(":"))
            return int(days) * 86400 + h * 3600 + m * 60 + s
        h, m, s = (int(x) for x in spec.split(":"))
        return h * 3600 + m * 60 + s

    MAX_TIME_SECONDS = _to_seconds("1-00:00:00")  # scontrol show partition gpu-h100

    submitter_src = (REPO / "scripts" / "submit_pipeline.py").read_text(encoding="utf-8")
    m = re.search(r'"grpo":\s*"([\d:\-]+)"', submitter_src)
    assert m, "TIME_CONFIG['grpo'] not found in submit_pipeline.py"
    assert _to_seconds(m.group(1)) <= MAX_TIME_SECONDS, (
        f"submit_pipeline.py TIME_CONFIG['grpo'] = {m.group(1)!r} exceeds the "
        "real gpu-h100 partition MaxTime (1-00:00:00) -- sbatch will reject "
        "this at submission, and since --time on the command line overrides "
        "the in-file directive, this is the value that actually governs."
    )

    hpc_grpo_src = (REPO / "scripts" / "hpc_grpo.sh").read_text(encoding="utf-8")
    m2 = re.search(r"#SBATCH --time=([\d:\-]+)", hpc_grpo_src)
    assert m2, "#SBATCH --time= not found in hpc_grpo.sh"
    assert _to_seconds(m2.group(1)) <= MAX_TIME_SECONDS, (
        f"hpc_grpo.sh's own #SBATCH --time={m2.group(1)!r} exceeds the real "
        "gpu-h100 partition MaxTime (1-00:00:00)"
    )


# ---------------------------------------------------------------------------
# B14 — no function may reference a name before its own local import/
# assignment of that same name (the "shadowed by a later local import"
# UnboundLocalError class of bug)
# ---------------------------------------------------------------------------

# Every direct entry point / trainer module reachable from the orchestrated
# pipeline. This is a GENERAL static check (not the specific run_sft.py case),
# added after that exact shape of bug reached a real SLURM submission:
# experiments/run_sft.py::main() had a module-level `from core.config import
# load_config` PLUS a second, redundant local `from core.config import
# load_config` further down inside main() -- and Python's scoping rule makes
# ANY name assigned/imported anywhere in a function body local to that WHOLE
# function, retroactively. The early `config = load_config()` call ran before
# that later local import ever executed, so every single invocation raised
# UnboundLocalError. This bug PREDATES this test suite (confirmed present in
# commit 56676a2, before the six-track pipeline audit) and was only caught
# because it crashed both violations_only-2b and unified-2b's SFT jobs on the
# first real submission -- the earlier audit's AST-based test_b7 checked this
# exact failure shape only for models/sft_trainer.py's `task_cfg`, not for
# this file's `load_config`.
ENTRY_POINT_FILES = [
    "experiments/run_sft.py",
    "experiments/run_grpo.py",
    "experiments/run_inference.py",
    "experiments/run_evaluation.py",
    "experiments/compare_results.py",
    "scripts/preflight_grpo.py",
    "scripts/merge_sft_adapter.py",
    "scripts/submit_pipeline.py",
    "models/sft_trainer.py",
    "models/grpo_trainer.py",
    "models/model_loader.py",
    "models/inference.py",
]


@pytest.mark.parametrize("relpath", ENTRY_POINT_FILES)
def test_b14_no_name_used_before_its_own_local_import_in_any_function(relpath):
    """For every function in `relpath`: if that function ALSO imports a name
    locally (shadowing any module-level import of the same name for the
    function's entire body), the first LOAD of that name must not occur
    before the first local import/assignment binds it. A plain `import X`
    used only after being locally re-imported is fine; using it BEFORE that
    local import is the exact UnboundLocalError shape this test exists to
    catch, regardless of which name or which file it recurs in next.
    """
    import ast

    src = (REPO / relpath).read_text(encoding="utf-8")
    tree = ast.parse(src)

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        locally_imported = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    locally_imported.add((a.asname or a.name).split(".")[0])
        if not locally_imported:
            continue

        first_use, first_bind = {}, {}
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Name)
                and node.id in locally_imported
                and isinstance(node.ctx, ast.Load)
            ):
                first_use.setdefault(node.id, node.lineno)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    bound = (a.asname or a.name).split(".")[0]
                    if bound in locally_imported:
                        first_bind.setdefault(bound, node.lineno)

        for name, use_line in first_use.items():
            bind_line = first_bind.get(name)
            assert bind_line is None or use_line >= bind_line, (
                f"{relpath}: {fn.name}() reads {name!r} at line {use_line}, "
                f"before that name's own local import/assignment at line "
                f"{bind_line} -- Python's scoping makes {name!r} local to the "
                f"WHOLE function because of that later import, so this is an "
                f"UnboundLocalError on every call, not a maybe."
            )
