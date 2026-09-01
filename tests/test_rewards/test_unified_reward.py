"""
Tests for rewards/unified_reward.py — the component registry and the single
task-aware batch-wrapper path.

The old "Mode 2" composite reward (compute_reward, compute_reward_with_breakdown,
build_grpo_reward_fn, get_reward_funcs_and_weights, _make_batch_reward) has been
deleted. It existed as a fallback for TRL versions without reward_weights support,
but it ignored the task's `reward_components` entirely and always scored all six
components at *unified* weights — so any non-unified task that fell into it would
have trained silently against the wrong objective. trl==0.23.0 supports
reward_weights natively, making the fallback unreachable dead weight with a live
footgun in it.

What replaced its one genuinely useful behaviour: the repetition-pathology penalty,
which only ever existed inside Mode 2 and therefore never fired in production, now
lives in the live path (`_apply_repetition_penalty`).
"""
import json

import pytest

from rewards.unified_reward import (
    REWARD_COMPONENTS,
    ALL_REWARD_COMPONENTS,
    get_reward_funcs_for_task,
    _apply_repetition_penalty,
)


def _make_valid_completion(caption="A construction site.", **overrides):
    payload = {
        "caption": caption,
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    payload.update(overrides)
    return "```json\n" + json.dumps(payload) + "\n```"


GT_SAFE = {
    "caption": "A construction site.",
    "rule_1_violation": None,
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
    "excavator": [],
    "rebar": [],
    "worker_with_white_hard_hat": [],
}


class TestComponentRegistry:
    def test_registry_has_six_components(self):
        assert len(REWARD_COMPONENTS) == 6

    def test_registry_names(self):
        names = [n for n, _, _ in REWARD_COMPONENTS]
        assert names == [
            "reward_format",
            "reward_caption",
            "reward_grounding",
            "reward_violation_id",
            "reward_violation_grounding",
            "reward_reasoning",
        ]

    def test_default_weights_sum_to_one(self):
        assert sum(w for _, _, w in REWARD_COMPONENTS) == pytest.approx(1.0)

    def test_all_reward_components_mirrors_the_registry(self):
        assert set(ALL_REWARD_COMPONENTS) == {n for n, _, _ in REWARD_COMPONENTS}


class TestMode2IsGone:
    """Static guard: the composite fallback must not come back."""

    @pytest.mark.parametrize("name", [
        "compute_reward",
        "compute_reward_with_breakdown",
        "build_grpo_reward_fn",
        "get_reward_funcs_and_weights",
        "_make_batch_reward",
    ])
    def test_symbol_is_deleted(self, name):
        import rewards.unified_reward as m
        assert not hasattr(m, name), (
            f"{name} is back. It ignored the task's reward_components and scored all "
            "six at unified weights."
        )

    def test_grpo_trainer_has_no_capability_probe(self):
        """Read the source as text: importing models.grpo_trainer pulls in wandb,
        which is an HPC-only dependency."""
        import pathlib as _p
        src = (_p.Path(__file__).resolve().parents[2] / "models" / "grpo_trainer.py").read_text(
            encoding="utf-8")
        assert "_check_trl_supports_reward_weights" not in src
        assert "use_native_weights" not in src
        assert "build_grpo_reward_fn" not in src


class TestBatchRewardFunctions:
    """The wrappers get_reward_funcs_for_task returns must match TRL's contract."""

    def test_returns_matching_lengths(self):
        funcs, weights = get_reward_funcs_for_task("unified")
        assert len(funcs) == len(weights) == 6

    def test_weights_are_positive(self):
        _, weights = get_reward_funcs_for_task("unified")
        assert all(w > 0 for w in weights)

    def test_batch_fn_returns_list_of_floats(self):
        funcs, _ = get_reward_funcs_for_task("unified")
        completions = [_make_valid_completion(), _make_valid_completion()]
        gts = [json.dumps(GT_SAFE)] * 2
        for fn in funcs:
            result = fn(completions=completions, ground_truth=gts)
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(s, float) for s in result)

    def test_batch_fn_none_ground_truth_returns_zeros(self):
        funcs, _ = get_reward_funcs_for_task("unified")
        for fn in funcs:
            assert fn(completions=[_make_valid_completion()], ground_truth=None) == [0.0]

    def test_batch_fn_accepts_prompts_positional(self):
        """Newer TRL passes prompts as the first positional arg."""
        funcs, _ = get_reward_funcs_for_task("unified")
        completions = [_make_valid_completion()]
        gts = [json.dumps(GT_SAFE)]
        for fn in funcs:
            try:
                result = fn(["dummy_prompt"], completions, ground_truth=gts)
            except TypeError:
                pytest.fail(f"{fn.__name__} rejects the prompts-positional TRL signature")
            assert isinstance(result, list) and len(result) == 1

    def test_malformed_completion_scores_zero_on_every_component(self):
        funcs, _ = get_reward_funcs_for_task("unified")
        gts = [json.dumps(GT_SAFE)]
        for fn in funcs:
            assert fn(completions=["bad json"], ground_truth=gts) == [0.0]


class TestRepetitionPenalty:
    """Now applied in the LIVE path, per component. Multiplying each component is
    identical to multiplying the total, because TRL sums linearly:
        sum_k w_k * (f * r_k) == f * sum_k w_k * r_k
    """

    @staticmethod
    def _repeated(n=10):
        return _make_valid_completion(excavator=[[100, 100, 500, 500]] * n)

    def test_penalty_fires_on_repeated_boxes(self):
        clean = _make_valid_completion(excavator=[[100, 100, 500, 500]])
        rep = self._repeated()
        assert _apply_repetition_penalty([1.0], [rep], "unified") == [pytest.approx(0.5)]
        assert _apply_repetition_penalty([1.0], [clean], "unified") == [pytest.approx(1.0)]

    def test_threshold_is_more_than_five_identical_boxes(self):
        five = _make_valid_completion(excavator=[[1, 1, 2, 2]] * 5)
        six = _make_valid_completion(excavator=[[1, 1, 2, 2]] * 6)
        assert _apply_repetition_penalty([1.0], [five], "unified") == [pytest.approx(1.0)]
        assert _apply_repetition_penalty([1.0], [six], "unified") == [pytest.approx(0.5)]

    def test_penalty_reaches_the_live_reward_functions(self):
        """The regression this fixes: the penalty existed only in Mode 2, so the
        path GRPO actually uses had no repetition check at all."""
        funcs, _ = get_reward_funcs_for_task("unified")
        by_name = {f.__name__: f for f in funcs}
        gts = [json.dumps(GT_SAFE)]
        clean = by_name["reward_format"](completions=[_make_valid_completion()], ground_truth=gts)[0]
        rep = by_name["reward_format"](completions=[self._repeated()], ground_truth=gts)[0]
        assert clean == pytest.approx(1.0)
        assert rep == pytest.approx(0.5), "repetition penalty did not reach the live path"

    def test_unparseable_completion_is_not_penalised_twice(self):
        assert _apply_repetition_penalty([0.0], ["bad json"], "unified") == [0.0]

    def test_caption_only_can_never_trigger_it(self):
        """caption_only parses to {"caption": ...} with no boxes."""
        assert _apply_repetition_penalty([1.0], ["a" * 50], "caption_only") == [pytest.approx(1.0)]

    def test_factor_of_one_disables_it(self, monkeypatch):
        import rewards.unified_reward as m
        monkeypatch.setattr(m, "reward_constant", lambda task, key, default: 1.0)
        assert _apply_repetition_penalty([1.0], [self._repeated()], "unified") == [1.0]
