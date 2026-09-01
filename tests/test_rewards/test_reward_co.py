import json

import pytest

from rewards.reward_format import compute_reward as reward_format
from rewards.reward_caption import compute_reward as reward_caption_batch
from rewards.unified_reward import get_reward_funcs_for_task
from tests.conftest import _make_caption_only_completion


PROSE = ("Two workers in white hard hats stand beside a yellow excavator on a "
         "muddy site; bundled rebar is stacked at the left edge.")


# ---------------------------------------------------------------------------
# Reward assembly
# ---------------------------------------------------------------------------

def test_get_reward_funcs_for_task_co():
    funcs, weights = get_reward_funcs_for_task(task="caption_only")

    names = [f.__name__ for f in funcs]
    assert names == ["reward_format", "reward_caption"]
    assert weights == pytest.approx([0.10, 0.90])
    assert sum(weights) == pytest.approx(1.0)

    for absent in ("reward_grounding", "reward_violation_id",
                   "reward_violation_grounding", "reward_reasoning"):
        assert absent not in names


# ---------------------------------------------------------------------------
# Format reward — rewards CLEAN PROSE for this task, not valid JSON
# ---------------------------------------------------------------------------

def test_format_reward_co_accepts_bare_prose():
    assert reward_format(PROSE, {}, task="caption_only") == 1.0


def test_format_reward_co_rejects_fenced_json():
    """A caption_only model that reverts to the JSON habit the other three tasks
    train must not be scored as perfectly formatted."""
    completion = '```json\n{"caption": "%s"}\n```' % PROSE
    assert reward_format(completion, {}, task="caption_only") == 0.0


def test_format_reward_co_rejects_bare_json_object():
    assert reward_format('{"caption": "%s"}' % PROSE, {}, task="caption_only") == 0.0


def test_format_reward_co_rejects_fenced_prose():
    assert reward_format("```\n%s\n```" % PROSE, {}, task="caption_only") == 0.0


def test_format_reward_co_rejects_caption_label():
    assert reward_format('"caption": %s' % PROSE, {}, task="caption_only") == 0.0


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_format_reward_co_rejects_blank(empty):
    """CaptionOnlyOutput rejects a blank caption, so an empty completion cannot
    collect the format reward — without that, the reward would be free."""
    assert reward_format(empty, {}, task="caption_only") == 0.0


def test_format_reward_co_allows_prose_containing_a_brace_mid_sentence():
    prose = "A site with a brace { in the middle of the sentence."
    assert reward_format(prose, {}, task="caption_only") == 1.0


# ---------------------------------------------------------------------------
# Caption reward
# ---------------------------------------------------------------------------

def test_caption_reward_co_scores_prose_against_reference(co_gt):
    """reward_caption is batched and reads pred/gt 'caption'; for caption_only the
    prediction arrives as bare prose and must still be scored."""
    scores = reward_caption_batch([PROSE], [co_gt], task="caption_only")
    assert len(scores) == 1
    assert 0.0 < scores[0] <= 1.0


def test_caption_reward_co_identical_caption_scores_higher_than_unrelated(co_gt):
    identical = co_gt["caption"]
    unrelated = "A close-up photograph of a cat sitting on a windowsill indoors."
    scores = reward_caption_batch([identical, unrelated], [co_gt, co_gt], task="caption_only")
    assert scores[0] > scores[1]


def test_caption_reward_co_blank_prediction_scores_zero(co_gt):
    scores = reward_caption_batch(["   "], [co_gt], task="caption_only")
    assert scores[0] == 0.0


def test_caption_reward_co_missing_gt_caption_scores_zero():
    """Guards the ground-truth builder: build_caption_only_ground_truth MUST emit
    a caption, or every sample silently scores zero."""
    scores = reward_caption_batch([PROSE], [{}], task="caption_only")
    assert scores[0] == 0.0


def test_co_reward_funcs_run_batched_end_to_end(co_gt):
    funcs, _ = get_reward_funcs_for_task(task="caption_only")
    completions = [_make_caption_only_completion(), ""]
    ground_truth = [json.dumps(co_gt)] * 2

    for fn in funcs:
        scores = fn(completions=completions, ground_truth=ground_truth)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert scores[1] == 0.0
