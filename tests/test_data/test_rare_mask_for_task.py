"""Task-aware rare masking for the SFT stratified sampler.

The mask decides which rows StratifiedRareClassSampler spreads evenly across an epoch.
It used to be violation-only, which silently gave object_only a plain shuffle.
"""
import pytest

from data.oversampling import build_rare_mask_for_task
from data.samplers import StratifiedRareClassSampler


def _row(rule2=False, rule3=False, rule4=False, excavator=False, rebar=False, hat=False):
    v = {"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": "r"}
    return {
        "rule_1_violation": None,
        "rule_2_violation": v if rule2 else None,
        "rule_3_violation": v if rule3 else None,
        "rule_4_violation": v if rule4 else None,
        "excavator": [[0.1, 0.1, 0.5, 0.5]] if excavator else [],
        "rebar": [[0.2, 0.2, 0.6, 0.6]] if rebar else [],
        "worker_with_white_hard_hat": [[0.3, 0.3, 0.4, 0.4]] if hat else [],
    }


@pytest.mark.parametrize("task", ["unified", "violations_only"])
def test_violation_tasks_mask_on_rules_234_only(task):
    """Unchanged from the legacy build_rare_mask: rule 1 is the common class and is not
    rare, and objects must not influence the mask for a violation task."""
    rows = [_row(), _row(rule2=True), _row(rule3=True), _row(rule4=True),
            _row(excavator=True, rebar=True, hat=True)]
    assert build_rare_mask_for_task(rows, task) == [False, True, True, True, False]


def test_object_only_masks_on_the_two_hard_classes():
    """rebar and white-hard-hat workers are the rare classes; excavator is not.

    Excavator is excluded deliberately: at 2415 train occurrences against 846 and 680 it
    is the common, visually salient class, and marking it rare would mark most images
    rare, degenerating the stratification into a plain shuffle.
    """
    rows = [_row(), _row(excavator=True), _row(rebar=True), _row(hat=True),
            _row(excavator=True, rebar=True)]
    assert build_rare_mask_for_task(rows, "object_only") == [False, False, True, True, True]


def test_object_only_ignores_violation_fields():
    """object_only never predicts violations, so a rule must not make a row rare."""
    assert build_rare_mask_for_task([_row(rule4=True)], "object_only") == [False]


def test_caption_only_has_no_rare_axis():
    """None means 'fall back to a plain shuffle' — there is no rare caption."""
    assert build_rare_mask_for_task([_row(), _row(rebar=True)], "caption_only") is None


def test_unified_prefers_violations_over_objects():
    """unified has both capabilities. Violations win: they carry 0.55 of its reward
    weight against 0.25 for grounding, and stratifying two axes at once would
    over-constrain the ordering with no clear winner."""
    rows = [_row(rebar=True), _row(rule2=True)]
    assert build_rare_mask_for_task(rows, "unified") == [False, True]


def test_sampler_reduces_batch_composition_variance():
    """The point of the mask: same rows, same count per epoch, lower variance in how
    many rare rows land in each batch."""
    import statistics as st

    n, batch = 640, 32
    mask = [(i % 10 == 0) for i in range(n)]          # 10 % rare
    order = list(StratifiedRareClassSampler(mask, seed=42))

    assert sorted(order) == list(range(n)), "every index must appear exactly once"

    counts = [sum(mask[i] for i in order[b:b + batch]) for b in range(0, n, batch)]
    assert st.pstdev(counts) < 1.0, f"stratified spread too uneven: {counts}"
    assert sum(counts) == sum(mask)


def test_sampler_reorders_between_epochs():
    """models/sft_trainer.py forwards HF's per-epoch set_epoch onto the sampler. Without
    that the same order replays every epoch, which is invisible at 1 epoch and silently
    wastes the second."""
    mask = [(i % 10 == 0) for i in range(320)]
    s = StratifiedRareClassSampler(mask, seed=42)
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    second = list(s)
    assert first != second
    assert sorted(first) == sorted(second)
