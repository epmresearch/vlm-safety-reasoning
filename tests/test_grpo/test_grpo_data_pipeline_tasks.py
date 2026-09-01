"""The SFT and GRPO dataset builders, exercised for every registered task.

The invariants pinned here are the ones that silently produce a blind or
mis-trained model rather than an error:

  - the GRPO dataset's image column must be named `image` (singular); TRL's
    rollout code looks for exactly that key (CLAUDE.md invariant #1),
  - the prompt must carry a bare `{"type": "image"}` placeholder with no PIL
    payload inlined into the message dict,
  - each task's ground truth must round-trip through JSON (it is serialized into
    the dataset and parsed back by the reward wrappers),
  - each task's SFT target must be exactly its own wire format, carrying only its
    own fields.
"""
import json

import pytest
from PIL import Image as PILImage

from core.tasks import (
    CAP_CAPTION,
    CAP_OBJECTS,
    CAP_VIOLATIONS,
    TASK_REGISTRY,
    is_plain_text_task,
    task_has,
)
from data.preprocessor import (
    build_grpo_dataset_for_task,
    build_sft_dataset,
    raw_sample_to_conversation_for_task,
    to_grpo_prompt_for_task,
)
from data.schemas import get_output_schema
from evaluation.output_parser import parse_output_for_task

TASKS = list(TASK_REGISTRY)


def _raw(image_id="0001234"):
    return {
        "image_id": image_id,
        "image_caption": "A construction site with an excavator.",
        "illumination": "normal lighting",
        "camera_distance": "mid distance",
        "view": "elevation view",
        "quality_of_info": "rich",
        "rule_1_violation": {
            "bounding_box": [[0.1, 0.1, 0.5, 0.5]],
            "reason": "Worker not wearing PPE.",
        },
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [[0.1, 0.05, 0.8, 0.7]],
        "rebar": [],
        "worker_with_white_hard_hat": [[0.85, 0.2, 0.95, 0.6]],
    }


def _img():
    return PILImage.new("RGB", (64, 64), color=(128, 128, 128))


class _FakeSplit:
    """Minimal stand-in for an HF Dataset slice."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, idx):
        return _FakeSplit([self._rows[i] for i in idx])


@pytest.fixture
def split():
    return _FakeSplit([{**_raw(f"img_{i}"), "image": _img()} for i in range(3)])


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_sft_target_matches_the_task_wire_format(task, split):
    convs = build_sft_dataset(split, task=task)
    assert len(convs) == 3

    target = convs[0]["messages"][2]["content"][0]["text"]
    if is_plain_text_task(task):
        assert "```" not in target
        assert not target.lstrip().startswith("{")
    else:
        assert target.startswith("```json")
        assert target.rstrip().endswith("```")


@pytest.mark.parametrize("task", TASKS)
def test_sft_target_validates_against_the_task_schema(task, split):
    convs = build_sft_dataset(split, task=task)
    target = convs[0]["messages"][2]["content"][0]["text"]

    parsed = parse_output_for_task(target, task=task)
    assert parsed is not None, f"{task}: its own SFT target does not parse"
    assert get_output_schema(task)(**parsed) is not None


@pytest.mark.parametrize("task", TASKS)
def test_sft_target_contains_only_the_task_own_fields(task, split):
    """A task must never be trained to emit a field it is not evaluated on."""
    convs = build_sft_dataset(split, task=task)
    parsed = parse_output_for_task(
        convs[0]["messages"][2]["content"][0]["text"], task=task
    )
    assert set(parsed) == set(get_output_schema(task).model_fields)


@pytest.mark.parametrize("task", TASKS)
def test_sft_conversation_inlines_the_pil_image(task):
    conv = raw_sample_to_conversation_for_task(_raw(), _img(), task=task)
    user_content = conv["messages"][1]["content"]
    image_items = [c for c in user_content if c["type"] == "image"]
    assert len(image_items) == 1
    # SFT embeds the PIL object directly (unlike the GRPO path).
    assert image_items[0]["image"] is not None


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_grpo_prompt_carries_a_bare_image_placeholder(task):
    prompt_dict = to_grpo_prompt_for_task(_raw(), _img(), task=task)
    user_content = prompt_dict["prompt"][1]["content"]
    image_items = [c for c in user_content if c["type"] == "image"]
    assert len(image_items) == 1
    # CLAUDE.md invariant #1: no PIL payload inlined here — the image travels in
    # its own column. Inlining it re-opens the model-is-blind bug.
    assert "image" not in image_items[0]
    assert prompt_dict["image"] is not None


@pytest.mark.parametrize("task", TASKS)
def test_grpo_ground_truth_round_trips_through_json(task):
    prompt_dict = to_grpo_prompt_for_task(_raw(), _img(), task=task)
    gt = json.loads(prompt_dict["ground_truth"])

    if task_has(task, CAP_CAPTION):
        assert gt["caption"]
    else:
        assert "caption" not in gt

    if task_has(task, CAP_OBJECTS):
        assert gt["excavator"] == [[0.1, 0.05, 0.8, 0.7]]
    else:
        assert "excavator" not in gt

    if task_has(task, CAP_VIOLATIONS):
        assert gt["rule_1_violation"]["reason"] == "Worker not wearing PPE."
    else:
        assert "rule_1_violation" not in gt


@pytest.mark.parametrize("task", TASKS)
def test_grpo_dataset_image_column_is_named_image(task, split):
    ds = build_grpo_dataset_for_task(split, task=task)
    assert "image" in ds.column_names, (
        "TRL's rollout code looks for exactly the column name 'image'"
    )
    assert "images" not in ds.column_names
    assert set(ds.column_names) >= {"prompt", "ground_truth", "image_id", "image"}
    assert len(ds) == 3


@pytest.mark.parametrize("task", TASKS)
def test_grpo_dataset_respects_max_samples(task, split):
    ds = build_grpo_dataset_for_task(split, task=task, max_samples=2)
    assert len(ds) == 2


def test_unregistered_task_raises_in_both_builders():
    with pytest.raises(ValueError):
        raw_sample_to_conversation_for_task(_raw(), _img(), task="no_such_task")
    with pytest.raises(ValueError):
        to_grpo_prompt_for_task(_raw(), _img(), task="no_such_task")
