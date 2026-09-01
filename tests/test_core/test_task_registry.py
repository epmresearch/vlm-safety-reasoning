"""Invariants of the task registry itself.

core/tasks.py is the one place a task is registered; every other layer derives from
it. These tests assert the derivation actually holds, so a half-added task fails
here instead of hours into a SLURM job.
"""
import pytest

from core.config import load_task_config
from core.constants import VALID_TASKS
from core.naming import TASK_PREFIXES, task_prefix
from core.tasks import (
    ALL_CAPABILITIES,
    CAP_CAPTION,
    CAP_OBJECTS,
    CAP_VIOLATIONS,
    FORMAT_FENCED_JSON,
    FORMAT_PLAIN_TEXT,
    TASK_REGISTRY,
    get_task_spec,
    is_plain_text_task,
    task_capabilities,
    task_has,
    task_output_format,
    tasks_with,
    validate_task,
)
from data.prompt_templates import PROMPT_REGISTRY, get_prompt_for_task
from data.schemas import SCHEMA_REGISTRY, get_output_schema


def test_all_four_pipelines_registered():
    assert set(TASK_REGISTRY) == {
        "unified",
        "violations_only",
        "object_only",
        "caption_only",
    }


def test_valid_tasks_derives_from_registry():
    assert list(VALID_TASKS) == list(TASK_REGISTRY)


def test_task_prefixes_derive_from_registry():
    assert TASK_PREFIXES == {n: s.prefix for n, s in TASK_REGISTRY.items()}


def test_prefixes_are_unique():
    prefixes = [s.prefix for s in TASK_REGISTRY.values()]
    assert len(prefixes) == len(set(prefixes)), f"duplicate task prefix in {prefixes}"


def test_expected_prefixes():
    assert task_prefix("unified") == "unified"
    assert task_prefix("violations_only") == "vo"
    assert task_prefix("object_only") == "oo"
    assert task_prefix("caption_only") == "co"


@pytest.mark.parametrize("task", list(TASK_REGISTRY))
def test_every_task_has_a_yaml_with_matching_name(task):
    cfg = load_task_config(task)
    assert cfg["task_name"] == task


@pytest.mark.parametrize("task", list(TASK_REGISTRY))
def test_every_task_has_a_registered_prompt(task):
    cfg = load_task_config(task)
    assert cfg["prompt_key"] in PROMPT_REGISTRY
    prompt = get_prompt_for_task(task)
    assert isinstance(prompt, str) and prompt.strip()


@pytest.mark.parametrize("task", list(TASK_REGISTRY))
def test_every_task_has_a_registered_schema(task):
    assert task in SCHEMA_REGISTRY
    assert get_output_schema(task) is SCHEMA_REGISTRY[task]


@pytest.mark.parametrize("task", list(TASK_REGISTRY))
def test_every_task_declares_at_least_one_capability(task):
    caps = task_capabilities(task)
    assert caps, f"{task} declares no capabilities"
    assert caps <= set(ALL_CAPABILITIES)


def test_capability_assignments():
    assert task_capabilities("unified") == {CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS}
    assert task_capabilities("violations_only") == {CAP_VIOLATIONS}
    assert task_capabilities("object_only") == {CAP_OBJECTS}
    assert task_capabilities("caption_only") == {CAP_CAPTION}


def test_tasks_with_capability():
    assert set(tasks_with(CAP_CAPTION)) == {"unified", "caption_only"}
    assert set(tasks_with(CAP_OBJECTS)) == {"unified", "object_only"}
    assert set(tasks_with(CAP_VIOLATIONS)) == {"unified", "violations_only"}


def test_output_formats():
    assert task_output_format("unified") == FORMAT_FENCED_JSON
    assert task_output_format("violations_only") == FORMAT_FENCED_JSON
    assert task_output_format("object_only") == FORMAT_FENCED_JSON
    assert task_output_format("caption_only") == FORMAT_PLAIN_TEXT
    assert is_plain_text_task("caption_only")
    assert not is_plain_text_task("unified")


def test_schema_fields_match_declared_capabilities():
    """The schema a task validates against must own exactly the fields its
    capabilities promise. This is what makes the capability gates in the
    evaluator and in structural repair trustworthy."""
    caption_fields = {"caption"}
    object_fields = {"excavator", "rebar", "worker_with_white_hard_hat"}
    violation_fields = {f"rule_{i}_violation" for i in range(1, 5)}

    for task in TASK_REGISTRY:
        fields = set(get_output_schema(task).model_fields)
        assert bool(fields & caption_fields) == task_has(task, CAP_CAPTION), task
        assert bool(fields & object_fields) == task_has(task, CAP_OBJECTS), task
        assert bool(fields & violation_fields) == task_has(task, CAP_VIOLATIONS), task


def test_unknown_task_raises_everywhere():
    with pytest.raises(ValueError):
        get_task_spec("no_such_task")
    with pytest.raises(ValueError):
        validate_task("no_such_task")
    with pytest.raises(ValueError):
        task_prefix("no_such_task")
    with pytest.raises(ValueError):
        get_output_schema("no_such_task")


def test_unknown_capability_raises():
    with pytest.raises(ValueError):
        task_has("unified", "not_a_capability")


def test_validate_task_returns_the_task():
    assert validate_task("object_only") == "object_only"
