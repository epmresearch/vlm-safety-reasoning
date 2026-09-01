"""
Single source of truth for what a *task* is.

A "task" is one full, independent baseline->SFT->merge->GRPO->eval pipeline over the
same images and the same base model. Tasks differ only in what the model is asked to
output. Every other layer of the stack -- naming, configs, prompts, schemas, target
builders, rewards, structural repair, evaluation metric gating, SLURM orchestration --
derives its task-specific behaviour from the TASK_REGISTRY below.

Adding a task means adding ONE TaskSpec here, plus the four data artifacts it points at
(a configs/tasks/<name>.yaml, a PROMPT_REGISTRY entry, a SCHEMA_REGISTRY entry, and a
pair of target/ground-truth builders in data/preprocessor.py). No conditional anywhere
else in the codebase should ever compare a task name to a literal string again --
ask a capability question instead:

    from core.tasks import task_has
    if task_has(task, CAP_CAPTION):
        ...

Why capabilities and not task names: before this registry existed, task-aware code was
written as binary negations (``if task != "violations_only":``). Every such site
silently gave a brand-new task the *unified* behaviour, which is how object_only ended
up scheduled to run captioning metrics over empty strings and caption_only to run
grounding metrics against real boxes. A capability set makes the question "does this
task even produce that field?" answerable without enumerating tasks.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List

# ---------------------------------------------------------------------------
# Capabilities — the output families a task may produce.
#
# These map 1:1 onto the field groups of the output schemas, the reward
# components, and the evaluation metric families:
#
#   CAP_CAPTION    -> "caption"                          -> reward_caption,
#                                                           captioning metrics
#   CAP_OBJECTS    -> excavator/rebar/worker_...          -> reward_grounding,
#                                                           grounding metrics
#   CAP_VIOLATIONS -> rule_1..4_violation                 -> reward_violation_id,
#                                                           reward_violation_grounding,
#                                                           reward_reasoning,
#                                                           violation + reasoning metrics
# ---------------------------------------------------------------------------
CAP_CAPTION = "caption"
CAP_OBJECTS = "objects"
CAP_VIOLATIONS = "violations"

ALL_CAPABILITIES = (CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS)

# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------
# A minimized JSON object inside a ```json ... ``` fence.
FORMAT_FENCED_JSON = "fenced_json"
# Bare prose. No fence, no JSON, no keys. Used by caption_only, where the caption
# *is* the entire output and wrapping it in JSON would only add a formatting
# confound to the thing being measured.
FORMAT_PLAIN_TEXT = "plain_text"


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of one pipeline task."""

    name: str
    prefix: str
    capabilities: FrozenSet[str]
    output_format: str

    def __post_init__(self):
        unknown = set(self.capabilities) - set(ALL_CAPABILITIES)
        if unknown:
            raise ValueError(
                f"TaskSpec {self.name!r} declares unknown capabilities {sorted(unknown)}. "
                f"Known: {list(ALL_CAPABILITIES)}"
            )
        if self.output_format not in (FORMAT_FENCED_JSON, FORMAT_PLAIN_TEXT):
            raise ValueError(
                f"TaskSpec {self.name!r} declares unknown output_format "
                f"{self.output_format!r}."
            )


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "unified": TaskSpec(
        name="unified",
        prefix="unified",
        capabilities=frozenset({CAP_CAPTION, CAP_OBJECTS, CAP_VIOLATIONS}),
        output_format=FORMAT_FENCED_JSON,
    ),
    "violations_only": TaskSpec(
        name="violations_only",
        prefix="vo",
        capabilities=frozenset({CAP_VIOLATIONS}),
        output_format=FORMAT_FENCED_JSON,
    ),
    "object_only": TaskSpec(
        name="object_only",
        prefix="oo",
        capabilities=frozenset({CAP_OBJECTS}),
        output_format=FORMAT_FENCED_JSON,
    ),
    "caption_only": TaskSpec(
        name="caption_only",
        prefix="co",
        capabilities=frozenset({CAP_CAPTION}),
        output_format=FORMAT_PLAIN_TEXT,
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    """Returns the TaskSpec for ``task``.

    Raises:
        ValueError: if the task is not registered. This is deliberately loud —
            an unregistered task used to fall through to unified behaviour
            everywhere, producing plausible-looking but wrong results.
    """
    spec = TASK_REGISTRY.get(task)
    if spec is None:
        raise ValueError(
            f"Unknown task: {task!r}. Registered tasks: {sorted(TASK_REGISTRY)}. "
            "Add a TaskSpec to core/tasks.py::TASK_REGISTRY to register a new one."
        )
    return spec


def validate_task(task: str) -> str:
    """Validates ``task`` and returns it unchanged, for use in argparse/entrypoints."""
    get_task_spec(task)
    return task


def task_has(task: str, capability: str) -> bool:
    """True if ``task``'s output contains the given capability's fields."""
    if capability not in ALL_CAPABILITIES:
        raise ValueError(
            f"Unknown capability: {capability!r}. Known: {list(ALL_CAPABILITIES)}"
        )
    return capability in get_task_spec(task).capabilities


def task_capabilities(task: str) -> FrozenSet[str]:
    """The full capability set for ``task``."""
    return get_task_spec(task).capabilities


def task_output_format(task: str) -> str:
    """Either FORMAT_FENCED_JSON or FORMAT_PLAIN_TEXT."""
    return get_task_spec(task).output_format


def is_plain_text_task(task: str) -> bool:
    """True if the task's completion is bare prose rather than fenced JSON."""
    return task_output_format(task) == FORMAT_PLAIN_TEXT


def tasks_with(capability: str) -> List[str]:
    """All registered task names that produce the given capability's fields."""
    return [n for n, s in TASK_REGISTRY.items() if capability in s.capabilities]


def all_task_names() -> List[str]:
    """All registered task names, in registration order."""
    return list(TASK_REGISTRY)


def all_task_prefixes() -> Dict[str, str]:
    """``{task_name: short_prefix}`` for every registered task."""
    return {n: s.prefix for n, s in TASK_REGISTRY.items()}
