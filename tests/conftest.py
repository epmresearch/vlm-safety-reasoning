"""
Shared pytest fixtures for the VLM Safety Reasoning test suite.
Run from project root: pytest tests/ -v
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Valid completion fixtures
# ---------------------------------------------------------------------------

def _make_completion(**overrides) -> str:
    """Build a valid ```json fenced completion matching UnifiedOutput schema."""
    base = {
        "caption": "A construction site with an excavator and workers wearing hard hats.",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [[100, 50, 800, 700]],
        "rebar": [],
        "worker_with_white_hard_hat": [[850, 200, 950, 600]],
    }
    base.update(overrides)
    return "```json\n" + json.dumps(base) + "\n```"


@pytest.fixture
def valid_completion_no_violation():
    return _make_completion()


@pytest.fixture
def valid_completion_with_rule1():
    return _make_completion(
        rule_1_violation={
            "bounding_box": [[850, 200, 950, 600]],
            "reason": "Worker not wearing a hard hat.",
        }
    )


@pytest.fixture
def valid_completion_with_rule2():
    return _make_completion(
        rule_2_violation={
            "bounding_box": [[300, 100, 600, 400]],
            "reason": "Worker at height without harness.",
        }
    )


@pytest.fixture
def valid_completion_multi_violation():
    return _make_completion(
        rule_1_violation={
            "bounding_box": [[850, 200, 950, 600]],
            "reason": "No PPE observed.",
        },
        rule_4_violation={
            "bounding_box": [[200, 300, 400, 500]],
            "reason": "Worker in excavator blind spot.",
        },
    )


@pytest.fixture
def invalid_json_completion():
    return "```json\n{this is not valid json}\n```"


@pytest.fixture
def valid_json_invalid_schema_completion():
    # Missing required 'caption' field
    bad = {"rule_1_violation": None, "excavator": []}
    return "```json\n" + json.dumps(bad) + "\n```"


@pytest.fixture
def empty_completion():
    return ""


@pytest.fixture
def unfenced_completion():
    """Valid JSON but without the ``` fences."""
    return json.dumps({
        "caption": "A site.",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    })


# ---------------------------------------------------------------------------
# Ground truth fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gt_no_violation():
    return {
        "caption": "An excavator on a construction site.",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [[0.1, 0.05, 0.8, 0.7]],    # [0,1] scale
        "rebar": [],
        "worker_with_white_hard_hat": [[0.85, 0.2, 0.95, 0.6]],
    }


@pytest.fixture
def gt_rule1_violation():
    return {
        "caption": "Workers without hard hats.",
        "rule_1_violation": {
            "bounding_box": [[0.85, 0.2, 0.95, 0.6]],
            "reason": "Worker not wearing a hard hat.",
        },
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [[0.1, 0.05, 0.8, 0.7]],
        "rebar": [],
        "worker_with_white_hard_hat": [[0.85, 0.2, 0.95, 0.6]],
    }


@pytest.fixture
def gt_multi_violation():
    return {
        "caption": "Multiple violations observed.",
        "rule_1_violation": {
            "bounding_box": [[0.85, 0.2, 0.95, 0.6]],
            "reason": "No PPE.",
        },
        "rule_4_violation": {
            "bounding_box": [[0.2, 0.3, 0.4, 0.5]],
            "reason": "Worker in blind spot.",
        },
        "rule_2_violation": None,
        "rule_3_violation": None,
        "excavator": [[0.1, 0.05, 0.8, 0.7]],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }


@pytest.fixture
def gt_no_objects():
    """Ground truth where no objects of interest are present."""
    return {
        "caption": "Empty site.",
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }

# ---------------------------------------------------------------------------
# Violations-only fixtures
# ---------------------------------------------------------------------------

def _make_violations_only_completion(**overrides) -> str:
    base = {
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
    }
    base.update(overrides)
    return "```json\n" + json.dumps(base) + "\n```"

@pytest.fixture
def vo_valid_completion_no_violation():
    return _make_violations_only_completion()

@pytest.fixture
def vo_valid_completion_with_rule1():
    return _make_violations_only_completion(
        rule_1_violation={
            "bounding_box": [[850, 200, 950, 600]],
            "reason": "Worker not wearing a hard hat.",
        }
    )

@pytest.fixture
def vo_gt_no_violation():
    return {
        "rule_1_violation": None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
    }

@pytest.fixture
def vo_gt_rule1_violation():
    return {
        "rule_1_violation": {
            "bounding_box": [[0.85, 0.2, 0.95, 0.6]],
            "reason": "Worker not wearing a hard hat.",
        },
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
    }

# ---------------------------------------------------------------------------
# Object-only fixtures
# ---------------------------------------------------------------------------

def _make_object_only_completion(**overrides) -> str:
    """A well-formed object_only completion: fenced minimized JSON, 3 class keys.

    All three keys are always present because ObjectOnlyOutput requires them —
    see the rationale in data/schemas.py.
    """
    base = {
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }
    base.update(overrides)
    return "```json\n" + json.dumps(base) + "\n```"


@pytest.fixture
def oo_valid_completion_no_objects():
    return _make_object_only_completion()


@pytest.fixture
def oo_valid_completion_with_excavator():
    # Predictions are on the [0,1000] Qwen scale.
    return _make_object_only_completion(excavator=[[100, 200, 300, 400]])


@pytest.fixture
def oo_gt_no_objects():
    # Ground truth stays on the dataset [0,1] scale.
    return {
        "excavator": [],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }


@pytest.fixture
def oo_gt_with_excavator():
    return {
        "excavator": [[0.1, 0.2, 0.3, 0.4]],
        "rebar": [],
        "worker_with_white_hard_hat": [],
    }


# ---------------------------------------------------------------------------
# Caption-only fixtures
#
# caption_only's wire format is BARE PROSE — no fence, no JSON. See
# core/tasks.py::FORMAT_PLAIN_TEXT.
# ---------------------------------------------------------------------------

def _make_caption_only_completion(caption: str = None) -> str:
    if caption is None:
        caption = ("Two workers in white hard hats stand beside a yellow excavator "
                   "on a muddy site; bundled rebar is stacked at the left edge.")
    return caption


@pytest.fixture
def co_valid_completion():
    return _make_caption_only_completion()


@pytest.fixture
def co_gt():
    return {
        "caption": ("Two workers wearing white hard hats next to an excavator, "
                    "with rebar stacked nearby."),
    }
