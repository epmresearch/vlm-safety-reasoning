import json

with open("tests/conftest.py", "r", encoding="utf-8") as f:
    content = f.read()

vo_fixtures = """
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
    return "```json\\n" + json.dumps(base) + "\\n```"

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
"""

if "Violations-only fixtures" not in content:
    with open("tests/conftest.py", "a", encoding="utf-8") as f:
        f.write("\n" + vo_fixtures)
    print("Added VO fixtures to conftest.py")
else:
    print("VO fixtures already present")
