from evaluation.metrics import compute_json_validity, compute_rule_violation_score, compute_grounding_score


def test_compute_json_validity():
    assert compute_json_validity('{"a": 1}') == 1.0
    assert compute_json_validity("not json") == 0.0


def test_rule_violation_multilabel_exact_match():
    pred = {"violations": [{"rule_id": "rule_1", "reasoning": "x", "bounding_box": None}]}
    gt = {"violations": [{"rule_id": "rule_1", "reasoning": "y", "bounding_box": None}]}
    assert compute_rule_violation_score(pred, gt) == 1.0


def test_rule_violation_multilabel_partial_match():
    pred = {"violations": [{"rule_id": "rule_1", "reasoning": "x", "bounding_box": None}]}
    gt = {"violations": [
        {"rule_id": "rule_1", "reasoning": "y", "bounding_box": None},
        {"rule_id": "rule_3", "reasoning": "z", "bounding_box": None},
    ]}
    score = compute_rule_violation_score(pred, gt)
    assert 0.0 < score < 1.0


def test_grounding_score_perfect_overlap():
    pred = {"class_name": "excavator", "bounding_boxes": [[0.1, 0.1, 0.5, 0.5]]}
    gt = {"class_name": "excavator", "bounding_boxes": [[0.1, 0.1, 0.5, 0.5]]}
    assert compute_grounding_score(pred, gt) == 1.0


def test_grounding_score_no_objects_correctly_predicted_empty():
    pred = {"class_name": "rebar", "bounding_boxes": []}
    gt = {"class_name": "rebar", "bounding_boxes": []}
    assert compute_grounding_score(pred, gt) == 1.0