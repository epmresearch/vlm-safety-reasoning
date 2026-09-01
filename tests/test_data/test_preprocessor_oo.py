import json

from data.preprocessor import (
    build_target_json,
    build_gt_dict,
    raw_sample_to_conversation_for_task,
)
from data.schemas import get_output_schema

RAW = {
    "image_id": "img_1",
    "image_caption": "A muddy site with an excavator.",
    "illumination": "normal lighting",
    "camera_distance": "short distance",
    "view": "ground view",
    "quality_of_info": "rich",
    "rule_1_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "No hard hat"},
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
    "excavator": [[0.5, 0.5, 0.6, 0.6]],
    "rebar": [],
    "worker_with_white_hard_hat": [[0.1, 0.2, 0.3, 0.4]],
}


def _parse_target(target_str):
    clean = target_str.strip().replace("```json\n", "").replace("\n```", "")
    return json.loads(clean)


def test_build_target_json_oo_has_only_object_keys():
    parsed = _parse_target(build_target_json(RAW, task="object_only"))

    assert set(parsed) == {"excavator", "rebar", "worker_with_white_hard_hat"}
    assert "caption" not in parsed
    assert not any(k.startswith("rule_") for k in parsed)


def test_build_target_json_oo_scales_boxes_to_1000():
    parsed = _parse_target(build_target_json(RAW, task="object_only"))

    assert parsed["excavator"] == [[500, 500, 600, 600]]
    assert parsed["worker_with_white_hard_hat"] == [[100, 200, 300, 400]]
    assert parsed["rebar"] == []


def test_build_target_json_oo_validates_against_its_schema():
    parsed = _parse_target(build_target_json(RAW, task="object_only"))
    get_output_schema("object_only")(**parsed)


def test_build_gt_dict_oo_keeps_boxes_in_0_1_scale():
    gt = build_gt_dict(RAW, task="object_only")

    assert gt["excavator"] == [[0.5, 0.5, 0.6, 0.6]]
    assert gt["worker_with_white_hard_hat"] == [[0.1, 0.2, 0.3, 0.4]]
    assert gt["rebar"] == []


def test_build_gt_dict_oo_omits_caption_and_violations_but_keeps_metadata():
    gt = build_gt_dict(RAW, task="object_only")

    assert "caption" not in gt
    assert not any(k.startswith("rule_") for k in gt)
    # Metadata is retained for stratified analysis.
    assert gt["illumination"] == "normal lighting"
    assert gt["view"] == "ground view"


def test_conversation_oo_prompt_and_target_exclude_other_families():
    conv = raw_sample_to_conversation_for_task(RAW, pil_image="FAKE_IMAGE", task="object_only")
    messages = conv["messages"]

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    user_prompt = messages[1]["content"][1]["text"]
    assert "excavator" in user_prompt
    assert "rule 1" not in user_prompt.lower()

    target_str = messages[2]["content"][0]["text"]
    assert "excavator" in target_str
    assert "caption" not in target_str
    assert "rule_1_violation" not in target_str
    # object_only is a fenced-JSON task.
    assert target_str.startswith("```json")


def test_oo_target_drops_out_of_range_boxes_before_scaling():
    """clean_boxes rejects anything outside [0,1]; it must run BEFORE the x1000
    scaling, otherwise every scaled box would be rejected."""
    raw = dict(RAW, excavator=[[0.1, 0.1, 0.2, 0.2], [1.5, 0.1, 2.0, 0.2]])
    parsed = _parse_target(build_target_json(raw, task="object_only"))
    assert parsed["excavator"] == [[100, 100, 200, 200]]
