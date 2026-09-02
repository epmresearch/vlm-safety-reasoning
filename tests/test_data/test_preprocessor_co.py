import json

import pytest

from data.preprocessor import (
    build_target_json,
    build_gt_dict,
    raw_sample_to_conversation_for_task,
)
from data.schemas import get_output_schema

RAW = {
    "image_id": "img_1",
    "image_caption": "  A muddy site with an excavator and two workers.  ",
    "illumination": "night",
    "camera_distance": "long distance",
    "view": "bird's-eye view",
    "quality_of_info": "average",
    "rule_1_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "No hard hat"},
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
    "excavator": [[0.5, 0.5, 0.6, 0.6]],
    "rebar": [],
    "worker_with_white_hard_hat": [[0.1, 0.2, 0.3, 0.4]],
}


def test_build_target_co_is_bare_prose():
    """caption_only's wire format is plain text — no fence, no JSON. Emitting a
    fence here would train exactly what reward_format penalizes."""
    target = build_target_json(RAW, task="caption_only")

    assert target == "A muddy site with an excavator and two workers."
    assert "```" not in target
    assert not target.lstrip().startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(target)


def test_build_target_co_validates_against_its_schema():
    target = build_target_json(RAW, task="caption_only")
    get_output_schema("caption_only")(caption=target)


def test_build_gt_dict_co_carries_caption_and_metadata_only():
    gt = build_gt_dict(RAW, task="caption_only")

    # reward_caption and the captioning metrics both read gt["caption"], so it is
    # mandatory for this task.
    assert gt["caption"] == RAW["image_caption"]
    assert not any(k.startswith("rule_") for k in gt)
    assert "excavator" not in gt
    assert gt["illumination"] == "night"
    assert gt["quality_of_info"] == "average"


def test_conversation_co_prompt_forbids_json_and_target_is_prose():
    conv = raw_sample_to_conversation_for_task(RAW, pil_image="FAKE_IMAGE", task="caption_only")
    messages = conv["messages"]

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    # JSON, code fences and bounding boxes are all mentioned only in order to be
    # forbidden — this task's completion is bare prose.
    user_prompt = messages[1]["content"][1]["text"].lower()
    # Assert the CONTRACT, not the exact wording: caption_only's reward_format is
    # output_parser.is_clean_prose, which rejects a fence, a JSON object and a
    # leading 'caption:' label. The prompt must forbid those three and is free to
    # phrase it however reads best. Pinning exact sentences made this test fail on a
    # pure rewording that left the contract intact.
    assert "json" in user_prompt
    assert "fence" in user_prompt
    assert "label" in user_prompt or "heading" in user_prompt
    # ...and it must not ask for the things that would break the contract.
    assert "```" not in user_prompt

    target_str = messages[2]["content"][0]["text"]
    assert "```" not in target_str
    assert target_str == "A muddy site with an excavator and two workers."


def test_build_target_co_on_missing_caption_is_empty_not_crash():
    """A raw row with no caption yields an empty target rather than raising —
    the empty string is then rejected downstream by CaptionOnlyOutput, which is
    where that failure belongs."""
    assert build_target_json({"image_id": "x"}, task="caption_only") == ""
    with pytest.raises(Exception):
        get_output_schema("caption_only")(caption="")
