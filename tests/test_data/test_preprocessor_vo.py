import json
import pytest
from data.preprocessor import (
    build_target_json,
    build_gt_dict,
    raw_sample_to_conversation_for_task,
)
from data.schemas import get_output_schema

def test_build_target_json_vo():
    raw = {
        "rule_1_violation": {
            "bounding_box": [[0.5, 0.5, 0.6, 0.6]],
            "reason": "No hard hat"
        }
    }
    # It scales 0-1 to 0-1000
    target_str = build_target_json(raw, task="violations_only")
    
    # Strip fences
    clean_str = target_str.strip().replace("```json\n", "").replace("\n```", "")
    parsed = json.loads(clean_str)
    
    assert "caption" not in parsed
    assert "excavator" not in parsed
    
    assert parsed["rule_1_violation"]["reason"] == "No hard hat"
    assert parsed["rule_1_violation"]["bounding_box"] == [[500, 500, 600, 600]]
    assert parsed["rule_2_violation"] is None
    assert parsed["rule_3_violation"] is None
    assert parsed["rule_4_violation"] is None
    
    # Validate against schema
    schema = get_output_schema("violations_only")
    schema(**parsed)

def test_build_gt_dict_vo():
    raw = {
        "rule_1_violation": {
            "bounding_box": [[0.5, 0.5, 0.6, 0.6]],
            "reason": "No hard hat"
        },
        "illumination": "good"
    }
    
    gt = build_gt_dict(raw, task="violations_only")
    assert "caption" not in gt
    assert "excavator" not in gt
    assert gt["illumination"] == "good"
    assert gt["rule_1_violation"]["reason"] == "No hard hat"
    # Ground truth remains in 0-1 scale
    assert gt["rule_1_violation"]["bounding_box"] == [[0.5, 0.5, 0.6, 0.6]]
    assert gt["rule_2_violation"] is None

def test_raw_sample_to_conversation_vo():
    raw = {
        "rule_1_violation": {
            "bounding_box": [[0.5, 0.5, 0.6, 0.6]],
            "reason": "No hard hat"
        }
    }
    conv = raw_sample_to_conversation_for_task(raw, pil_image="FAKE_IMAGE", task="violations_only")
    messages = conv["messages"]
    
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    
    user_prompt = messages[1]["content"][1]["text"]
    assert "caption" not in user_prompt.lower()
    
    
    target_str = messages[2]["content"][0]["text"]
    assert "rule_1_violation" in target_str
    assert "caption" not in target_str
