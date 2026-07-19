import pytest
from typing import List
from data.preprocessor import build_unified_sft_dataset
from datasets import Dataset
from PIL import Image

def test_build_unified_sft_dataset():
    # Mock dataset matching actual raw data
    raw_data = {
        "image": [Image.new("RGB", (100, 100), color="red")],
        "image_id": ["test_001"],
        "image_caption": ["A construction site"],
        "illumination": ["normal lighting"],
        "camera_distance": ["mid distance"],
        "view": ["ground view"],
        "quality_of_info": ["average"],
        "rule_1_violation": [{"reason": "No hard hat", "bounding_box": [[0.1, 0.2, 0.3, 0.4]]}],
        "rule_2_violation": [None],
        "rule_3_violation": [None],
        "rule_4_violation": [None],
        "excavator": [[[0.5, 0.5, 0.8, 0.8]]],
        "rebar": [[]],
        "worker_with_white_hard_hat": [[]],
    }
    ds = Dataset.from_dict(raw_data)
    
    # Process
    processed = build_unified_sft_dataset(ds)
    
    # Check return type
    assert isinstance(processed, list)
    assert len(processed) == 1
    
    # Check messages structure
    assert "messages" in processed[0]
    messages = processed[0]["messages"]
    assert len(messages) == 3
    
    # Check roles
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    
    # Assistant content contains json fenced JSON
    assistant_content = messages[2]["content"][0]["text"]
    assert "```json" in assistant_content
    
    # Target JSON contains expected keys
    assert '"caption":' in assistant_content
    assert '"rule_1_violation":' in assistant_content
    assert '"excavator":' in assistant_content