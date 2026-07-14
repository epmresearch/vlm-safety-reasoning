import pytest
from evaluation.output_parser import parse_model_output

def test_parse_valid_json():
    raw_output = '''Some thought process...
```json
{
  "caption": "A construction site",
  "detected_objects": [
    {"class_name": "helmet", "bbox": [0, 0, 10, 10]}
  ],
  "safety_violations": []
}
```
'''
    parsed = parse_model_output(raw_output)
    assert parsed.is_valid_json is True
    assert parsed.caption == "A construction site"
    assert len(parsed.detected_objects) == 1
    assert len(parsed.safety_violations) == 0

def test_parse_invalid_json():
    raw_output = '''```json
{
  "caption": "A construction site",
  "detected_objects": [
'''
    parsed = parse_model_output(raw_output)
    assert parsed.is_valid_json is False
    assert parsed.caption == ""
    assert len(parsed.detected_objects) == 0
