import pytest
from evaluation.output_parser import parse_model_output

def test_parse_valid_json():
    raw_output = '''Some thought process...
```json
{
  "caption": "A construction site",
  "excavator": [[0, 0, 10, 10]],
  "rule_1_violation": {"bounding_box": [[0, 0, 10, 10]], "reason": "reason"}
}
```
'''
    parsed = parse_model_output(raw_output)
    assert parsed is not None
    assert parsed.get("caption") == "A construction site"
    assert len(parsed.get("excavator", [])) == 1
    assert parsed.get("rule_1_violation") is not None

def test_parse_invalid_json():
    raw_output = '''```json
{
  "caption": "A construction site",
  "excavator": [
'''
    parsed = parse_model_output(raw_output)
    assert parsed is None
