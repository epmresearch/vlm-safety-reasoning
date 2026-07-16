import pytest

from evaluation.output_parser import strip_fences, parse_model_output, validate_unified_output

def test_strip_fences():
    """Test regex extraction of JSON from code fences."""
    # Standard fence
    assert strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    
    # Fence with preamble and postamble (common hallucination)
    assert strip_fences("Here is the output:\n```json\n{\"a\": 1}\n```\nHope this helps!") == '{"a": 1}'
    
    # Missing json identifier
    assert strip_fences("```\n{\"a\": 1}\n```") == '{"a": 1}'
    
    # No fences at all (fallback to strip)
    assert strip_fences("  {\"a\": 1}  ") == '{"a": 1}'
    
    # Empty string
    assert strip_fences("") == ""

def test_parse_model_output():
    """Test parsing of raw strings into dictionaries."""
    # Valid fenced JSON
    res = parse_model_output("```json\n{\"key\": \"value\"}\n```")
    assert isinstance(res, dict)
    assert res["key"] == "value"
    
    # Valid unfenced JSON
    assert parse_model_output("{\"key\": \"value\"}") == {"key": "value"}
    
    # Invalid JSON (missing quotes)
    assert parse_model_output("{key: value}") is None
    
    # Empty string
    assert parse_model_output("") is None

def test_validate_unified_output():
    """Test validation of parsed dictionaries against Pydantic schema."""
    # Valid minimal schema (UnifiedOutput usually allows empty/null for most fields, but caption is required)
    res = validate_unified_output({"caption": "test"})
    assert res is not None # Assuming dict with caption is valid against the schema structure
    
    # Valid full schema
    full = {
        "caption": "A construction site",
        "rule_1_violation": {"reason": "Missing hard hat", "bounding_box": [[0,0,1,1]]},
        "excavator": [[0,0,1,1]]
    }
    res = validate_unified_output(full)
    assert res is not None
    
    # Passed None
    assert validate_unified_output(None) is None
