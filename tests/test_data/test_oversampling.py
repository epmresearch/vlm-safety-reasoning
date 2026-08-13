import pytest
from data.oversampling import build_oversampled_indices, build_rare_mask

def _make_sample(*rules):
    # Pass a list of rule numbers, e.g., (1, 2)
    sample = {}
    for r in rules:
        sample[f"rule_{r}_violation"] = {"bounding_box": [[0,0,1,1]], "reason": "test"}
    return sample

def test_build_oversampled_indices():
    dataset = [
        _make_sample(1),            # Normal -> 1 copy (index 0)
        _make_sample(2),            # Rule 2 -> 4 copies (index 1)
        _make_sample(4),            # Rule 4 -> 4 copies (index 2)
        _make_sample(3),            # Rule 3 -> 2 copies (index 3)
        _make_sample(3, 4),         # Rule 3+4 -> 4 copies (index 4)
        _make_sample(),             # Safe -> 1 copy (index 5)
    ]
    
    indices, manifest = build_oversampled_indices(dataset)
    
    # Expected indices count: 1 + 4 + 4 + 2 + 4 + 1 = 16
    assert len(indices) == 16
    
    # Check exact occurrences
    assert indices.count(0) == 1
    assert indices.count(1) == 4
    assert indices.count(2) == 4
    assert indices.count(3) == 2
    assert indices.count(4) == 4
    assert indices.count(5) == 1
    
    # Check manifest
    assert manifest["total_rows_before"] == 6
    assert manifest["total_rows_after"] == 16
    assert manifest["rule24_unique_images"] == 3  # index 1, 2, 4
    assert manifest["rule3_only_unique_images"] == 1 # index 3
    assert manifest["net_added_rows"] == 10

def test_build_rare_mask():
    dataset = [
        _make_sample(1),            # False
        _make_sample(2),            # True
        _make_sample(3),            # True
        _make_sample(4),            # True
        _make_sample(1, 3),         # True
        _make_sample(),             # False
    ]
    
    mask = build_rare_mask(dataset)
    assert mask == [False, True, True, True, True, False]
