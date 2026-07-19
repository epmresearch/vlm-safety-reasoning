import pytest
from data.box_utils import (
    normalize_boxes,
    clean_boxes,
    is_valid_box,
    compute_iou,
    greedy_multibox_iou,
    scale_01_to_1000,
    scale_1000_to_01
)

def test_normalize_boxes():
    assert normalize_boxes(None) == []
    assert normalize_boxes([]) == []
    
    # Flat single box
    assert normalize_boxes([10, 20, 30, 40]) == [[10, 20, 30, 40]]
    
    # Already normalized
    assert normalize_boxes([[10, 20, 30, 40]]) == [[10, 20, 30, 40]]
    
    # Dict wrapper hallucination
    assert normalize_boxes([{"bounding_box": [10, 20, 30, 40]}]) == [[10, 20, 30, 40]]
    assert normalize_boxes([{"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40}]) == [[10.0, 20.0, 30.0, 40.0]]

def test_clean_boxes():
    # Valid boxes
    assert clean_boxes([[10, 20, 30, 40], [100, 200, 300, 400]]) == [[10, 20, 30, 40], [100, 200, 300, 400]]
    
    # Degenerate boxes (x2<=x1 or y2<=y1)
    assert clean_boxes([[30, 40, 10, 20], [10, 20, 10, 40]]) == []
    
    # Non-numeric
    assert clean_boxes([["a", "b", "c", "d"]]) == []
    
    # Wrong length
    assert clean_boxes([[10, 20, 30]]) == []
    
    # Out of bounds
    assert clean_boxes([[1500, 200, 2000, 400]]) == []

def test_scale_01_to_1000():
    assert scale_01_to_1000([0.1, 0.2, 0.3, 0.4]) == [100, 200, 300, 400]
    assert scale_01_to_1000([0.0, 0.0, 1.0, 1.0]) == [0, 0, 1000, 1000]

def test_scale_1000_to_01():
    assert scale_1000_to_01([100, 200, 300, 400]) == [0.1, 0.2, 0.3, 0.4]
    assert scale_1000_to_01([0, 0, 1000, 1000]) == [0.0, 0.0, 1.0, 1.0]

def test_is_valid_box():
    # Valid
    assert is_valid_box([10, 20, 30, 40]) is True
    
    # None
    assert is_valid_box(None) is False
    
    # Wrong length
    assert is_valid_box([10, 20, 30]) is False
    
    # Non-numeric
    assert is_valid_box([10, "20", 30, 40]) is False
    
    # Zero-area (or very small)
    assert is_valid_box([10, 20, 10, 40]) is False
    
    # Negative coords
    assert is_valid_box([-10, 20, 30, 40]) is False
