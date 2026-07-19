import pytest
import math

from data.box_utils import (
    compute_iou,
    greedy_multibox_iou,
    scale_1000_to_01,
    clean_boxes,
    normalize_boxes
)
from evaluation.metrics_grounding import compute_grounding_metrics
from core.constants import GROUNDING_CLASSES

def test_compute_iou():
    """Test standard single bounding box IoU logic."""
    # Exact match
    boxA = [0, 0, 1, 1]
    boxB = [0, 0, 1, 1]
    iou, inter, union = compute_iou(boxA, boxB)
    assert iou == 1.0
    assert inter == 1.0
    assert union == 1.0

    # No overlap
    boxC = [2, 2, 3, 3]
    iou, inter, union = compute_iou(boxA, boxC)
    assert iou == 0.0
    assert inter == 0.0
    assert union == 2.0

    # Partial overlap
    boxD = [0.5, 0.5, 1.5, 1.5]
    iou, inter, union = compute_iou(boxA, boxD)
    assert math.isclose(iou, 0.25 / 1.75) # inter=0.25, union=1+1-0.25=1.75

def test_normalize_boxes():
    """Test that we correctly convert raw outputs to list-of-lists."""
    # Empty
    assert normalize_boxes([]) == []
    assert normalize_boxes(None) == []
    
    # Already normalized
    assert normalize_boxes([[0, 0, 1, 1], [0.5, 0.5, 1, 1]]) == [[0, 0, 1, 1], [0.5, 0.5, 1, 1]]
    
    # Flat box (The bug we fixed!)
    assert normalize_boxes([0, 0, 1, 1]) == [[0, 0, 1, 1]]
    
    # Invalid flat (wrong length) - normalize_boxes just wraps it, clean_boxes handles length validation
    assert normalize_boxes([0, 0, 1]) == [[0, 0, 1]]

def test_clean_boxes():
    """Test that we filter out invalid or mathematically impossible coordinates."""
    # Valid
    boxes = [[0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8]]
    assert clean_boxes(boxes) == boxes
    
    # Invalid (x2 <= x1)
    bad_boxes = [[1, 1, 0, 0], [0, 0, 0, 0]]
    assert clean_boxes(bad_boxes) == []
    
    # Mixed
    mixed = [[0, 0, 1, 1], [1, 1, 0, 0]]
    assert clean_boxes(mixed) == [[0, 0, 1, 1]]
    
    # Non-numeric
    assert clean_boxes([["a", "b", "c", "d"]]) == []
    
    # Invalid length (this handles the [0, 0, 1] case)
    assert clean_boxes([[0, 0, 1]]) == []
    
    # Out of bounds for 1000 scale
    assert clean_boxes([[1500, -500, 2000, 100]]) == []

def test_scale_1000_to_01():
    """Test bounding box scaling."""
    assert scale_1000_to_01([0, 0, 1000, 1000]) == [0.0, 0.0, 1.0, 1.0]
    assert scale_1000_to_01([500, 500, 750, 750]) == [0.5, 0.5, 0.75, 0.75]
    # Scaling just divides by 1000, out-of-bounds rejection is handled by clean_boxes
    assert scale_1000_to_01([1500, -500, 2000, 100]) == [1.5, -0.5, 2.0, 0.1]

def test_greedy_multibox_iou():
    """Test greedy matching logic and True Negative/FP/FN edge cases."""
    # True Negative (Neither model nor GT have boxes)
    iou, inter, union = greedy_multibox_iou([], [])
    assert iou == 1.0
    assert inter == 0.0
    assert union == 0.0
    
    # False Positive (predicts box, GT empty)
    iou, inter, union = greedy_multibox_iou([[0, 0, 1, 1]], [])
    assert iou == 0.0
    assert inter == 0.0
    assert union == 1.0
    
    # False Negative (GT box, pred empty)
    iou, inter, union = greedy_multibox_iou([], [[0, 0, 1, 1]])
    assert iou == 0.0
    assert inter == 0.0
    assert union == 1.0
    
    # Multiple boxes matching
    pred = [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]
    gt = [[0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1]]
    iou, inter, union = greedy_multibox_iou(pred, gt)
    
    # Box 1 matches perfectly (inter=0.25, union=0.25)
    # Box 2 matches partially (inter=0.16, union=0.25+0.16-0.16 = 0.25)
    # Total inter = 0.41, union = 0.50
    # Average IoU = (1.0 + (0.16/0.25)) / 2 = (1.0 + 0.64)/2 = 0.82
    assert math.isclose(iou, 0.82)
    assert math.isclose(inter, 0.41)
    assert math.isclose(union, 0.5)

def test_compute_grounding_metrics():
    """
    Test the full metric aggregation including the difference between 
    Macro vs Micro and Total vs Exist logic from the dataset paper.
    """
    # Mock data
    # 2 images. 
    # Img 1: has excavator, no rebar. Model predicts excavator perfectly, no rebar.
    # Img 2: has rebar, no excavator. Model predicts rebar poorly (IoU=0), predicts excavator (FP).
    
    # GT
    refs = [
        {"excavator": [[0, 0, 0.5, 0.5]]}, # Img 1
        {"rebar": [[0.5, 0.5, 1.0, 1.0]]}  # Img 2
    ]
    
    # Preds (in 1000 scale)
    preds = [
        {"excavator": [[0, 0, 500, 500]]}, # Img 1 (Perfect)
        {"rebar": [[0, 0, 100, 100]], "excavator": [[0, 0, 1000, 1000]]} # Img 2 (Poor rebar, FP excavator)
    ]
    
    res = compute_grounding_metrics(preds, refs)
    
    # --- Excavator Math Analysis ---
    # Img 1: TP, IoU=1.0, inter=0.25, union=0.25. GT Exists=True
    # Img 2: FP, IoU=0.0, inter=0.0, union=1.0. GT Exists=False
    # Total Macro: (1.0 + 0.0) / 2 = 0.5
    # Total Micro: inter(0.25+0) / union(0.25+1.0) = 0.25 / 1.25 = 0.2
    # Exist Macro: Img 1 only -> 1.0
    # Exist Micro: Img 1 only -> 0.25 / 0.25 = 1.0
    
    assert res["grounding_iou_all_macro_excavator"] == 0.5
    assert math.isclose(res["grounding_iou_all_micro_excavator"], 0.2)
    assert res["grounding_iou_existing_macro_excavator"] == 1.0
    assert res["grounding_iou_existing_micro_excavator"] == 1.0
    
    # --- Rebar Math Analysis ---
    # Img 1: TN, IoU=1.0, inter=0, union=0. GT Exists=False
    # Img 2: TP but IoU=0, inter=0, union=0.26 (0.01 + 0.25). GT Exists=True
    # Total Macro: (1.0 + 0.0) / 2 = 0.5
    # Total Micro: inter(0+0) / union(0+0.26) = 0.0
    # Exist Macro: Img 2 only -> 0.0
    # Exist Micro: Img 2 only -> 0.0 / 0.26 = 0.0
    
    assert res["grounding_iou_all_macro_rebar"] == 0.5
    assert res["grounding_iou_all_micro_rebar"] == 0.0
    assert res["grounding_iou_existing_macro_rebar"] == 0.0
    assert res["grounding_iou_existing_micro_rebar"] == 0.0
    
    # Check that empty edge cases don't crash
    empty_res = compute_grounding_metrics([], [])
    assert empty_res == {}
