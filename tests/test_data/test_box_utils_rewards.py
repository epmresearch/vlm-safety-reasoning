"""
Tests for data/box_utils.py functions used by reward functions.
Focus: scale conversion, IoU computation, mask union IoU.
"""
import pytest
from data.box_utils import (
    scale_1000_to_01,
    scale_01_to_1000,
    compute_mask_union_iou,
    greedy_multibox_iou,
    clean_boxes,
    normalize_boxes,
)


class TestScaleConversion:
    def test_scale_1000_to_01_basic(self):
        result = scale_1000_to_01([100, 200, 500, 700])
        assert result == pytest.approx([0.1, 0.2, 0.5, 0.7])

    def test_scale_1000_to_01_full_range(self):
        result = scale_1000_to_01([0, 0, 1000, 1000])
        assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_scale_1000_to_01_clips_over_range(self):
        result = scale_1000_to_01([0, 0, 1100, 1000])
        assert result[2] == pytest.approx(1.0)

    def test_scale_1000_to_01_clips_negative(self):
        result = scale_1000_to_01([-50, 0, 500, 500])
        assert result[0] == pytest.approx(0.0)

    def test_roundtrip(self):
        original = [100.0, 200.0, 500.0, 700.0]
        scaled = scale_01_to_1000(scale_1000_to_01(original))
        assert scaled == pytest.approx(original, abs=1)


class TestMaskUnionIoU:
    def test_perfect_overlap_single_box(self):
        box = [[0.1, 0.1, 0.9, 0.9]]
        result = compute_mask_union_iou(box, box)
        assert result["iou"] == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = [[0.0, 0.0, 0.4, 0.4]]
        gt = [[0.6, 0.6, 1.0, 1.0]]
        result = compute_mask_union_iou(pred, gt)
        assert result["iou"] == pytest.approx(0.0)

    def test_true_negative_returns_none(self):
        """Both empty → iou=None (true negative, caller decides score)."""
        result = compute_mask_union_iou([], [])
        assert result["iou"] is None

    def test_false_positive_pred_only(self):
        result = compute_mask_union_iou([[0.1, 0.1, 0.5, 0.5]], [])
        assert result["iou"] == pytest.approx(0.0)

    def test_false_negative_gt_only(self):
        result = compute_mask_union_iou([], [[0.1, 0.1, 0.5, 0.5]])
        assert result["iou"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        pred = [[0.0, 0.0, 0.5, 1.0]]
        gt = [[0.5, 0.0, 1.0, 1.0]]
        result = compute_mask_union_iou(pred, gt)
        assert result["iou"] == pytest.approx(0.0, abs=0.01)  # edge touching, no overlap

    def test_multiple_boxes_merged(self):
        """Two pred boxes that together cover the GT box exactly."""
        gt = [[0.0, 0.0, 1.0, 1.0]]
        pred = [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
        result = compute_mask_union_iou(pred, gt)
        assert result["iou"] == pytest.approx(1.0, abs=0.01)


class TestNormalizeBoxes:
    def test_flat_single_box_wrapped(self):
        result = normalize_boxes([0.1, 0.2, 0.5, 0.6])
        assert result == [[0.1, 0.2, 0.5, 0.6]]

    def test_nested_list_returned_as_is(self):
        boxes = [[0.1, 0.2, 0.5, 0.6], [0.3, 0.3, 0.7, 0.7]]
        result = normalize_boxes(boxes)
        assert result == boxes

    def test_none_returns_empty(self):
        assert normalize_boxes(None) == []

    def test_empty_list_returns_empty(self):
        assert normalize_boxes([]) == []


class TestCleanBoxes:
    def test_valid_boxes_pass_through(self):
        boxes = [[0.1, 0.1, 0.9, 0.9]]
        assert clean_boxes(boxes) == boxes

    def test_degenerate_zero_width_removed(self):
        boxes = [[0.5, 0.1, 0.5, 0.9]]  # zero width
        assert clean_boxes(boxes) == []

    def test_none_input_returns_empty(self):
        assert clean_boxes(None) == []

    def test_mixed_valid_invalid(self):
        boxes = [[0.1, 0.1, 0.9, 0.9], [0.5, 0.5, 0.5, 0.8]]  # second is degenerate
        result = clean_boxes(boxes)
        assert len(result) == 1
        assert result[0] == [0.1, 0.1, 0.9, 0.9]