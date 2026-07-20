"""
N3 diagnostic: greedy_multibox_iou true-negative behavior.

Tests four cases with hand-computed expected IoU values to expose
the TN=1.0 inflation issue before any fix is applied.

Hand computations for each case:

(a) Both empty [True Negative]:
    No boxes on either side. Nothing to measure.
    Current code: returns (1.0, 0.0, 0.0) — treats "nothing" as "perfect"
    Expected:     returns (0.0, 0.0, 0.0) — nothing to measure → 0 contribution

(b) Pred empty, GT non-empty [False Negative]:
    GT has one box [0, 0, 0.5, 0.5] → area = 0.25
    Pred has nothing → missed detection.
    Current code: returns (0.0, 0.0, 0.25)  ← correct
    Expected:     returns (0.0, 0.0, 0.25)

(c) Pred non-empty, GT empty [False Positive]:
    Pred has one box [0, 0, 0.5, 0.5] → area = 0.25
    GT has nothing → hallucinated detection.
    Current code: returns (0.0, 0.0, 0.25)  ← correct
    Expected:     returns (0.0, 0.0, 0.25)

(d) Both non-empty with known overlap [Normal Case]:
    Pred = [[0, 0, 1.0, 0.5]], GT = [[0, 0, 1.0, 1.0]]
    intersection = [0, 0, 1.0, 0.5] → area = 0.5
    area_pred = 1.0 * 0.5 = 0.5
    area_gt = 1.0 * 1.0 = 1.0
    union = 0.5 + 1.0 - 0.5 = 1.0
    IoU = 0.5 / 1.0 = 0.5
    Current code: returns (0.5, 0.5, 1.0)   ← correct
    Expected:     returns (0.5, 0.5, 1.0)
"""
import pytest
from data.box_utils import greedy_multibox_iou


class TestGreedyMultiboxIoUCurrentBehavior:
    """Asserts CURRENT (pre-fix) behavior to document the TN=1.0 issue."""

    def test_a_true_negative_both_empty(self):
        """TN: no GT boxes, no pred boxes → current code returns IoU=0.0 (N3 fixed)."""
        iou, inter, union = greedy_multibox_iou([], [])
        # Fixed behavior: IoU=0.0 for "nothing vs nothing" to avoid inflation
        assert iou == 0.0, f"Expected 0.0 for TN, got {iou}"
        assert inter == 0.0
        assert union == 0.0

    def test_b_false_negative_pred_empty(self):
        """FN: GT has boxes, pred is empty → IoU=0.0 (correct)."""
        gt = [[0.0, 0.0, 0.5, 0.5]]  # area = 0.25
        iou, inter, union = greedy_multibox_iou([], gt)
        assert iou == 0.0
        assert inter == 0.0
        assert abs(union - 0.25) < 1e-9

    def test_c_false_positive_gt_empty(self):
        """FP: pred has boxes, GT is empty → IoU=0.0 (correct)."""
        pred = [[0.0, 0.0, 0.5, 0.5]]  # area = 0.25
        iou, inter, union = greedy_multibox_iou(pred, [])
        assert iou == 0.0
        assert inter == 0.0
        assert abs(union - 0.25) < 1e-9

    def test_d_normal_overlap(self):
        """Normal: both have boxes with 50% IoU (correct)."""
        pred = [[0.0, 0.0, 1.0, 0.5]]   # area = 0.5
        gt = [[0.0, 0.0, 1.0, 1.0]]      # area = 1.0
        # intersection = [0, 0, 1.0, 0.5] → area = 0.5
        # union = 0.5 + 1.0 - 0.5 = 1.0
        # IoU = 0.5 / 1.0 = 0.5
        iou, inter, union = greedy_multibox_iou(pred, gt)
        assert abs(iou - 0.5) < 1e-9
        assert abs(inter - 0.5) < 1e-9
        assert abs(union - 1.0) < 1e-9


class TestTrueNegativeInflationDemo:
    """Demonstrates how TN=1.0 inflates macro-averaged IoU.

    Scenario: 3 images evaluated for one object class.
      - Image 1: Object exists, pred matches with IoU=0.4
      - Image 2: Object exists, pred matches with IoU=0.6
      - Image 3: Object does NOT exist, pred correctly abstains (TN)

    With TN=1.0 (current): macro IoU = (0.4 + 0.6 + 1.0) / 3 = 0.6667
    With TN=0.0 (fixed):   macro IoU = (0.4 + 0.6 + 0.0) / 3 = 0.3333
    Excluding TNs entirely: macro IoU = (0.4 + 0.6) / 2       = 0.5000

    The "correct" number for reporting depends on whether you want TNs
    counted at all. The paper's "IoU-Object Exist" excludes them.
    The paper's "IoU-Total" includes FP/FN as 0 but is silent on TN.
    """

    def test_inflation_demo_fixed(self):
        """Shows fixed TN=0.0 preventing inflation."""
        # Simulate what metrics_grounding.py does: collect per-image IoUs
        per_image_ious = []

        # Image 1: object exists, partial overlap
        pred1 = [[0.0, 0.0, 0.8, 0.5]]  # area = 0.4
        gt1 = [[0.0, 0.0, 1.0, 1.0]]    # area = 1.0
        iou1, _, _ = greedy_multibox_iou(pred1, gt1)
        per_image_ious.append(iou1)

        # Image 2: object exists, better overlap
        pred2 = [[0.0, 0.0, 1.0, 0.6]]  # area = 0.6
        gt2 = [[0.0, 0.0, 1.0, 1.0]]    # area = 1.0
        iou2, _, _ = greedy_multibox_iou(pred2, gt2)
        per_image_ious.append(iou2)

        # Image 3: TN — no object in GT, model correctly abstains
        iou3, _, _ = greedy_multibox_iou([], [])
        per_image_ious.append(iou3)

        # Fixed behavior: TN contributes 0.0
        assert iou3 == 0.0

        # Macro average with TN=0.0 (FIXED)
        macro_current = sum(per_image_ious) / len(per_image_ious)
        assert abs(macro_current - (0.4 + 0.6 + 0.0) / 3) < 1e-9
        # = 0.3333 — un-inflated score
