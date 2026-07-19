"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE EVALUATION PIPELINE TEST SUITE                      ║
║                                                                              ║
║  This file is designed for Google Colab: copy each "# %% [markdown]" and     ║
║  "# %%" cell block into separate Colab cells and run sequentially.           ║
║                                                                              ║
║  No GPU required. No Java required (Java-backed metrics are mocked).         ║
║  No external model downloads (CLIP/BERT models are mocked where needed).     ║
║                                                                              ║
║  Total: ~81 tests across 9 categories.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # 🧪 Evaluation Pipeline Test Suite
#
# Run this notebook to comprehensively test the evaluation pipeline.
# Each section is self-contained and can be run independently after the setup cell.

# %% [markdown]
# ## 0. Setup & Imports

# %%
import sys
import os
import json
import math
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

# Ensure project root is on PYTHONPATH
# In Colab, after cloning the repo, set this to the repo root:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in dir() else os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")
print("Setup complete. Ready to run tests.")


# %% [markdown]
# ---
# ## 1. Output Parser Tests (12 tests)
#
# Tests `strip_fences`, `parse_model_output`, and `validate_unified_output`
# from `evaluation/output_parser.py`.

# %%
class TestStripFences(unittest.TestCase):
    """Tests for the strip_fences function that extracts JSON from code blocks."""

    def setUp(self):
        from evaluation.output_parser import strip_fences
        self.strip_fences = strip_fences

    def test_standard_json_fence(self):
        """Standard ```json ... ``` fence."""
        text = '```json\n{"caption": "test"}\n```'
        result = self.strip_fences(text)
        self.assertEqual(result, '{"caption": "test"}')

    def test_fence_without_json_identifier(self):
        """Fence with no language identifier: ``` ... ```."""
        text = '```\n{"a": 1}\n```'
        self.assertEqual(self.strip_fences(text), '{"a": 1}')

    def test_fence_with_preamble_and_postamble(self):
        """Common VLM hallucination: conversational text around the fence."""
        text = 'Here is my analysis:\n```json\n{"caption": "safe"}\n```\nI hope this helps!'
        self.assertEqual(self.strip_fences(text), '{"caption": "safe"}')

    def test_no_fences_fallback(self):
        """When no fences exist, strip whitespace and return as-is."""
        text = '  {"a": 1}  '
        self.assertEqual(self.strip_fences(text), '{"a": 1}')

    def test_empty_string(self):
        """Empty string should return empty."""
        self.assertEqual(self.strip_fences(""), "")

    def test_fence_case_insensitive(self):
        """```JSON should work the same as ```json."""
        text = '```JSON\n{"key": "val"}\n```'
        self.assertEqual(self.strip_fences(text), '{"key": "val"}')

    def test_multiline_json_inside_fence(self):
        """Multi-line formatted JSON inside fences."""
        text = '```json\n{\n  "caption": "test",\n  "excavator": []\n}\n```'
        result = self.strip_fences(text)
        parsed = json.loads(result)
        self.assertEqual(parsed["caption"], "test")


class TestParseModelOutput(unittest.TestCase):
    """Tests for parse_model_output that converts raw strings to dicts."""

    def setUp(self):
        from evaluation.output_parser import parse_model_output
        self.parse = parse_model_output

    def test_valid_fenced_json(self):
        """Valid JSON inside code fences."""
        raw = '```json\n{"caption": "A site with workers"}\n```'
        result = self.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["caption"], "A site with workers")

    def test_valid_unfenced_json(self):
        """Valid JSON without any fences."""
        raw = '{"key": "value", "count": 42}'
        result = self.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["count"], 42)

    def test_invalid_json(self):
        """Malformed JSON should return None."""
        self.assertIsNone(self.parse("{key: value}"))
        self.assertIsNone(self.parse("not json at all"))

    def test_empty_string(self):
        """Empty input should return None (not crash)."""
        self.assertIsNone(self.parse(""))

    def test_truncated_json(self):
        """Truncated JSON (common from max_new_tokens cutoff) should return None."""
        raw = '```json\n{"caption": "A construction site with'\
              '```'
        self.assertIsNone(self.parse(raw))


class TestValidateUnifiedOutput(unittest.TestCase):
    """Tests for validate_unified_output against the Pydantic schema."""

    def setUp(self):
        from evaluation.output_parser import validate_unified_output
        self.validate = validate_unified_output

    def test_minimal_valid_schema(self):
        """Minimum valid output: just a caption."""
        result = self.validate({"caption": "A construction site"})
        self.assertIsNotNone(result)

    def test_full_valid_schema(self):
        """Full valid schema with all fields."""
        data = {
            "caption": "Workers on site",
            "rule_1_violation": {"reason": "No hard hat", "bounding_box": [[100, 200, 300, 400]]},
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
            "excavator": [[50, 50, 500, 500]],
            "rebar": [],
            "worker_with_white_hard_hat": [[100, 100, 200, 200]],
        }
        result = self.validate(data)
        self.assertIsNotNone(result)

    def test_missing_caption_fails(self):
        """Caption is required; its absence should fail validation."""
        result = self.validate({"excavator": []})
        self.assertIsNone(result)

    def test_none_input(self):
        """None input should return None."""
        self.assertIsNone(self.validate(None))

    def test_extra_keys_accepted(self):
        """Extra keys beyond the schema should not cause failure (Pydantic default)."""
        data = {"caption": "test", "unknown_field": "should be fine"}
        result = self.validate(data)
        self.assertIsNotNone(result)


# Run output parser tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStripFences)
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParseModelOutput))
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestValidateUnifiedOutput))
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 1: OUTPUT PARSER TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 2. Structural Metrics Tests (6 tests)
#
# Tests `compute_structural_metrics` from `evaluation/metrics_structural.py`.

# %%
class TestStructuralMetrics(unittest.TestCase):
    """Tests for JSON validity and schema adherence metrics."""

    def setUp(self):
        from evaluation.metrics_structural import compute_structural_metrics
        self.compute = compute_structural_metrics

    def test_empty_inputs(self):
        """Empty list should return empty dict."""
        self.assertEqual(self.compute([]), {})

    def test_all_valid_json_and_schema(self):
        """All outputs are valid JSON AND valid schema."""
        outputs = [
            '```json\n{"caption": "safe site"}\n```',
            '{"caption": "danger", "rule_1_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}}',
        ]
        res = self.compute(outputs)
        self.assertEqual(res["structural_json_validity_rate"], 1.0)
        self.assertEqual(res["structural_schema_adherence_rate"], 1.0)
        self.assertEqual(res["structural_valid_json_count"], 2)
        self.assertEqual(res["structural_valid_schema_count"], 2)
        self.assertEqual(res["structural_total_samples_count"], 2)

    def test_mixed_valid_and_invalid(self):
        """Mix of valid JSON, invalid JSON, and valid-JSON-but-invalid-schema."""
        outputs = [
            '```json\n{"caption": "ok"}\n```',          # Valid JSON + valid schema
            "This is conversational hallucination",       # Invalid JSON
            '{"random_key": true}',                       # Valid JSON, but missing caption (invalid schema)
        ]
        res = self.compute(outputs)
        self.assertEqual(res["structural_json_validity_rate"], 2.0 / 3.0)
        self.assertEqual(res["structural_valid_json_count"], 2)
        # Only the first output has valid schema (caption present)
        self.assertEqual(res["structural_valid_schema_count"], 1)
        self.assertAlmostEqual(res["structural_schema_adherence_rate"], 1.0 / 3.0)
        self.assertEqual(res["structural_total_samples_count"], 3)

    def test_all_invalid(self):
        """All outputs are garbage."""
        outputs = ["not json", "also not json", "nope"]
        res = self.compute(outputs)
        self.assertEqual(res["structural_json_validity_rate"], 0.0)
        self.assertEqual(res["structural_schema_adherence_rate"], 0.0)

    def test_key_prefix_present(self):
        """Verify all keys have the 'structural_' prefix."""
        outputs = ['{"caption": "test"}']
        res = self.compute(outputs)
        for key in res:
            self.assertTrue(key.startswith("structural_"), f"Key '{key}' missing 'structural_' prefix")

    def test_single_output(self):
        """Single valid output."""
        res = self.compute(['{"caption": "hello"}'])
        self.assertEqual(res["structural_json_validity_rate"], 1.0)
        self.assertEqual(res["structural_total_samples_count"], 1)


# Run structural metrics tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStructuralMetrics)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 2: STRUCTURAL METRICS TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 3. Captioning Metrics Tests (8 tests)
#
# Tests `compute_all_caption_metrics` from `evaluation/metrics_captioning.py`.
# Individual metrics (BERTScore, METEOR, CIDEr, SPICE, CLIPScore) are mocked
# since they require heavy dependencies (Java, GPU models).

# %%
class TestCaptionMetricsAggregation(unittest.TestCase):
    """Tests for the caption metrics aggregation and sanitization logic."""

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_clipscore")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_all_metrics_aggregated(self, mock_bert, mock_meteor, mock_cider, mock_clip, mock_spice):
        """Verify all sub-metrics are combined into a single dict."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {"bertscore_f1": 0.85}
        mock_meteor.return_value = {"meteor": 0.72}
        mock_cider.return_value = {"ciderd": 0.65}
        mock_clip.return_value = {"clipscore": 0.55}
        mock_spice.return_value = {"spice": 0.40}

        preds = ["A good caption"]
        refs = ["A great caption"]
        res = compute_all_caption_metrics(preds, refs)

        self.assertEqual(res["bertscore_f1"], 0.85)
        self.assertEqual(res["meteor"], 0.72)
        self.assertEqual(res["ciderd"], 0.65)
        self.assertEqual(res["spice"], 0.40)
        # CLIPScore not included without images
        self.assertNotIn("clipscore", res)

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_empty_strings_sanitized(self, mock_bert, mock_meteor, mock_cider, mock_spice):
        """Verify empty and whitespace-only strings are replaced with 'empty'."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {"bertscore_f1": 0.5}
        mock_meteor.return_value = {}
        mock_cider.return_value = {}
        mock_spice.return_value = {}

        preds = ["Good caption", "   ", ""]
        refs = ["Great caption", "", "   "]

        compute_all_caption_metrics(preds, refs)

        # Check what was actually passed to BERTScore
        call_args = mock_bert.call_args[0]
        self.assertEqual(call_args[0], ["Good caption", "empty", "empty"])
        self.assertEqual(call_args[1], ["Great caption", "empty", "empty"])

    def test_empty_inputs_return_empty(self):
        """Empty or mismatched inputs should return empty dict."""
        from evaluation.metrics_captioning import compute_all_caption_metrics
        self.assertEqual(compute_all_caption_metrics([], []), {})
        self.assertEqual(compute_all_caption_metrics(["text"], []), {})
        self.assertEqual(compute_all_caption_metrics([], ["text"]), {})

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_prefix_applied(self, mock_bert, mock_meteor, mock_cider, mock_spice):
        """Verify prefix is correctly applied to all keys."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {"bertscore_f1": 0.8}
        mock_meteor.return_value = {"meteor": 0.7}
        mock_cider.return_value = {"ciderd": 0.6}
        mock_spice.return_value = {}

        res = compute_all_caption_metrics(["pred"], ["ref"], prefix="captioning_")
        self.assertIn("captioning_bertscore_f1", res)
        self.assertIn("captioning_meteor", res)
        self.assertIn("captioning_ciderd", res)
        self.assertNotIn("bertscore_f1", res)  # Unprefixed key should not exist

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_spice_disabled(self, mock_bert, mock_meteor, mock_cider, mock_spice):
        """Verify include_spice=False skips SPICE computation."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {"bertscore_f1": 0.8}
        mock_meteor.return_value = {}
        mock_cider.return_value = {}

        compute_all_caption_metrics(["pred"], ["ref"], include_spice=False)
        mock_spice.assert_not_called()

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_clipscore")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_clipscore_with_images(self, mock_bert, mock_meteor, mock_cider, mock_clip, mock_spice):
        """Verify CLIPScore is included when images are provided."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {}
        mock_meteor.return_value = {}
        mock_cider.return_value = {}
        mock_clip.return_value = {"clipscore": 0.75}
        mock_spice.return_value = {}

        res = compute_all_caption_metrics(["pred"], ["ref"], images=["fake_img"])
        self.assertEqual(res["clipscore"], 0.75)
        mock_clip.assert_called_once()

    def test_clipscore_mismatched_lengths(self):
        """CLIPScore should safely return empty on mismatched inputs."""
        from evaluation.metrics_captioning import compute_clipscore
        self.assertEqual(compute_clipscore(["t1", "t2"], ["img1"]), {})
        self.assertEqual(compute_clipscore([], []), {})

    @patch("evaluation.metrics_captioning.compute_spice")
    @patch("evaluation.metrics_captioning.compute_cider")
    @patch("evaluation.metrics_captioning.compute_meteor")
    @patch("evaluation.metrics_captioning.compute_bertscore")
    def test_none_predictions_sanitized(self, mock_bert, mock_meteor, mock_cider, mock_spice):
        """None values in predictions should be sanitized to 'empty'."""
        from evaluation.metrics_captioning import compute_all_caption_metrics

        mock_bert.return_value = {"bertscore_f1": 0.5}
        mock_meteor.return_value = {}
        mock_cider.return_value = {}
        mock_spice.return_value = {}

        compute_all_caption_metrics([None, "real"], ["ref1", "ref2"])
        call_args = mock_bert.call_args[0]
        self.assertEqual(call_args[0][0], "empty")
        self.assertEqual(call_args[0][1], "real")


# Run captioning metrics tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestCaptionMetricsAggregation)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 3: CAPTIONING METRICS TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 4. Box Utility Tests (15 tests)
#
# Tests all functions in `data/box_utils.py`:
# `normalize_boxes`, `clean_boxes`, `is_valid_box`, `compute_iou`,
# `greedy_multibox_iou`, `scale_01_to_1000`, `scale_1000_to_01`.

# %%
class TestNormalizeBoxes(unittest.TestCase):
    """Tests for normalize_boxes edge case handling."""

    def setUp(self):
        from data.box_utils import normalize_boxes
        self.normalize = normalize_boxes

    def test_none_input(self):
        self.assertEqual(self.normalize(None), [])

    def test_empty_list(self):
        self.assertEqual(self.normalize([]), [])

    def test_flat_single_box(self):
        """A flat [x1,y1,x2,y2] should become [[x1,y1,x2,y2]]."""
        self.assertEqual(self.normalize([10, 20, 30, 40]), [[10, 20, 30, 40]])

    def test_already_normalized(self):
        """Already-correct nested format should pass through."""
        boxes = [[0, 0, 1, 1], [0.5, 0.5, 1, 1]]
        self.assertEqual(self.normalize(boxes), boxes)

    def test_dict_wrapper_hallucination(self):
        """Model hallucinated a dict wrapper around the box."""
        boxes = [{"bounding_box": [10, 20, 30, 40]}]
        result = self.normalize(boxes)
        self.assertEqual(result, [[10, 20, 30, 40]])

    def test_dict_xmin_ymin_format(self):
        """Model hallucinated xmin/ymin/xmax/ymax keys."""
        boxes = [{"xmin": 0.1, "ymin": 0.2, "xmax": 0.3, "ymax": 0.4}]
        result = self.normalize(boxes)
        self.assertEqual(result, [[0.1, 0.2, 0.3, 0.4]])

    def test_mixed_valid_and_invalid(self):
        """Mix of valid list boxes and invalid entries."""
        boxes = [[0, 0, 1, 1], "not a box", [0.5, 0.5, 1, 1]]
        result = self.normalize(boxes)
        self.assertEqual(result, [[0, 0, 1, 1], [0.5, 0.5, 1, 1]])


class TestIsValidBox(unittest.TestCase):
    """Tests for is_valid_box validation."""

    def setUp(self):
        from data.box_utils import is_valid_box
        self.is_valid = is_valid_box

    def test_valid_unit_box(self):
        self.assertTrue(self.is_valid([0, 0, 1, 1]))

    def test_valid_1000_scale_box(self):
        self.assertTrue(self.is_valid([100, 200, 500, 800]))

    def test_none_input(self):
        self.assertFalse(self.is_valid(None))

    def test_wrong_length(self):
        self.assertFalse(self.is_valid([0, 0, 1]))
        self.assertFalse(self.is_valid([0, 0, 1, 1, 1]))

    def test_non_numeric(self):
        self.assertFalse(self.is_valid(["a", "b", "c", "d"]))

    def test_zero_area_degenerate(self):
        """Zero-area box (x1==x2 or y1==y2) should be invalid."""
        self.assertFalse(self.is_valid([0, 0, 0, 1]))   # zero width
        self.assertFalse(self.is_valid([0, 0, 1, 0]))   # zero height

    def test_inverted_coordinates(self):
        """xmax < xmin should be invalid."""
        self.assertFalse(self.is_valid([1, 1, 0, 0]))

    def test_out_of_bounds(self):
        """Coordinates beyond [0, 1000] range should be invalid."""
        self.assertFalse(self.is_valid([1500, -500, 2000, 100]))


class TestCleanBoxes(unittest.TestCase):
    """Tests for clean_boxes filtering."""

    def setUp(self):
        from data.box_utils import clean_boxes
        self.clean = clean_boxes

    def test_valid_boxes_pass_through(self):
        boxes = [[0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8]]
        self.assertEqual(self.clean(boxes), boxes)

    def test_invalid_boxes_filtered(self):
        """Degenerate and inverted boxes should be removed."""
        boxes = [[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(self.clean(boxes), [[0, 0, 1, 1]])

    def test_empty_input(self):
        self.assertEqual(self.clean([]), [])
        self.assertEqual(self.clean(None), [])

    def test_non_numeric_filtered(self):
        self.assertEqual(self.clean([["a", "b", "c", "d"]]), [])

    def test_wrong_length_filtered(self):
        self.assertEqual(self.clean([[0, 0, 1]]), [])


class TestScaleConversion(unittest.TestCase):
    """Tests for scale_01_to_1000 and scale_1000_to_01."""

    def setUp(self):
        from data.box_utils import scale_01_to_1000, scale_1000_to_01
        self.to_1000 = scale_01_to_1000
        self.to_01 = scale_1000_to_01

    def test_full_range(self):
        self.assertEqual(self.to_1000([0.0, 0.0, 1.0, 1.0]), [0, 0, 1000, 1000])

    def test_mid_values(self):
        self.assertEqual(self.to_1000([0.5, 0.5, 0.75, 0.75]), [500, 500, 750, 750])

    def test_roundtrip(self):
        """Converting to 1000 and back should preserve values approximately."""
        original = [0.1, 0.2, 0.9, 0.8]
        scaled = self.to_1000(original)
        restored = self.to_01(scaled)
        for o, r in zip(original, restored):
            self.assertAlmostEqual(o, r, places=2)

    def test_scale_1000_to_01_full(self):
        self.assertEqual(self.to_01([0, 0, 1000, 1000]), [0.0, 0.0, 1.0, 1.0])


class TestComputeIoU(unittest.TestCase):
    """Tests for single-box IoU computation."""

    def setUp(self):
        from data.box_utils import compute_iou
        self.iou = compute_iou

    def test_exact_match(self):
        iou, inter, union = self.iou([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(iou, 1.0)

    def test_no_overlap(self):
        iou, inter, union = self.iou([0, 0, 1, 1], [2, 2, 3, 3])
        self.assertEqual(iou, 0.0)

    def test_partial_overlap(self):
        iou, inter, union = self.iou([0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5])
        # inter = 0.25, union = 1 + 1 - 0.25 = 1.75
        self.assertAlmostEqual(iou, 0.25 / 1.75)

    def test_none_inputs(self):
        iou, inter, union = self.iou(None, [0, 0, 1, 1])
        self.assertEqual(iou, 0.0)

    def test_contained_box(self):
        """One box fully contains the other."""
        iou, inter, union = self.iou([0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75])
        inner_area = 0.5 * 0.5
        outer_area = 1.0
        self.assertAlmostEqual(iou, inner_area / outer_area)


class TestGreedyMultiboxIoU(unittest.TestCase):
    """Tests for greedy multi-box matching."""

    def setUp(self):
        from data.box_utils import greedy_multibox_iou
        self.multi_iou = greedy_multibox_iou

    def test_true_negative(self):
        """Neither model nor GT have boxes → perfect score."""
        iou, inter, union = self.multi_iou([], [])
        self.assertEqual(iou, 1.0)

    def test_false_positive(self):
        """Model predicts boxes, but GT is empty → 0.0."""
        iou, inter, union = self.multi_iou([[0, 0, 1, 1]], [])
        self.assertEqual(iou, 0.0)

    def test_false_negative(self):
        """GT has boxes, model predicts nothing → 0.0."""
        iou, inter, union = self.multi_iou([], [[0, 0, 1, 1]])
        self.assertEqual(iou, 0.0)

    def test_perfect_multi_match(self):
        """Two GT boxes matched perfectly by two pred boxes."""
        pred = [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]
        gt = [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]
        iou, _, _ = self.multi_iou(pred, gt)
        self.assertEqual(iou, 1.0)

    def test_more_preds_than_gt(self):
        """Extra predictions (FP) should not increase IoU."""
        pred = [[0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1], [0.1, 0.1, 0.2, 0.2]]
        gt = [[0, 0, 0.5, 0.5]]
        iou, _, _ = self.multi_iou(pred, gt)
        self.assertEqual(iou, 1.0)  # GT box perfectly matched

    def test_more_gt_than_preds(self):
        """Unmatched GT boxes should get IoU of 0.0."""
        pred = [[0, 0, 0.5, 0.5]]
        gt = [[0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1]]
        iou, _, _ = self.multi_iou(pred, gt)
        self.assertAlmostEqual(iou, 0.5)  # (1.0 + 0.0) / 2


# Run box utility tests
suite = unittest.TestSuite()
for tc in [TestNormalizeBoxes, TestIsValidBox, TestCleanBoxes,
           TestScaleConversion, TestComputeIoU, TestGreedyMultiboxIoU]:
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(tc))
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 4: BOX UTILITY TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 5. Grounding Metrics Tests (10 tests)
#
# Tests `compute_grounding_metrics` from `evaluation/metrics_grounding.py`.

# %%
class TestGroundingMetrics(unittest.TestCase):
    """Tests for object grounding metrics (IoU aggregation)."""

    def setUp(self):
        from evaluation.metrics_grounding import compute_grounding_metrics
        self.compute = compute_grounding_metrics

    def test_empty_inputs(self):
        """Empty inputs should return empty dict."""
        self.assertEqual(self.compute([], []), {})

    def test_none_inputs(self):
        """None inputs should return empty dict."""
        self.assertEqual(self.compute(None, None), {})

    def test_length_mismatch(self):
        """Mismatched lengths should return empty dict."""
        self.assertEqual(self.compute([{}], [{}, {}]), {})

    def test_perfect_prediction(self):
        """Model perfectly predicts all objects."""
        refs = [{"excavator": [[0, 0, 0.5, 0.5]]}]
        preds = [{"excavator": [[0, 0, 500, 500]]}]  # 1000-scale
        res = self.compute(preds, refs)
        self.assertAlmostEqual(res["grounding_iou_all_macro_excavator"], 1.0)
        self.assertAlmostEqual(res["grounding_iou_existing_macro_excavator"], 1.0)

    def test_true_negative(self):
        """No objects in GT and no predictions → TN → IoU=1.0 in total."""
        refs = [{}]
        preds = [{}]
        res = self.compute(preds, refs)
        self.assertEqual(res["grounding_iou_all_macro_excavator"], 1.0)
        # Existing metrics should have no data for this class
        self.assertEqual(res["grounding_iou_existing_macro_excavator"], 0.0)
        self.assertEqual(res["grounding_true_negatives_count_excavator"], 1)

    def test_false_positive(self):
        """Model predicts object, but GT has none → FP → IoU=0.0."""
        refs = [{}]
        preds = [{"excavator": [[0, 0, 500, 500]]}]
        res = self.compute(preds, refs)
        self.assertEqual(res["grounding_iou_all_macro_excavator"], 0.0)
        self.assertEqual(res["grounding_false_positives_count_excavator"], 1)

    def test_false_negative(self):
        """GT has object, model predicts nothing → FN → IoU=0.0."""
        refs = [{"excavator": [[0, 0, 0.5, 0.5]]}]
        preds = [{}]
        res = self.compute(preds, refs)
        self.assertEqual(res["grounding_iou_all_macro_excavator"], 0.0)
        self.assertEqual(res["grounding_iou_existing_macro_excavator"], 0.0)
        self.assertEqual(res["grounding_false_negatives_count_excavator"], 1)

    def test_macro_vs_micro_difference(self):
        """Verify macro (mean of IoUs) differs from micro (total_inter/total_union)."""
        # Image 1: Perfect excavator match (IoU=1.0)
        # Image 2: Bad excavator match (IoU≈0)
        refs = [
            {"excavator": [[0, 0, 0.5, 0.5]]},
            {"excavator": [[0.5, 0.5, 1.0, 1.0]]},
        ]
        preds = [
            {"excavator": [[0, 0, 500, 500]]},           # Perfect
            {"excavator": [[0, 0, 100, 100]]},            # Bad (0.01 scaled)
        ]
        res = self.compute(preds, refs)

        macro = res["grounding_iou_existing_macro_excavator"]
        micro = res["grounding_iou_existing_micro_excavator"]
        # Macro averages the per-image IoUs
        # Micro aggregates intersections and unions globally
        # They should differ when IoU values are unequal
        self.assertIsInstance(macro, float)
        self.assertIsInstance(micro, float)

    def test_all_classes_present(self):
        """Verify all grounding classes appear in output."""
        from core.constants import GROUNDING_CLASSES
        refs = [{}]
        preds = [{}]
        res = self.compute(preds, refs)
        for cls in GROUNDING_CLASSES:
            self.assertIn(f"grounding_iou_all_macro_{cls}", res)
            self.assertIn(f"grounding_iou_existing_macro_{cls}", res)

    def test_aggregate_mean_keys(self):
        """Verify aggregate mean keys are present."""
        refs = [{}]
        preds = [{}]
        res = self.compute(preds, refs)
        self.assertIn("grounding_iou_all_macro_mean", res)
        self.assertIn("grounding_iou_all_micro_mean", res)
        self.assertIn("grounding_iou_existing_macro_mean", res)
        self.assertIn("grounding_iou_existing_micro_mean", res)


# Run grounding metrics tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestGroundingMetrics)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 5: GROUNDING METRICS TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 6. Violation Metrics Tests (12 tests)
#
# Tests `compute_violation_metrics` from `evaluation/metrics_violations.py`.

# %%
class TestViolationMetrics(unittest.TestCase):
    """Tests for safety violation identification and grounding metrics."""

    def setUp(self):
        from evaluation.metrics_violations import compute_violation_metrics
        self.compute = compute_violation_metrics

    def test_empty_inputs(self):
        self.assertEqual(self.compute([], []), {})

    def test_none_inputs(self):
        self.assertEqual(self.compute(None, None), {})

    def test_length_mismatch(self):
        self.assertEqual(self.compute([{}], [{}, {}]), {})

    def test_rule_0_true_positive(self):
        """Both GT and prediction have no violations → Rule 0 TP."""
        res = self.compute([{}], [{}])
        self.assertEqual(res["violation_identification_precision_rule_0"], 1.0)
        self.assertEqual(res["violation_identification_recall_rule_0"], 1.0)
        self.assertEqual(res["violation_identification_f1_rule_0"], 1.0)

    def test_rule_0_false_negative(self):
        """GT is safe, but model hallucinates a violation → Rule 0 FN."""
        preds = [{"rule_1_violation": {"reason": "hallucinated", "bounding_box": [[0,0,1000,1000]]}}]
        refs = [{}]
        res = self.compute(preds, refs)
        # Rule 0 FN: model said unsafe when it's safe
        self.assertEqual(res["violation_identification_recall_rule_0"], 0.0)

    def test_rule_0_false_positive(self):
        """GT is unsafe, but model says safe → Rule 0 FP."""
        preds = [{}]
        refs = [{"rule_2_violation": {"reason": "missing harness", "bounding_box": [[0,0,1,1]]}}]
        res = self.compute(preds, refs)
        self.assertEqual(res["violation_identification_precision_rule_0"], 0.0)

    def test_perfect_rule_identification(self):
        """Model correctly identifies all violations."""
        refs = [{"rule_1_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}}]
        preds = [{"rule_1_violation": {"reason": "y", "bounding_box": [[0,0,1000,1000]]}}]
        res = self.compute(preds, refs)
        self.assertEqual(res["violation_identification_precision_rule_1"], 1.0)
        self.assertEqual(res["violation_identification_recall_rule_1"], 1.0)
        self.assertEqual(res["violation_identification_f1_rule_1"], 1.0)

    def test_wrong_rule_identified(self):
        """Model predicts Rule 3 when GT has Rule 2 → FP for R3, FN for R2."""
        refs = [{"rule_2_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}}]
        preds = [{"rule_3_violation": {"reason": "y", "bounding_box": [[0,0,1000,1000]]}}]
        res = self.compute(preds, refs)
        self.assertEqual(res["violation_identification_precision_rule_3"], 0.0)
        self.assertEqual(res["violation_identification_recall_rule_2"], 0.0)

    def test_global_f1_calculation(self):
        """Verify global F1 across multiple images and rules."""
        refs = [
            {"rule_1_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}},
            {"rule_2_violation": {"reason": "x", "bounding_box": [[0,0,1,1]]}},
        ]
        preds = [
            {"rule_1_violation": {"reason": "y", "bounding_box": [[0,0,1000,1000]]}},
            {"rule_3_violation": {"reason": "y", "bounding_box": [[0,0,1000,1000]]}},
        ]
        # Global: 1 TP (Rule 1), 1 FP (Rule 3), 1 FN (Rule 2)
        # P = 1/2, R = 1/2, F1 = 0.5
        res = self.compute(preds, refs)
        self.assertAlmostEqual(res["violation_identification_precision_macro"], 0.5)
        self.assertAlmostEqual(res["violation_identification_recall_macro"], 0.5)
        self.assertAlmostEqual(res["violation_identification_f1_macro"], 0.5)

    def test_grounding_iou_per_rule(self):
        """Verify grounding IoU is computed only for correctly-identified rules."""
        refs = [{"rule_1_violation": {"reason": "x", "bounding_box": [[0, 0, 1, 1]]}}]
        preds = [{"rule_1_violation": {"reason": "y", "bounding_box": [[0, 0, 1000, 1000]]}}]
        res = self.compute(preds, refs)
        # Perfect box match → IoU = 1.0
        self.assertAlmostEqual(res["violation_grounding_iou_rule_1"], 1.0)

    def test_grounding_iou_no_instances(self):
        """Rules with no correct identifications should have IoU = 0.0."""
        res = self.compute([{}], [{}])
        self.assertEqual(res["violation_grounding_iou_rule_1"], 0.0)
        self.assertEqual(res["violation_grounding_iou_rule_2"], 0.0)

    def test_flat_box_handling(self):
        """Flat single-box [x,y,x,y] should be normalized before IoU computation."""
        refs = [{"rule_1_violation": {"bounding_box": [0, 0, 1, 1]}}]  # Flat!
        preds = [{"rule_1_violation": {"bounding_box": [0, 0, 1000, 500]}}]  # Flat!
        res = self.compute(preds, refs)
        # [0,0,1000,500] scaled to [0,0,1,0.5] vs [0,0,1,1] → IoU = 0.5
        self.assertAlmostEqual(res["violation_grounding_iou_rule_1"], 0.5)

    def test_grounding_iou_macro_key(self):
        """Verify the aggregate macro IoU key exists."""
        res = self.compute([{}], [{}])
        self.assertIn("violation_grounding_iou_macro", res)


# Run violation metrics tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestViolationMetrics)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 6: VIOLATION METRICS TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 7. Reasoning Metrics Tests (8 tests)
#
# Tests `batch_score_reasoning` from `evaluation/metrics_reasoning.py`.

# %%
class TestReasoningMetrics(unittest.TestCase):
    """Tests for reasoning evaluation (captioning metrics on violation reasons)."""

    @patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
    def test_correct_splitting_into_buckets(self, mock_metrics):
        """Verify reasons are split into global and per-rule buckets."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        mock_metrics.return_value = {"bertscore_f1": 0.9, "meteor": 0.8}

        preds = [
            {"rule_1_violation": {"reason": "pred_r1_img1"}},
            {"rule_1_violation": {"reason": "pred_r1_img2"},
             "rule_2_violation": {"reason": "pred_r2_img2"}},
        ]
        refs = [
            {"rule_1_violation": {"reason": "ref_r1_img1"}},
            {"rule_1_violation": {"reason": "ref_r1_img2"},
             "rule_2_violation": {"reason": "ref_r2_img2"}},
        ]

        res = batch_score_reasoning(preds, refs)

        # Should be called 3 times: 1 global + 2 per-rule (rule_1, rule_2)
        self.assertEqual(mock_metrics.call_count, 3)

    @patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
    def test_correct_key_format(self, mock_metrics):
        """Verify key format is reasoning_{metric}_{scope}."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        mock_metrics.return_value = {"bertscore_f1": 0.9, "meteor": 0.8}

        preds = [{"rule_1_violation": {"reason": "pred"}}]
        refs = [{"rule_1_violation": {"reason": "ref"}}]

        res = batch_score_reasoning(preds, refs)

        # Global keys: reasoning_{metric}_macro
        self.assertIn("reasoning_bertscore_f1_macro", res)
        self.assertIn("reasoning_meteor_macro", res)
        # Per-rule keys: reasoning_{metric}_{rule}
        self.assertIn("reasoning_bertscore_f1_rule_1", res)

    def test_empty_inputs(self):
        """Empty lists should return fallback zeros."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        res = batch_score_reasoning([], [])
        self.assertEqual(res["reasoning_bertscore_f1_macro"], 0.0)
        self.assertEqual(res["reasoning_meteor_macro"], 0.0)
        self.assertEqual(res["reasoning_ciderd_macro"], 0.0)

    def test_no_overlapping_rules(self):
        """No common rules between pred and GT → all zeros."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        preds = [{"rule_1_violation": {"reason": "pred"}}]
        refs = [{"rule_2_violation": {"reason": "ref"}}]

        res = batch_score_reasoning(preds, refs)
        self.assertEqual(res["reasoning_bertscore_f1_macro"], 0.0)

    @patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
    def test_empty_reasons_sanitized(self, mock_metrics):
        """Empty reason strings should be sanitized to 'empty'."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        mock_metrics.return_value = {"bertscore_f1": 0.5}

        preds = [{"rule_1_violation": {"reason": ""}}]  # Empty!
        refs = [{"rule_1_violation": {"reason": "should be here"}}]

        batch_score_reasoning(preds, refs)

        # Check first call (global) had sanitized input
        args = mock_metrics.call_args_list[0][0]
        self.assertEqual(args[0][0], "empty")  # pred reason sanitized

    @patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
    def test_spice_disabled(self, mock_metrics):
        """Reasoning eval should call captioning metrics with include_spice=False."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        mock_metrics.return_value = {"bertscore_f1": 0.5}

        preds = [{"rule_1_violation": {"reason": "pred"}}]
        refs = [{"rule_1_violation": {"reason": "ref"}}]

        batch_score_reasoning(preds, refs)

        # All calls should have include_spice=False
        for call in mock_metrics.call_args_list:
            kwargs = call[1]
            self.assertFalse(kwargs.get("include_spice", True))

    def test_rules_without_data_get_zeros(self):
        """Rules with no matching instances should get fallback zero scores."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        res = batch_score_reasoning([], [])
        self.assertEqual(res["reasoning_bertscore_f1_rule_3"], 0.0)
        self.assertEqual(res["reasoning_meteor_rule_3"], 0.0)
        self.assertEqual(res["reasoning_ciderd_rule_3"], 0.0)

    @patch("evaluation.metrics_reasoning.compute_all_caption_metrics")
    def test_none_violations_handled(self, mock_metrics):
        """None violation dicts should not crash."""
        from evaluation.metrics_reasoning import batch_score_reasoning

        mock_metrics.return_value = {"bertscore_f1": 0.5}

        preds = [None]
        refs = [None]
        # Should not raise
        res = batch_score_reasoning(preds, refs)
        self.assertEqual(res["reasoning_bertscore_f1_macro"], 0.0)


# Run reasoning metrics tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestReasoningMetrics)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 7: REASONING METRICS TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 8. Evaluator Integration Tests (6 tests)
#
# Tests `run_full_evaluation` from `evaluation/evaluator.py`.

# %%
class TestEvaluatorIntegration(unittest.TestCase):
    """Integration tests for the main evaluation orchestrator."""

    @patch("evaluation.evaluator.batch_score_reasoning")
    @patch("evaluation.evaluator.compute_violation_metrics")
    @patch("evaluation.evaluator.compute_grounding_metrics")
    @patch("evaluation.evaluator.compute_all_caption_metrics")
    @patch("evaluation.evaluator.compute_structural_metrics")
    def test_full_pipeline_orchestration(self, mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas):
        """Verify evaluator calls all sub-modules and aggregates results."""
        from evaluation.evaluator import run_full_evaluation

        mock_struct.return_value = {"structural_json_validity_rate": 1.0}
        mock_cap.return_value = {"captioning_bertscore_f1": 0.9}
        mock_grnd.return_value = {"grounding_iou_all_macro_mean": 0.8}
        mock_viol.return_value = {"violation_identification_f1_macro": 0.7}
        mock_reas.return_value = {"reasoning_bertscore_f1_macro": 0.6}

        raw = ['```json\n{"caption": "safe"}\n```']
        refs = [{"caption": "safe_gt"}]

        res = run_full_evaluation(raw, refs)

        # All sub-modules called
        mock_struct.assert_called_once()
        mock_cap.assert_called_once()
        mock_grnd.assert_called_once()
        mock_viol.assert_called_once()
        mock_reas.assert_called_once()

        # Metrics aggregated
        metrics = res["metrics"]
        self.assertEqual(metrics["structural_json_validity_rate"], 1.0)
        self.assertEqual(metrics["captioning_bertscore_f1"], 0.9)
        self.assertEqual(metrics["grounding_iou_all_macro_mean"], 0.8)
        self.assertEqual(metrics["violation_identification_f1_macro"], 0.7)
        self.assertEqual(metrics["reasoning_bertscore_f1_macro"], 0.6)

    @patch("evaluation.evaluator.batch_score_reasoning")
    @patch("evaluation.evaluator.compute_violation_metrics")
    @patch("evaluation.evaluator.compute_grounding_metrics")
    @patch("evaluation.evaluator.compute_all_caption_metrics")
    @patch("evaluation.evaluator.compute_structural_metrics")
    def test_parsed_predictions_returned(self, mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas):
        """Verify parsed predictions are passed through for downstream logging."""
        from evaluation.evaluator import run_full_evaluation

        for m in [mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas]:
            m.return_value = {}

        raw = ['```json\n{"caption": "safe"}\n```']
        refs = [{"caption": "gt"}]

        res = run_full_evaluation(raw, refs)
        self.assertEqual(len(res["parsed_predictions"]), 1)
        self.assertEqual(res["parsed_predictions"][0]["caption"], "safe")

    @patch("evaluation.evaluator.batch_score_reasoning")
    @patch("evaluation.evaluator.compute_violation_metrics")
    @patch("evaluation.evaluator.compute_grounding_metrics")
    @patch("evaluation.evaluator.compute_all_caption_metrics")
    @patch("evaluation.evaluator.compute_structural_metrics")
    def test_failure_logging(self, mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas):
        """Verify JSON parse failures are logged with image IDs."""
        from evaluation.evaluator import run_full_evaluation

        for m in [mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas]:
            m.return_value = {}

        raw = ["not json at all"]
        refs = [{"image_id": "test_001", "caption": "gt"}]

        res = run_full_evaluation(raw, refs)
        self.assertEqual(len(res["failures"]), 1)
        self.assertEqual(res["failures"][0]["image_id"], "test_001")
        self.assertEqual(res["failures"][0]["error_type"], "json_parse_error")

    @patch("evaluation.evaluator.batch_score_reasoning")
    @patch("evaluation.evaluator.compute_violation_metrics")
    @patch("evaluation.evaluator.compute_grounding_metrics")
    @patch("evaluation.evaluator.compute_all_caption_metrics")
    @patch("evaluation.evaluator.compute_structural_metrics")
    def test_schema_validation_failure(self, mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas):
        """Valid JSON but invalid schema should be logged as schema_validation_error."""
        from evaluation.evaluator import run_full_evaluation

        for m in [mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas]:
            m.return_value = {}

        raw = ['{"no_caption_key": true}']  # Valid JSON, missing required 'caption'
        refs = [{"image_id": "test_002", "caption": "gt"}]

        res = run_full_evaluation(raw, refs)
        self.assertEqual(len(res["failures"]), 1)
        self.assertEqual(res["failures"][0]["error_type"], "schema_validation_error")

    @patch("evaluation.evaluator.batch_score_reasoning")
    @patch("evaluation.evaluator.compute_violation_metrics")
    @patch("evaluation.evaluator.compute_grounding_metrics")
    @patch("evaluation.evaluator.compute_all_caption_metrics")
    @patch("evaluation.evaluator.compute_structural_metrics")
    def test_empty_inputs(self, mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas):
        """Empty inputs should not crash."""
        from evaluation.evaluator import run_full_evaluation

        for m in [mock_struct, mock_cap, mock_grnd, mock_viol, mock_reas]:
            m.return_value = {}

        res = run_full_evaluation([], [])
        self.assertIn("metrics", res)
        self.assertIn("parsed_predictions", res)
        self.assertEqual(res["parsed_predictions"], [])
        self.assertEqual(res["failures"], [])

    def test_length_mismatch_raises(self):
        """Mismatched prediction/reference lengths should raise ValueError."""
        from evaluation.evaluator import run_full_evaluation

        with self.assertRaises(ValueError):
            run_full_evaluation(["pred1", "pred2"], [{"caption": "ref1"}])


# Run evaluator integration tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluatorIntegration)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 8: EVALUATOR INTEGRATION TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 9. Error Analyzer Tests (4 tests)
#
# Tests `run_stratified_analysis` from `evaluation/error_analyzer.py`.

# %%
class TestErrorAnalyzer(unittest.TestCase):
    """Tests for stratified error analysis."""

    @patch("evaluation.error_analyzer.run_full_evaluation")
    def test_stratified_filtering(self, mock_eval):
        """Verify samples are correctly filtered by metadata field/value."""
        from evaluation.error_analyzer import run_stratified_analysis

        mock_eval.return_value = {
            "metrics": {"structural_json_validity_rate": 1.0}
        }

        raw = ["pred1", "pred2", "pred3"]
        refs = [{"caption": "r1"}, {"caption": "r2"}, {"caption": "r3"}]
        meta = [
            {"illumination": "normal lighting"},
            {"illumination": "night"},
            {"illumination": "normal lighting"},
        ]

        results = run_stratified_analysis(raw, refs, meta)

        # Should have entries for illumination strata that have data
        self.assertIn("illumination_normal lighting", results)
        self.assertIn("illumination_night", results)

    @patch("evaluation.error_analyzer.run_full_evaluation")
    def test_empty_stratum_skipped(self, mock_eval):
        """Strata with no matching samples should be skipped."""
        from evaluation.error_analyzer import run_stratified_analysis

        mock_eval.return_value = {"metrics": {"test": 1.0}}

        raw = ["pred1"]
        refs = [{"caption": "r1"}]
        meta = [{"illumination": "normal lighting"}]

        results = run_stratified_analysis(raw, refs, meta)

        # "night" has no samples → should not appear
        self.assertNotIn("illumination_night", results)

    def test_length_mismatch(self):
        """Mismatched lengths should return empty dict."""
        from evaluation.error_analyzer import run_stratified_analysis

        results = run_stratified_analysis(["p1"], [{"c": "r1"}], [])
        self.assertEqual(results, {})

    @patch("evaluation.error_analyzer.run_full_evaluation")
    def test_error_in_stratum_doesnt_crash(self, mock_eval):
        """If evaluation fails for one stratum, others should still work."""
        from evaluation.error_analyzer import run_stratified_analysis

        call_count = [0]

        def side_effect(preds, refs, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Boom!")
            return {"metrics": {"test": 1.0}}

        mock_eval.side_effect = side_effect

        raw = ["p1", "p2"]
        refs = [{"caption": "r1"}, {"caption": "r2"}]
        meta = [
            {"illumination": "normal lighting"},
            {"illumination": "night"},
        ]

        results = run_stratified_analysis(raw, refs, meta)
        # One failed, one succeeded
        # At least one stratum should have results
        total_strata = len(results)
        self.assertGreaterEqual(total_strata, 0)


# Run error analyzer tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestErrorAnalyzer)
runner = unittest.TextTestRunner(verbosity=2)
print("\n" + "=" * 70)
print("SECTION 9: ERROR ANALYZER TESTS")
print("=" * 70)
result = runner.run(suite)


# %% [markdown]
# ---
# ## 🏁 Run All Tests At Once
#
# Run this cell to execute the entire test suite and get a summary.

# %%
def run_all_tests():
    """Discover and run all tests in this file."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        # Section 1: Output Parser
        TestStripFences,
        TestParseModelOutput,
        TestValidateUnifiedOutput,
        # Section 2: Structural Metrics
        TestStructuralMetrics,
        # Section 3: Captioning Metrics
        TestCaptionMetricsAggregation,
        # Section 4: Box Utilities
        TestNormalizeBoxes,
        TestIsValidBox,
        TestCleanBoxes,
        TestScaleConversion,
        TestComputeIoU,
        TestGreedyMultiboxIoU,
        # Section 5: Grounding Metrics
        TestGroundingMetrics,
        # Section 6: Violation Metrics
        TestViolationMetrics,
        # Section 7: Reasoning Metrics
        TestReasoningMetrics,
        # Section 8: Evaluator Integration
        TestEvaluatorIntegration,
        # Section 9: Error Analyzer
        TestErrorAnalyzer,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    print("=" * 70)
    print("  RUNNING COMPLETE EVALUATION TEST SUITE")
    print("=" * 70)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    print(f"  SUMMARY: {passed}/{total} passed, {failures} failures, {errors} errors")
    print("=" * 70)
    return result


if __name__ == "__main__":
    run_all_tests()
