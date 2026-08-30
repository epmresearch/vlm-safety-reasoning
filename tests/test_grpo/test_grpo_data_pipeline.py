"""
Tests for the GRPO data pipeline:
- build_ground_truth_dict
- to_grpo_prompt
- build_grpo_dataset
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from data.preprocessor import build_ground_truth_dict, to_grpo_prompt, build_grpo_dataset


def _make_raw_sample(image_id="0001234", with_violation=False, with_objects=False):
    sample = {
        "image_id": image_id,
        "image_caption": "A construction site with an excavator.",
        "illumination": "normal lighting",
        "camera_distance": "mid distance",
        "view": "elevation view",
        "quality_of_info": "rich",
        "rule_1_violation": {
            "bounding_box": [[0.1, 0.1, 0.5, 0.5]],
            "reason": "Worker not wearing PPE.",
        } if with_violation else None,
        "rule_2_violation": None,
        "rule_3_violation": None,
        "rule_4_violation": None,
        "excavator": [[0.1, 0.05, 0.8, 0.7]] if with_objects else [],
        "rebar": [],
        "worker_with_white_hard_hat": [[0.85, 0.2, 0.95, 0.6]] if with_objects else [],
    }
    return sample


def _make_pil_image():
    return PILImage.new("RGB", (100, 100), color=(128, 128, 128))


class TestBuildGroundTruthDict:
    def test_no_violation_structure(self):
        raw = _make_raw_sample()
        gt = build_ground_truth_dict(raw)
        assert gt["rule_1_violation"] is None
        assert gt["rule_2_violation"] is None
        assert gt["caption"] == "A construction site with an excavator."

    def test_violation_structure(self):
        raw = _make_raw_sample(with_violation=True)
        gt = build_ground_truth_dict(raw)
        v = gt["rule_1_violation"]
        assert v is not None
        assert isinstance(v, dict)
        assert "bounding_box" in v
        assert "reason" in v
        assert v["reason"] == "Worker not wearing PPE."

    def test_boxes_in_01_scale(self):
        """GT boxes must remain in [0,1] scale (not scaled to [0,1000])."""
        raw = _make_raw_sample(with_objects=True)
        gt = build_ground_truth_dict(raw)
        for box in gt["excavator"]:
            assert all(0.0 <= c <= 1.0 for c in box), \
                f"GT excavator box {box} is not in [0,1] scale"

    def test_violation_boxes_in_01_scale(self):
        raw = _make_raw_sample(with_violation=True)
        gt = build_ground_truth_dict(raw)
        for box in gt["rule_1_violation"]["bounding_box"]:
            assert all(0.0 <= c <= 1.0 for c in box)

    def test_all_required_keys_present(self):
        raw = _make_raw_sample()
        gt = build_ground_truth_dict(raw)
        required = {
            "caption", "rule_1_violation", "rule_2_violation",
            "rule_3_violation", "rule_4_violation",
            "excavator", "rebar", "worker_with_white_hard_hat",
        }
        assert required.issubset(set(gt.keys()))

    def test_empty_violation_boxes_normalized(self):
        raw = {**_make_raw_sample(), "rule_1_violation": {"bounding_box": [], "reason": "some reason"}}
        gt = build_ground_truth_dict(raw)
        assert gt["rule_1_violation"]["bounding_box"] == []

    def test_flat_single_box_normalized(self):
        """A flat [x,y,x,y] single box should be wrapped in a list."""
        raw = {**_make_raw_sample()}
        raw["rule_1_violation"] = {
            "bounding_box": [0.1, 0.1, 0.5, 0.5],  # flat single box
            "reason": "test",
        }
        gt = build_ground_truth_dict(raw)
        v = gt["rule_1_violation"]
        assert isinstance(v["bounding_box"], list)
        # Should be [[0.1, 0.1, 0.5, 0.5]] after normalization
        assert all(isinstance(b, list) for b in v["bounding_box"])


class TestToGrpoPrompt:
    def test_returns_required_keys(self):
        raw = _make_raw_sample()
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        assert "prompt" in result
        assert "ground_truth" in result
        assert "image_id" in result

    def test_image_id_passed_through(self):
        raw = _make_raw_sample(image_id="9999999")
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        assert result["image_id"] == "9999999"

    def test_prompt_is_list_of_messages(self):
        raw = _make_raw_sample()
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        prompt = result["prompt"]
        assert isinstance(prompt, list)
        assert len(prompt) == 2  # system + user
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"

    def test_user_message_contains_image(self):
        raw = _make_raw_sample()
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        user_content = result["prompt"][1]["content"]
        image_parts = [c for c in user_content if c.get("type") == "image"]
        assert len(image_parts) == 1
        assert "image" not in image_parts[0]
        assert "image" in result
        assert isinstance(result["image"], PILImage.Image)

    def test_ground_truth_is_dict(self):
        raw = _make_raw_sample()
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        gt = result["ground_truth"]
        assert isinstance(gt, str)
        gt_dict = json.loads(gt)
        assert "caption" in gt_dict

    def test_ground_truth_boxes_in_01_scale(self):
        raw = _make_raw_sample(with_objects=True)
        pil = _make_pil_image()
        result = to_grpo_prompt(raw, pil)
        gt_dict = json.loads(result["ground_truth"])
        for box in gt_dict["excavator"]:
            assert all(0.0 <= c <= 1.0 for c in box)


class TestBuildGrpoDataset:
    def _make_mock_hf_dataset(self, n=5, with_violation=False):
        """Create a list that mimics an HF Dataset."""
        samples = []
        for i in range(n):
            raw = _make_raw_sample(image_id=f"{i:07d}", with_violation=with_violation)
            raw["image"] = _make_pil_image()
            samples.append(raw)
        return samples

    def test_output_length_matches_input(self):
        ds = self._make_mock_hf_dataset(n=10)
        result = build_grpo_dataset(ds)
        assert len(result) == 10

    def test_max_samples_respected(self):
        ds = self._make_mock_hf_dataset(n=20)
        result = build_grpo_dataset(ds, max_samples=5)
        assert len(result) == 5

    def test_each_sample_has_required_keys(self):
        ds = self._make_mock_hf_dataset(n=3)
        result = build_grpo_dataset(ds)
        for item in result:
            assert "prompt" in item
            assert "ground_truth" in item
            assert "image_id" in item

    def test_ground_truth_serializable_to_json(self):
        """ground_truth must be JSON-serializable (for Dataset storage)."""
        ds = self._make_mock_hf_dataset(n=3, with_violation=True)
        result = build_grpo_dataset(ds)
        for item in result:
            try:
                json.dumps(item["ground_truth"])
            except (TypeError, ValueError) as e:
                pytest.fail(f"ground_truth is not JSON-serializable: {e}")