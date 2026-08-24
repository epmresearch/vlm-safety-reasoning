"""
Tests for GRPO configuration loading and merging.
"""
import pytest
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


class TestGrpoYaml:
    def _load(self, filename):
        with open(os.path.join(CONFIG_DIR, filename)) as f:
            return yaml.safe_load(f)

    def test_required_keys_present(self):
        cfg = self._load("grpo.yaml")
        required = {
            "num_generations", "max_prompt_length", "max_completion_length",
            "learning_rate", "per_device_train_batch_size",
            "gradient_accumulation_steps", "num_train_epochs",
            "logging_steps", "save_steps", "beta",
        }
        missing = required - set(cfg.keys())
        assert not missing, f"grpo.yaml missing keys: {missing}"

    def test_max_prompt_length_exceeds_verified_max(self):
        cfg = self._load("grpo.yaml")
        VERIFIED_MAX = 1519
        assert cfg["max_prompt_length"] > VERIFIED_MAX, (
            f"max_prompt_length={cfg['max_prompt_length']} <= {VERIFIED_MAX}. "
            "Prompts will be truncated."
        )

    def test_max_seq_length_covers_prompt_plus_completion(self):
        cfg = self._load("grpo.yaml")
        assert "max_seq_length" in cfg, "max_seq_length must be in grpo.yaml"
        min_needed = cfg["max_prompt_length"] + cfg["max_completion_length"] + 200
        assert cfg["max_seq_length"] >= min_needed

    def test_num_generations_positive(self):
        cfg = self._load("grpo.yaml")
        assert cfg["num_generations"] > 0

    def test_beta_reasonable_range(self):
        cfg = self._load("grpo.yaml")
        assert 0.0 < cfg["beta"] < 1.0, "KL penalty beta should be between 0 and 1"

    def test_learning_rate_small(self):
        cfg = self._load("grpo.yaml")
        assert cfg["learning_rate"] <= 1e-5, "GRPO LR should be small (≤1e-5)"


class TestConfigMerge:
    def test_grpo_overrides_model_registry_batch_size(self):
        """Top-level grpo.yaml per_device_train_batch_size should win over nested registry value."""
        from core.config import load_config

        cfg = load_config(task="unified", training_kind="grpo")
        # grpo.yaml has per_device_train_batch_size: 16
        # model_registry.yaml has per_device_train_batch_size: 32 NESTED under models.2b
        # Top-level from grpo.yaml should win
        assert cfg["per_device_train_batch_size"] == 16, (
            "grpo.yaml per_device_train_batch_size should be 16, "
            "but model registry nested value may be leaking through"
        )

    def test_task_config_keys_present_after_merge(self):
        from core.config import load_config
        cfg = load_config(task="unified", training_kind="grpo")
        assert "task_name" in cfg
        assert cfg["task_name"] == "unified"

    def test_base_config_keys_present(self):
        from core.config import load_config
        cfg = load_config(task="unified", training_kind="grpo")
        assert "drive_root" in cfg
        assert "seed" in cfg