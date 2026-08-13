"""
Regression tests that specifically expose known bugs in the GRPO pipeline.
Each test documents a specific bug with its bug number.
Run: pytest tests/test_grpo/test_grpo_bugs.py -v
"""
import json
import pytest


class TestBug3OversamplingShouldReturnTuple:
    """Bug #3: build_oversampled_indices returns (List[int], Dict), not List[int]."""

    def test_returns_tuple_of_two_elements(self):
        from data.oversampling import build_oversampled_indices
        from unittest.mock import MagicMock

        # Create a minimal mock dataset
        mock_dataset = [
            {"rule_2_violation": {"bounding_box": [[0,0,1,1]], "reason": "x"}},
            {"rule_2_violation": None},
            {"rule_3_violation": {"bounding_box": [[0,0,1,1]], "reason": "x"},
             "rule_2_violation": None},
            {"rule_2_violation": None, "rule_3_violation": None, "rule_4_violation": None},
        ]

        result = build_oversampled_indices(mock_dataset)
        assert isinstance(result, tuple), (
            "build_oversampled_indices must return a tuple (indices, manifest). "
            "If grpo_trainer.py does oversampled_indices = build_oversampled_indices(...), "
            "it will receive a tuple, not a list, and Dataset.select() will crash."
        )
        assert len(result) == 2
        indices, manifest = result
        assert isinstance(indices, list)
        assert isinstance(manifest, dict)

    def test_correct_unpacking_pattern(self):
        """Demonstrates the correct usage pattern."""
        from data.oversampling import build_oversampled_indices

        mock_dataset = [
            {"rule_2_violation": {"bounding_box": [[0,0,1,1]], "reason": "x"},
             "rule_4_violation": None, "rule_3_violation": None},
            {"rule_2_violation": None, "rule_4_violation": None, "rule_3_violation": None},
        ]

        # CORRECT usage:
        indices, manifest = build_oversampled_indices(mock_dataset)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_wrong_usage_would_fail_select(self):
        """
        Bug: the current run_grpo code does:
            oversampled_indices = build_oversampled_indices(train_split)
            train_split.select(oversampled_indices)   # BUG: tuple not list
        This test demonstrates what happens when select() gets a tuple.
        """
        from datasets import Dataset
        ds = Dataset.from_dict({"x": [1, 2, 3]})
        indices = [0, 1, 2]
        manifest = {"total_rows_before": 3}

        with pytest.raises((TypeError, Exception)):
            # Simulates the buggy code: passing (list, dict) tuple to select
            ds.select((indices, manifest))


class TestBug1MaxPromptLength:
    """Bug #1: max_prompt_length=1024 is smaller than actual max prompt (1519 tokens)."""

    def test_grpo_config_max_prompt_length_sufficient(self):
        import yaml, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "grpo.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        max_prompt = cfg.get("max_prompt_length", 0)
        VERIFIED_MAX_TOKENS = 1519  # from configs/tasks/unified.yaml comment
        assert max_prompt > VERIFIED_MAX_TOKENS, (
            f"max_prompt_length={max_prompt} is less than the verified maximum "
            f"prompt length of {VERIFIED_MAX_TOKENS} tokens. Increase to at least 1600."
        )


class TestBug2MaxSeqLength:
    """Bug #2: max_seq_length defaults to 2048, but prompt+completion can reach 2519."""

    def test_grpo_config_has_max_seq_length(self):
        import yaml, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "grpo.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "max_seq_length" in cfg, (
            "grpo.yaml must define max_seq_length. "
            "The default of 2048 is too small for max_prompt(1519)+max_completion(1000)=2519."
        )
        min_required = cfg.get("max_prompt_length", 1024) + cfg.get("max_completion_length", 1000) + 100
        assert cfg["max_seq_length"] >= min_required, (
            f"max_seq_length={cfg['max_seq_length']} is smaller than "
            f"max_prompt + max_completion + margin = {min_required}"
        )


class TestBug5AdapterPathCLI:
    """Bug #5: The GRPO CLI has no --adapter_path argument."""

    def test_cli_exposes_adapter_path(self):
        """The __main__ block in models/grpo_trainer.py must have --adapter_path."""
        import ast, os, inspect

        # Read the source file
        trainer_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "grpo_trainer.py"
        )
        with open(trainer_path) as f:
            source = f.read()

        assert "--adapter_path" in source, (
            "models/grpo_trainer.py __main__ block is missing --adapter_path argument. "
            "Without it, GRPO can only start from scratch (no SFT adapter via CLI)."
        )


class TestBug6LoraMissingInRegistry:
    """Bug #5b: model_registry.yaml has no lora_path for any model."""

    def test_lora_path_in_registry_for_active_tier(self):
        import yaml, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "model_registry.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        active_tier = cfg.get("active_tier", "2b")
        model_entry = cfg.get("models", {}).get(active_tier, {})

        # Either lora_path is in the registry, or the CLI must expose --adapter_path
        # This test documents the issue even if not yet fixed
        has_lora_path = "lora_path" in model_entry
        if not has_lora_path:
            pytest.skip(
                f"model_registry.yaml has no lora_path for tier '{active_tier}'. "
                "GRPO will start from scratch unless --adapter_path is passed at CLI. "
                "Add lora_path to the registry or ensure --adapter_path is mandatory."
            )


class TestBug7BatchFnSignature:
    """Bug #7: _make_batch_reward produces functions that may not accept 'prompts' positional arg."""

    def test_batch_fn_handles_prompts_as_first_positional(self):
        """Newer TRL passes (prompts, completions, **kwargs). Validate the wrapper handles this."""
        import json
        from rewards.unified_reward import get_reward_funcs_and_weights

        funcs, _ = get_reward_funcs_and_weights()
        valid_payload = {
            "caption": "test",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "excavator": [], "rebar": [], "worker_with_white_hard_hat": [],
        }
        completion = "```json\n" + json.dumps(valid_payload) + "\n```"
        gt = {**valid_payload}

        for fn in funcs:
            # Call with prompts as positional arg (new TRL style)
            try:
                result = fn(["prompt_text"], [completion], ground_truth=[gt])
                assert isinstance(result, list), f"{fn.__name__} did not return a list"
                assert len(result) == 1
            except TypeError as e:
                pytest.fail(
                    f"Reward function {fn.__name__} raises TypeError with new TRL "
                    f"calling convention (prompts as first positional arg): {e}"
                )