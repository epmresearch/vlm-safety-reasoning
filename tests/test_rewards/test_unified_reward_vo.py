import pytest
from rewards.unified_reward import get_reward_funcs_for_task

def test_get_reward_funcs_for_task_vo():
    # In configs/tasks/violations_only.yaml, we specified 4 components.
    # The config loader is dynamically hit, so let's check it.
    funcs, weights = get_reward_funcs_for_task(task="violations_only")
    
    assert len(funcs) == 4
    assert len(weights) == 4
    
    names = [f.__name__ for f in funcs]
    assert "reward_format" in names
    assert "reward_violation_id" in names
    assert "reward_violation_grounding" in names
    assert "reward_reasoning" in names
    
    assert "reward_caption" not in names
    assert "reward_grounding" not in names
    
    # Check weights sum to 1.0 based on the yaml config
    assert sum(weights) == pytest.approx(1.0)
    
    # Check unified fallback
    funcs_uni, weights_uni = get_reward_funcs_for_task(task="unified")
    assert len(funcs_uni) == 6
    assert len(weights_uni) == 6
