"""
Reward functions for GRPO training.

Individual reward modules:
    - json_validity: Valid JSON parse check (0 or 1)
    - grounding_iou: Multi-box greedy IoU matching
    - rule_violation_accuracy: Multi-label F1 over rule IDs
    - caption_quality: Sentence-transformer cosine similarity

Composite:
    - unified_reward: Weighted combination of all rewards
"""
