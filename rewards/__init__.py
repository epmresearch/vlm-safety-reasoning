"""
Reward functions for GRPO training.

Individual reward modules (new, 6-component design):
    - reward_format: Schema-validated JSON parse check (0 or 1)
    - reward_caption: Semantic + lexical caption similarity with length calibration
    - reward_grounding: Mask-union IoU for object detection with TN fix
    - reward_violation_id: F-beta (β=2) violation identification
    - reward_violation_grounding: TP-conditioned violation box IoU
    - reward_reasoning: TP-conditioned reasoning text quality

Shared utilities:
    - reward_utils: Strict parsing, embedding, ngram F1, repetition detection

Legacy modules (kept for backwards compatibility, not used in new pipeline):
    - json_validity, caption_quality, rule_violation_accuracy, grounding_iou
    - unified_reward (legacy weighted combination)
"""
