"""
Project 1 is now a SINGLE unified task: one prompt, one JSON response per
image, containing caption + detected objects + safety violations together.
No per-task prompting, no per-class grounding prompts.
"""
UNIFIED_TASK_NAME = "full_unified"
GROUNDING_CLASSES = ["excavators", "rebar", "white_hard_hat_workers"]
RULES = ["rule_1", "rule_2", "rule_3", "rule_4"]