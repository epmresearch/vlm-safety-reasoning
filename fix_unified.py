with open("rewards/unified_reward.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("from rewards.reward_utils import _strict_parse, _has_repetition_pathology", "from rewards.reward_utils import _strict_parse, _strict_parse_for_task, _has_repetition_pathology")

content = content.replace("def compute_reward(\n    prediction: str,\n    ground_truth: dict,\n    weights: Optional[Dict[str, float]] = None,\n) -> float:", "def compute_reward(\n    prediction: str,\n    ground_truth: dict,\n    weights: Optional[Dict[str, float]] = None,\n    task: str = \"unified\",\n) -> float:")
content = content.replace("def compute_reward_with_breakdown(\n    prediction: str,\n    ground_truth: dict,\n    weights: Optional[Dict[str, float]] = None,\n) -> Dict[str, float]:", "def compute_reward_with_breakdown(\n    prediction: str,\n    ground_truth: dict,\n    weights: Optional[Dict[str, float]] = None,\n    task: str = \"unified\",\n) -> Dict[str, float]:")

content = content.replace("parsed = _strict_parse(prediction)", "parsed = _strict_parse_for_task(prediction, task=task)")
content = content.replace("score = reward_fn([prediction], [ground_truth])[0]", "score = reward_fn([prediction], [ground_truth], task=task)[0]")
content = content.replace("score = reward_fn(prediction, ground_truth)", "score = reward_fn(prediction, ground_truth, task=task)")

with open("rewards/unified_reward.py", "w", encoding="utf-8") as f:
    f.write(content)
