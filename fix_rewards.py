import re
import os

reward_files = [
    "rewards/reward_caption.py",
    "rewards/reward_grounding.py",
    "rewards/reward_reasoning.py",
    "rewards/reward_violation_grounding.py",
    "rewards/reward_violation_id.py"
]

for file_path in reward_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace _strict_parse with _strict_parse_for_task
    content = content.replace("from rewards.reward_utils import _strict_parse,", "from rewards.reward_utils import _strict_parse_for_task,")
    content = content.replace("from rewards.reward_utils import _strict_parse ", "from rewards.reward_utils import _strict_parse_for_task ")
    
    if "is_batched" in content:
        # For batched functions
        content = re.sub(r'parsed = _strict_parse\(completion\)', r'task = kwargs.get("task", "unified")\n        parsed = _strict_parse_for_task(completion, task=task)', content)
    else:
        # For non-batched functions
        content = re.sub(r'parsed = _strict_parse\(completion\)', r'task = kwargs.get("task", "unified")\n    parsed = _strict_parse_for_task(completion, task=task)', content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
print("Updated reward components to use task context.")
