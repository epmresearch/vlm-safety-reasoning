"""
All prompt text lives here as named constants.
Never hardcode prompt strings anywhere else in the codebase.

The unified inspection prompt asks the model to perform caption + object
detection + safety violation analysis in a single pass and respond with
a minimized JSON code block.
"""

SYSTEM_PROMPT = "You are an expert AI construction safety inspector."

UNIFIED_INSPECTION_PROMPT = (
    "Analyze the construction safety in this image. Output strictly a single JSON code block.\n\n"
    "1. For the 'caption' key: Provide a detailed description of foreground, background, workers, equipment, and conditions.\n"
    "2. For the object keys ('excavator', 'rebar', 'worker_with_white_hard_hat'): Provide a list of bounding boxes [[xmin, ymin, xmax, ymax]] scaled from 0 to 1000. Return [] if absent.\n"
    "3. For the safety violation keys: Evaluate against the following 4 rules:\n"
    "   - Rule 1: Use of basic PPE (hard hats, proper clothing, closed-toe shoes, vests).\n"
    "   - Rule 2: Use of safety harness when working at height >3m.\n"
    "   - Rule 3: Edge protection for underground projects >3m deep.\n"
    "   - Rule 4: Worker in excavator blind spot or operation radius.\n"
    "   If violated, output {'reason':'...', 'bounding_box':[[xmin, ymin, xmax, ymax]]}. If NOT violated, output null.\n\n"
    "Respond exactly in this JSON format:\n"
    "```json\n"
    '{"caption":"detailed description","rule_1_violation":{"bounding_box":[[xmin, ymin, xmax, ymax]],"reason":"..."},"rule_2_violation":null,"rule_3_violation":null,"rule_4_violation":null,"excavator":[[xmin, ymin, xmax, ymax]],"rebar":[],"worker_with_white_hard_hat":[[xmin, ymin, xmax, ymax]]}'
    "\n```"
)

VIOLATIONS_ONLY_PROMPT = (
    "Analyze the construction safety in this image. Output strictly a single JSON code block.\n\n"
    "Evaluate against the following 4 safety rules:\n"
    "   - Rule 1: Use of basic PPE (hard hats, proper clothing, closed-toe shoes, vests).\n"
    "   - Rule 2: Use of safety harness when working at height >3m.\n"
    "   - Rule 3: Edge protection for underground projects >3m deep.\n"
    "   - Rule 4: Worker in excavator blind spot or operation radius.\n"
    "   If violated, output {'reason':'...', 'bounding_box':[[xmin, ymin, xmax, ymax]]} "
    "with bounding boxes scaled from 0 to 1000. If NOT violated, output null.\n\n"
    "Respond exactly in this JSON format:\n"
    "```json\n"
    '{"rule_1_violation":{"bounding_box":[[xmin, ymin, xmax, ymax]],"reason":"..."},'
    '"rule_2_violation":null,"rule_3_violation":null,"rule_4_violation":null}'
    "\n```"
)

OBJECT_ONLY_PROMPT = (
    "Detect the specified objects in this construction site image. Output strictly a "
    "single JSON code block.\n\n"
    "For each of the three object keys ('excavator', 'rebar', "
    "'worker_with_white_hard_hat'), provide a list of bounding boxes "
    "[[xmin, ymin, xmax, ymax]] scaled from 0 to 1000. Return [] for any class that is "
    "absent from the image.\n"
    "   - excavator: any excavator or similar tracked digging machine.\n"
    "   - rebar: exposed steel reinforcing bar, whether bundled, laid out, or "
    "installed.\n"
    "   - worker_with_white_hard_hat: a person wearing a white hard hat.\n\n"
    "Do not include a caption and do not include any safety assessment.\n\n"
    "Respond exactly in this JSON format:\n"
    "```json\n"
    '{"excavator":[[xmin, ymin, xmax, ymax]],"rebar":[],'
    '"worker_with_white_hard_hat":[[xmin, ymin, xmax, ymax]]}'
    "\n```"
)

# caption_only is the one task whose completion is NOT JSON. The caption is the entire
# output, so wrapping it in a JSON object would add a formatting confound to the exact
# quantity being measured. The prompt therefore forbids JSON explicitly, and
# evaluation/output_parser.py::parse_output_for_task wraps the raw prose into
# {"caption": ...} so the rest of the stack stays dict-shaped.
CAPTION_ONLY_PROMPT = (
    "Describe this construction site image in detail. Cover the foreground, the "
    "background, any workers and what they are wearing, equipment and machinery, and "
    "the site conditions.\n\n"
    "Respond with the description only, as a single plain-text paragraph. Do not "
    "output JSON, do not use a code fence, do not use key/value pairs, do not add a "
    "heading or a label, and do not include bounding boxes. Start directly with the "
    "description."
)

PROMPT_REGISTRY = {
    "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT,
    "VIOLATIONS_ONLY_PROMPT": VIOLATIONS_ONLY_PROMPT,
    "OBJECT_ONLY_PROMPT": OBJECT_ONLY_PROMPT,
    "CAPTION_ONLY_PROMPT": CAPTION_ONLY_PROMPT,
}


def get_prompt_for_task(task: str) -> str:
    """Returns the user prompt string for the given task.

    Looks up the task's ``prompt_key`` in its YAML config and resolves it
    through the prompt registry.

    Args:
        task: Task name (e.g., 'unified', 'violations_only').

    Returns:
        The prompt string to use as the user message.

    Raises:
        ValueError: If the prompt_key is not found in the registry.
    """
    from core.config import load_task_config
    task_cfg = load_task_config(task)
    prompt_key = task_cfg["prompt_key"]
    if prompt_key not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt_key: {prompt_key!r}. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[prompt_key]