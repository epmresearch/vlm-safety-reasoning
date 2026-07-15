"""
All prompt text lives here as named constants.
Never hardcode prompt strings anywhere else in the codebase.

The unified inspection prompt asks the model to perform caption + object
detection + safety violation analysis in a single pass and respond with
a minimized JSON code block.
"""


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert AI construction safety inspector. Analyze the "
    "construction site image and output a single JSON code block containing "
    "your full inspection. You must strictly adhere to the requested JSON schema. "
    "Absolutely no conversational text, preambles, or explanations outside the "
    "```json ... ``` fences."
)

# ---------------------------------------------------------------------------
# Unified Inspection Prompt (user turn)
# ---------------------------------------------------------------------------
UNIFIED_INSPECTION_PROMPT = (
    "Perform a full site inspection of this construction image. Your response "
    "must include ALL of the following in a single JSON object:\n\n"
    "1. CAPTION — a detailed description covering both foreground and background "
    "elements (workers, equipment, PPE, material stockpiles, site conditions).\n\n"
    "2. DETECTED OBJECTS — bounding boxes for every instance of:\n"
    "   - excavator\n"
    "   - rebar\n"
    "   - worker_with_white_hard_hat\n"
    "   If an object class is absent, return an empty array []. All bounding box "
    "coordinates must be in [xmin, ymin, xmax, ymax] format scaled from 0 to 1000.\n\n"
    "3. SAFETY VIOLATIONS — evaluate the image against ALL four rules. You must "
    "include an entry for EVERY rule:\n"
    "   Rule 1: Use of basic PPE when on foot (hard hats, appropriate clothing, "
    "closed-toe shoes, high-visibility vests at night, face protection for "
    "cutting/welding/grinding/drilling).\n"
    "   Rule 2: Use of a safety harness when working from a height of >3 meters "
    "with unprotected edges.\n"
    "   Rule 3: Edge protection/warning for underground projects >3 meters deep "
    "with steep retaining walls.\n"
    "   Rule 4: Worker appearing in an excavator's blind spot or operation radius.\n"
    "   For each rule, if it is violated, output an object with 'reason' (1-2 "
    "sentences) and 'bounding_box' (list of boxes). If a rule is NOT violated, "
    "the value must be strictly null (not a string, not an empty object).\n\n"
    "Respond with a ```json code block in this exact JSON format:\n"
    '{"caption":"your detailed caption",\n'
    '"rule_1_violation":{"bounding_box":[[xmin,ymin,xmax,ymax],...], "reason":"1-2 sentences"},\n'
    '"rule_2_violation":null,\n'
    '"rule_3_violation":null,\n'
    '"rule_4_violation":null,\n'
    '"excavator":[[xmin,ymin,xmax,ymax],...],\n'
    '"rebar":[],\n'
    '"worker_with_white_hard_hat":[[xmin,ymin,xmax,ymax],...]\n'
    "}"
)


# ---------------------------------------------------------------------------
# Prompt Registry — maps config prompt_key to actual template
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT,
}