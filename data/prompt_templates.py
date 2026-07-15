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
    "You are an expert AI construction safety inspector. Analyze the provided "
    "construction site image and perform a full inspection in a single pass. "
    "Always respond with a JSON code block (```json ... ```) containing your "
    "analysis. Do not include any text outside the code block."
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
    "   If a class has no instances, return an empty list for it.\n\n"
    "3. SAFETY VIOLATIONS — evaluate the image against ALL four rules. You must include an entry for EVERY rule:\n"
    "   Rule 1: Use of basic PPE when on foot (hard hats, appropriate clothing, "
    "closed-toe shoes, high-visibility vests at night, face protection for "
    "cutting/welding/grinding/drilling).\n"
    "   Rule 2: Use of a safety harness when working from a height of >3 meters "
    "with unprotected edges.\n"
    "   Rule 3: Edge protection/warning for underground projects >3 meters deep "
    "with steep retaining walls.\n"
    "   Rule 4: Worker appearing in an excavator's blind spot or operation radius.\n"
    "   For each rule, if it is violated, output an object with 'reason' (1-2 sentences) and "
    "'bounding_box' (list of boxes). If a rule is NOT violated, output null for that rule.\n\n"
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