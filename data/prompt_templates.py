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
    "3. SAFETY VIOLATIONS — check the image against these four rules and list "
    "EVERY rule that is violated (there may be zero, one, or multiple):\n"
    "   Rule 1: Use of basic PPE when on foot (hard hats, appropriate clothing, "
    "closed-toe shoes, high-visibility vests at night, face protection for "
    "cutting/welding/grinding/drilling).\n"
    "   Rule 2: Use of a safety harness when working from a height of >3 meters "
    "with unprotected edges.\n"
    "   Rule 3: Edge protection/warning for underground projects >3 meters deep "
    "with steep retaining walls.\n"
    "   Rule 4: Worker appearing in an excavator's blind spot or operation radius.\n"
    "   For each violation, give the rule id, 1-2 sentences of reasoning, and "
    "bounding box(es) for the violator(s).\n\n"
    "Respond with a ```json code block in this exact JSON format:\n"
    '{"caption":"your detailed caption",'
    '"detected_objects":{'
    '"excavator":[[xmin,ymin,xmax,ymax],...],'
    '"rebar":[[xmin,ymin,xmax,ymax],...],'
    '"worker_with_white_hard_hat":[[xmin,ymin,xmax,ymax],...]'
    "},"
    '"safety_violations":['
    '{"rule_id":"rule_1","reason":"1-2 sentences","bounding_boxes":[[xmin,ymin,xmax,ymax],...]}'
    "]}"
)

# ---------------------------------------------------------------------------
# Prompt Registry — maps config prompt_key to actual template
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT,
}