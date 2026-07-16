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
SYSTEM_PROMPT = "You are an expert AI construction safety inspector."


# ---------------------------------------------------------------------------
# Unified Inspection Prompt (user turn)
# ---------------------------------------------------------------------------
UNIFIED_INSPECTION_PROMPT = (
    "Perform a construction safety inspection on this image. Output strictly a single JSON code block. Do not include any conversational text.\n\n"
    "1. CAPTION: A detailed description of foreground, background, workers, equipment, and conditions.\n"
    "2. SPATIAL GROUNDING: Flat arrays [xmin, ymin, xmax, ymax] scaled from 0 to 1000 for every 'excavator', 'rebar', and 'worker_with_white_hard_hat'. Return [] if absent.\n"
    "3. SAFETY VIOLATIONS: Evaluate against the following 4 rules:\n"
    "   - Rule 1: Use of basic PPE (hard hats, proper clothing, closed-toe shoes, vests).\n"
    "   - Rule 2: Use of safety harness when working at height >3m.\n"
    "   - Rule 3: Edge protection for underground projects >3m deep.\n"
    "   - Rule 4: Worker in excavator blind spot or operation radius.\n"
    "   If violated, output {'reason':'...', 'bounding_box':[...]}. If NOT violated, output null.\n\n"
    "Respond exactly in this JSON format:\n"
    "```json\n"
    '{"caption":"detailed description","rule_1_violation":{"bounding_box":[[...]],"reason":"..."},"rule_2_violation":null,"rule_3_violation":null,"rule_4_violation":null,"excavator":[[...]],"rebar":[],"worker_with_white_hard_hat":[[...]]}\n'
    "```"
)

# ---------------------------------------------------------------------------
# Prompt Registry — maps config prompt_key to actual template
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT,
}