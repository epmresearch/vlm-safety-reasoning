"""
A SINGLE system + user prompt. The model is asked to perform the full site
inspection (caption + object detection + safety rule check) in one pass and
return one JSON object. No separate prompts per task, no per-class grounding
prompts.
"""

SYSTEM_PROMPT_BASE = (
    "You are an expert AI construction safety inspection pipeline. Analyze the "
    "provided construction site image and perform a full inspection in a single "
    "pass. Always respond with valid JSON only — no extra text, no markdown "
    "formatting, no code fences."
)

UNIFIED_INSPECTION_PROMPT = (
    "Perform a full site inspection of this construction image. Your response "
    "must include ALL of the following in a single JSON object:\n\n"
    "1. CAPTION — a detailed description covering both foreground and background "
    "elements (workers, equipment, PPE, material stockpiles, site conditions).\n\n"
    "2. DETECTED OBJECTS — bounding boxes for every instance of:\n"
    "   - excavators\n"
    "   - rebar\n"
    "   - workers wearing a white hard hat\n"
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
    "Respond ONLY in this exact JSON format:\n"
    "{\n"
    '  "caption": "your detailed caption",\n'
    '  "detected_objects": {\n'
    '    "excavators": [[ymin, xmin, ymax, xmax], ...],\n'
    '    "rebar": [[ymin, xmin, ymax, xmax], ...],\n'
    '    "white_hard_hat_workers": [[ymin, xmin, ymax, xmax], ...]\n'
    "  },\n"
    '  "safety_violations": [\n'
    '    {"rule_id": "rule_1", "reason": "1-2 sentences", '
    '"bounding_boxes": [[ymin, xmin, ymax, xmax], ...]}\n'
    "  ]\n"
    "}"
)

PROMPT_REGISTRY = {
    "UNIFIED_INSPECTION_PROMPT": UNIFIED_INSPECTION_PROMPT,
}