"""
All prompt text lives here as named constants. Never hardcode prompt
strings anywhere else in the codebase.
"""

SYSTEM_PROMPT_BASE = (
    "You are a construction safety expert analyzing construction site images. "
    "Always respond with valid JSON only, no extra text or markdown formatting."
)

RULE_VIOLATION_PROMPT = (
    "Analyze this construction site image for the following safety rules:\n"
    "Rule 1: Use of basic PPE when on foot (hard hats, appropriate clothing, "
    "closed-toe shoes, high-visibility vests at night, face protection for cutting/welding).\n"
    "Rule 2: Use of a safety harness when working from a height of >3 meters with unprotected edges.\n"
    "Rule 3: Edge protection/warning for underground projects >3 meters deep with steep retaining walls.\n"
    "Rule 4: Worker appearing in an excavator's blind spot or operation radius.\n\n"
    "Identify if ANY rule is violated. Respond ONLY in this JSON format:\n"
    '{"rule_id": "rule_1|rule_2|rule_3|rule_4|none", "violated": true|false, '
    '"reasoning": "1-2 sentences", "bounding_box": [ymin, xmin, ymax, xmax] or null}'
)

CAPTIONING_PROMPT = (
    "Describe this construction site image in detail, covering both foreground "
    "and background elements (equipment, workers, PPE, site conditions). "
    'Respond ONLY in this JSON format: {"caption": "your detailed caption"}'
)

GROUNDING_PROMPT = (
    "Locate all instances of the following in this image: excavator, rebar, "
    "worker_with_white_hard_hat. Respond ONLY in this JSON format: "
    '{"class_name": "excavator|rebar|worker_with_white_hard_hat", '
    '"bounding_boxes": [[ymin, xmin, ymax, xmax], ...]}'
)

ATTRIBUTES_PROMPT = (
    "Classify this construction site image along four dimensions: "
    "illumination, camera_distance, view, and quality_of_info. "
    'Respond ONLY in this JSON format: {"illumination": "...", '
    '"camera_distance": "...", "view": "...", "quality_of_info": "..."}'
)

# Maps config's prompt_key -> actual template, used by preprocessor.py
PROMPT_REGISTRY = {
    "RULE_VIOLATION_PROMPT": RULE_VIOLATION_PROMPT,
    "CAPTIONING_PROMPT": CAPTIONING_PROMPT,
    "GROUNDING_PROMPT": GROUNDING_PROMPT,
    "ATTRIBUTES_PROMPT": ATTRIBUTES_PROMPT,
}