"""
Prompts for the MOCS auto-annotation pass.

The rule wording is IMPORTED from data/prompt_templates.py::SAFETY_RULE_TEXTS, never
copied. That dict is the repo's single source of truth for rule text -- the training
prompts, the LLM judge and now the annotator all read the same strings, so a teacher
can never be annotating against a different definition of rule_2 than the student is
trained on. tests/test_evaluation/test_llm_judge.py pins those strings by sha256.

WHY THIS PROMPT IS CALIBRATED TOWARD RECALL, NOT PRECISION
----------------------------------------------------------
This is an annotation harvest with a human verification stage, which makes the two
error types wildly asymmetric:

  * a FALSE POSITIVE costs the reviewing engineer a few seconds -- they reject it;
  * a FALSE NEGATIVE is permanently lost -- the engineer only ever sees images the
    teacher proposed, so a violation the teacher missed can never be recovered.

So the instruction is "report a plausible violation you can point at, a human will
confirm" -- the opposite of the training prompt's posture, which is deliberately
built to suppress over-flagging (see data/prompt_templates.py's note on the paper's
Table 7: zero-shot VLMs average 3.3-16.5% violation precision). Do not copy this
prompt back into the training path; it is tuned for the wrong objective there.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No images in the few-shot blocks. Their job is format and register conditioning, and
the paper does exactly this for LLaVA ("we input five captions without corresponding
images for in-context learning"). Each example image would also cost ~1,200 vision
tokens, which on an 80 GB H100 holding 66 GB of bf16 weights is memory you do not
have. Text-only few-shot is ~700 tokens total.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from data.prompt_templates import SAFETY_RULE_TEXTS

# Rule order is fixed everywhere in this repo (core/constants.py::RULES).
_RULE_KEYS = ("rule_1", "rule_2", "rule_3", "rule_4")

SAFETY_RULES_BLOCK = "".join(SAFETY_RULE_TEXTS[r] for r in _RULE_KEYS)


ANNOTATOR_SYSTEM_PROMPT = (
    "You are a construction safety expert preparing annotations for a research "
    "dataset. You are shown one photograph taken on a construction site. You "
    "produce a factual description of the scene and a record of which safety rules "
    "it violates, in exactly the format requested.\n\n"
    "Two standards govern everything you write. First, describe and report only what "
    "is visible in this photograph -- never what a construction site of this kind "
    "usually contains. Second, every violation you report must be one you can point "
    "at: a specific person, edge or machine you could draw a box around. If you "
    "cannot point at it, report null for that rule.\n\n"
    "A human safety engineer reviews every record you produce and rejects the "
    "incorrect ones. So when you can see a plausible violation and point to who or "
    "what is at fault, report it rather than leaving it out -- an over-report is "
    "corrected in review, an omission is lost. This does not license invention: a "
    "rule you cannot locate in the image is still null."
)


_OUTPUT_CONTRACT = (
    "Respond with a single JSON code block and nothing else, in exactly this shape:\n"
    "```json\n"
    '{"caption":"...","rule_1_violation":{"bounding_box":[[xmin, ymin, xmax, ymax]],'
    '"reason":"..."},"rule_2_violation":null,"rule_3_violation":null,'
    '"rule_4_violation":null,"confidence":{"rule_1":0.0,"rule_2":0.0,"rule_3":0.0,'
    '"rule_4":0.0}}'
    "\n```\n"
)

_FIELD_RULES = (
    "Field requirements:\n"
    "1. 'caption': one paragraph describing the people, equipment and materials that "
    "are actually present and what they are doing. State only facts you can see. Be "
    "concise -- around fifty words -- and let the amount of detail follow what the "
    "image shows.\n"
    "2. Each violation key: null when the rule is not violated, otherwise "
    '{"bounding_box": [[xmin, ymin, xmax, ymax]], "reason": "..."}. Rules are '
    "independent; any number of them may be violated, or none.\n"
    "3. 'bounding_box': coordinates scaled 0-1000 relative to the image, where 0,0 "
    "is the top-left corner. Each box encloses ONE person, edge or machine that "
    "violates that rule. List several boxes if several instances violate the same "
    "rule.\n"
    "4. 'reason': ONE sentence naming who or what is at fault, identified by "
    "position or appearance, and what the breach is. Around twelve to sixteen "
    "words. Do not hedge and do not mention the photograph.\n"
    "5. 'confidence': your own probability from 0.0 to 1.0 that each rule really is "
    "violated in this image, including rules you reported as null. This is used to "
    "order the human review queue, so report what you actually believe rather than "
    "rounding to 0 or 1.\n"
)


def render_fewshot_block(example: Dict[str, Any], index: int) -> str:
    """Renders one few-shot example as an EXAMPLE n: <json> block.

    `example` is a record from fewshot.json, produced by build_fewshot.py out of the
    real ConstructionSite train split. Boxes in it are already 0-1000, matching what
    the teacher is asked to emit.
    """
    payload: Dict[str, Any] = {"caption": example.get("caption", "")}
    for r in _RULE_KEYS:
        payload[f"{r}_violation"] = example.get(f"{r}_violation")
    if example.get("confidence"):
        payload["confidence"] = example["confidence"]

    label = example.get("label") or "example"
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"EXAMPLE {index} ({label}):\n```json\n{body}\n```\n"


def render_box_hints(record: Dict[str, Any], max_boxes: int = 12) -> str:
    """Turns a selection record's human-annotated MOCS boxes into a prompt hint.

    WHY GIVE THESE TO THE TEACHER. MOCS ships human-annotated Worker and machine
    boxes. Handing them over does two things a zero-shot VLM is otherwise bad at:
    it grounds the teacher's attention on the people and machines that are really
    there, and for rule_4 it supplies the exact geometry the rule is about, so the
    teacher is judging a relationship rather than also having to find the objects.

    THE TRADEOFF, STATED. Naming a worker next to an excavator primes rule_4, which
    inflates false positives. That is the correct direction to be wrong in here --
    the engineer filters false positives and cannot recover false negatives -- but
    it does mean the yield from a hinted bucket is NOT an unbiased estimate of true
    rule_4 prevalence. That is exactly what the `random_control` bucket is for.
    Disable with --no-box-hints.
    """
    boxes = record.get("mocs_boxes") or []
    if not boxes:
        return ""

    def to_1000(box: Sequence[float]) -> List[int]:
        return [int(round(c * 1000)) for c in box]

    lines: List[str] = []
    for entry in boxes[:max_boxes]:
        lines.append(f"   - {entry['category']}: {to_1000(entry['box'])}")
    omitted = len(boxes) - len(lines)
    if omitted > 0:
        lines.append(f"   - ... and {omitted} more object(s) of the same kinds")

    hint = (
        "This site's own object annotations, verified by human annotators, mark the "
        "following objects in this image (boxes scaled 0-1000):\n"
        + "\n".join(lines)
        + "\nUse them to ground your answer. They cover only machines and workers, so "
        "they say nothing about hard hats, harnesses, guard rails or excavation edges "
        "-- judge those from the image itself. They are also not a list of violations: "
        "an object being present is not a breach.\n"
    )

    pairs = record.get("worker_machine_pairs") or []
    if pairs:
        machines = sorted({p["machine_category"] for p in pairs})
        hint += (
            f"Note that {len(pairs)} worker/machine pair(s) in this image are close "
            f"enough to overlap once the machine's footprint is expanded "
            f"({', '.join(machines)}). Judge rule 4 carefully here: decide whether "
            "each of those workers is genuinely inside the machine's operating radius "
            "or blind spot, or merely nearby and out of reach.\n"
        )
    return hint


def build_annotation_prompt(
    fewshot: Optional[Sequence[Dict[str, Any]]] = None,
    record: Optional[Dict[str, Any]] = None,
    include_box_hints: bool = True,
) -> str:
    """The full user-turn prompt for one image."""
    parts: List[str] = [
        "Annotate this construction site image for a safety-inspection dataset.\n\n"
        "Judge the image against these four safety rules:\n",
        SAFETY_RULES_BLOCK,
        "\n",
        _FIELD_RULES,
        "\n",
    ]

    if fewshot:
        parts.append(
            "These examples show the exact output format and the wording style "
            "required. They are from a different set of images, so do not copy their "
            "content -- only their shape and register.\n\n"
        )
        for i, ex in enumerate(fewshot, start=1):
            parts.append(render_fewshot_block(ex, i))
        parts.append("\n")

    if include_box_hints and record is not None:
        hint = render_box_hints(record)
        if hint:
            parts.append(hint)
            parts.append("\n")

    parts.append(_OUTPUT_CONTRACT)
    return "".join(parts)


def prompt_fingerprint(prompt: str, system_prompt: str = ANNOTATOR_SYSTEM_PROMPT) -> str:
    """sha256 of system+user prompt, recorded in the run manifest.

    Same purpose as the LLM judge's `rubric_sha256`: two annotation batches can be
    checked for having been produced under identical instructions, which is the only
    way to know whether a yield difference is the images or the prompt.
    """
    import hashlib

    return hashlib.sha256((system_prompt + "\n" + prompt).encode("utf-8")).hexdigest()
