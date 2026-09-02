"""
All prompt text lives here as named constants.
Never hardcode prompt strings anywhere else in the codebase.

These prompts are calibrated against two sources rather than written freehand.

1. The source paper (Chen & Zou, *ConstructionSite 10k*, docs/dataset_paper.md), which
   defines the four safety rules, the three grounding classes, and the reasoning rubric
   the annotations were written to. Two of its findings shaped the wording:

   * **VLMs over-report violations.** Table 7: across seven models, violation precision
     averages 3.3-16.5 % against recall of 47-63 %, while the human upper bound is
     81-96 % precision. Over-flagging, not missing, is the dominant zero-shot failure,
     so every rule is phrased as something the model must be able to *point at*.
   * **Quantitative thresholds do not survive.** The paper lists its own "three meters"
     wording as a limitation: "quantitative estimations ... remain challenging for VLMs,
     as their training data often lack precise and reliable spatial annotations." The
     rules below name *observable* cues (a scaffold, an open trench) instead of metre
     thresholds the model cannot estimate from a single 2D image.

2. The ground-truth text itself. Measured per split by `scripts/dataset_report.py` on
   ARC 2026-09-02 -- and the two splits differ, which an earlier version of this note got
   wrong by quoting the test figures as if they were the training reference:

   | split          | rows | caption words: mean / median / p10 / p90 / max | sentences |
   |----------------|-----:|-----------------------------------------------:|----------:|
   | train (6308)   | 6308 |            55.0 / 54 / 36 / 75 / 134           |      3.02 |
   | test  (3004)   | 3004 |            48.5 / 44 / 14 / 90 / 163           |      3.58 |

   Reason length varies by rule, all one sentence: rule_1 11.8 words, rule_2 12.1,
   rule_3 10.1, **rule_4 18.1**.

   Note the train/test shift: training captions are longer and much tighter (p10 36)
   than test captions (p10 14). That is a property of the dataset, not something a prompt
   can fix -- and it is a further argument against naming any fixed length, since a model
   tuned to the training mean is scored against per-image test references that reach as
   low as 14 words.

   `reward_caption` and `reward_reasoning` both multiply their similarity score by a
   Gaussian length penalty, `exp(-0.5*((Lp-Lg)/sigma)^2)`, `sigma = max(0.6*Lg, 5)`.
   Against a 48-word reference a 90-word caption scores 0.345 and a 120-word one 0.044.
   An earlier caption prompt said "in detail" and listed five categories to cover, which
   reliably produces 90-120 words -- it was fighting the reward it was meant to maximise,
   on the component weighted 0.90 in `caption_only`.

   **No prompt states a word count.** The fix for that is brevity and relative framing,
   not a number. `Lg` in the penalty above is *this image's own* reference length, and on
   test those span p10 = 14 words to p90 = 90 -- a close-up of one bucket against a busy
   site. So a global "about 50 words" is only correct at the mean and is actively wrong at
   both tails, and how much to say about a given image is exactly what SFT learns from
   6308 targets. The prompts therefore say "concise", "facts only" and "let
   the detail follow the image" -- which is also how the paper words its own caption
   prompt ("Do not make assumptions, be concise, and describe facts only. Please
   describe with only one paragraph"), so our captioning numbers stay comparable to the
   benchmark. Reason gets "ONE sentence" for the same reason: GT reasons are 1.0
   sentences, so the sentence constraint lands the length without naming a count.

Two things deliberately kept OUT:

  * **No few-shot examples, including inline ones.** The paper uses 5-shot blocks to
    format-condition un-tuned models; SFT does that job here, and they would inflate
    every prompt in every phase. An earlier draft carried one example reason. It was
    removed for a second reason as well: it was necessarily a *rule 1* example, and
    rule 1 is already the dominant class, so it pushed the model toward the very
    over-flagging the rest of the prompt works against.
  * **Negatives the reward does not check.** Pydantic schemas ignore extra keys, so
    "do not write a caption" in `object_only` was unenforced; the fenced JSON template
    already pins the output shape. `caption_only` keeps exactly three negatives -- no
    fence, no JSON, no label -- because those are precisely what
    `output_parser.is_clean_prose` tests, which is what `reward_format` scores for that
    task. A negative the reward does not check only spends tokens and can prime the
    thing it forbids.

The single inline JSON template per task IS kept: it pins key order and the `null`
convention, and its key order matches `preprocessor.build_target_json` exactly
(caption -> rules -> objects for unified), so the numbered instructions and the template
cannot disagree.
"""

SYSTEM_PROMPT = (
    "You are an expert construction site safety inspector. Review the image you are "
    "given, report only what is actually visible in it, and answer in exactly the "
    "format requested."
)

# Shared by unified and violations_only so the two can never drift apart -- a wording
# difference between them would confound the multi-task vs single-task comparison.
_SAFETY_RULES = (
    "   - Rule 1 - basic PPE: a person on foot is missing basic PPE, e.g. no hard hat, "
    "or clothing that leaves the shoulders or legs uncovered.\n"
    "   - Rule 2 - safety harness: a person working at height (on a scaffold, roof, "
    "beam, ladder or other elevated structure) is not wearing a safety harness.\n"
    "   - Rule 3 - edge protection: an open excavation, trench, pit or floor edge has no "
    "guard rail, barrier or warning marking.\n"
    "   - Rule 4 - blind spot: a person is standing within the operating radius or "
    "blind spot of an excavator or other heavy machine.\n"
)

_VIOLATION_INSTRUCTIONS = (
    "   Report a rule only when you can point to the specific person, edge or machine in "
    "this image that violates it. If you cannot see such a violation, output null for "
    "that rule. Rules are independent: any number of them may be violated, or none.\n"
    "   For a violated rule output "
    "{\"reason\":\"...\", \"bounding_box\":[[xmin, ymin, xmax, ymax]]}, where the box "
    "encloses the violation and is scaled 0-1000.\n"
    "   Write the reason as ONE sentence saying who or what is at fault, identified by "
    "position or appearance, and what the breach is.\n"
)

# Shared by unified and object_only, for the same non-drift reason as the rules.
_OBJECT_CLASSES = (
    "   - excavator: any excavator or similar tracked digging machine, including its arm "
    "and bucket.\n"
    "   - rebar: exposed steel reinforcing bar - bundled, laid out, or tied in place. Do "
    "not confuse it with pipes, scaffolding tubes or lumber.\n"
    "   - worker_with_white_hard_hat: a person wearing a hard hat that is WHITE. A worker "
    "in a yellow, red, blue or orange hard hat does not count.\n"
)

# Numbered sections follow the key order of the JSON template below, which is the key
# order build_target_json emits: caption -> rules -> objects.
UNIFIED_INSPECTION_PROMPT = (
    "Inspect this construction site image. Output strictly a single JSON code block.\n\n"
    "1. 'caption': one concise paragraph covering the people, equipment and materials "
    "that are actually present and what they are doing. State only facts you can see, "
    "and let the amount of detail follow what the image shows.\n"
    "2. Violation keys: judge the image against these four safety rules.\n"
    + _SAFETY_RULES
    + _VIOLATION_INSTRUCTIONS +
    "3. Object keys ('excavator', 'rebar', 'worker_with_white_hard_hat'): a list of "
    "bounding boxes [[xmin, ymin, xmax, ymax]] scaled 0-1000, one box per instance. "
    "Use [] for a class that is absent.\n"
    + _OBJECT_CLASSES +
    "\nRespond exactly in this JSON format:\n"
    "```json\n"
    '{"caption":"...","rule_1_violation":{"bounding_box":[[xmin, ymin, xmax, ymax]],'
    '"reason":"..."},"rule_2_violation":null,"rule_3_violation":null,'
    '"rule_4_violation":null,"excavator":[[xmin, ymin, xmax, ymax]],"rebar":[],'
    '"worker_with_white_hard_hat":[[xmin, ymin, xmax, ymax]]}'
    "\n```"
)

VIOLATIONS_ONLY_PROMPT = (
    "Inspect this construction site image for safety rule violations. Output strictly a "
    "single JSON code block.\n\n"
    "Judge the image against these four safety rules:\n"
    + _SAFETY_RULES
    + _VIOLATION_INSTRUCTIONS +
    "\nRespond exactly in this JSON format:\n"
    "```json\n"
    '{"rule_1_violation":{"bounding_box":[[xmin, ymin, xmax, ymax]],"reason":"..."},'
    '"rule_2_violation":null,"rule_3_violation":null,"rule_4_violation":null}'
    "\n```"
)

OBJECT_ONLY_PROMPT = (
    "Locate the three target objects in this construction site image. Output strictly a "
    "single JSON code block.\n\n"
    "For each of the keys 'excavator', 'rebar' and 'worker_with_white_hard_hat', give a "
    "list of bounding boxes [[xmin, ymin, xmax, ymax]] scaled 0-1000, one box per "
    "instance. Use [] for a class that is absent.\n"
    + _OBJECT_CLASSES +
    "\nBox every instance you can see, including partly occluded ones, and box only what "
    "you can see. All three keys must appear, even when their value is [].\n\n"
    "Respond exactly in this JSON format:\n"
    "```json\n"
    '{"excavator":[[xmin, ymin, xmax, ymax]],"rebar":[],'
    '"worker_with_white_hard_hat":[[xmin, ymin, xmax, ymax]]}'
    "\n```"
)

# caption_only is the one task whose completion is NOT JSON. The caption is the entire
# output, so wrapping it in a JSON object would add a formatting confound to the exact
# quantity being measured. evaluation/output_parser.py::parse_output_for_task wraps the
# raw prose into {"caption": ...} so the rest of the stack stays dict-shaped.
CAPTION_ONLY_PROMPT = (
    "Describe this construction site image in one paragraph.\n\n"
    "Cover the people, the construction equipment and the material stockpiles that are "
    "actually present, what they are doing, and where they are in the scene. Be "
    "concise and describe only facts you can see - do not make assumptions and do not "
    "speculate about intent or danger. Let the amount of detail follow what the image "
    "shows: a close-up of a single machine needs far less than a busy site.\n\n"
    "Reply with the description itself and nothing else - no code fence, no JSON, and no "
    "heading or label in front of it."
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
