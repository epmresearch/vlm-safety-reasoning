"""
LLM-as-a-judge scoring of violation reasoning, replicating the ConstructionSite-10k paper.

Runs ALONGSIDE the text-similarity reasoning metrics (evaluation/metrics_reasoning.py);
it replaces none of them. Those measure lexical/semantic overlap with the reference
sentence. This measures what the dataset paper measures -- whether an explanation is
about the right rule, describes the same violation, and pins down the violator -- so
our numbers land in the same units as the paper's Table 8 ("average score from LLM
judge", 0-6 per rule; human upper bound 5.2 on rule_1).

WHAT THE PAPER SPECIFIES (docs/others_extra/dataset_paper.md, "Safety rule violation VQA")
------------------------------------------------------------------------------------------
  * Three criteria, verbatim definitions copied into CRITERIA_DEFINITIONS below.
  * Integer marks 0-2 each: "0 indicates incapability for the criterion, 1 suggests
    attempting but not succeeding well, and 2 signifies acceptability". Max total 6.
  * Judge: Meta Llama 3 8B Instruct. Text-only -- it never sees the image.
  * "we only assess reasoning ... for correctly selected violations" -> true positives
    only. The population comes from metrics_reasoning.collect_tp_reason_pairs, the same
    list the text-similarity suite scores, so the two scored_counts are always equal.
  * "Evaluation of reasoning for each rule was conducted separately" with "three human
    judgment examples ... for each rule" -> a per-rule three-shot prompt.
  * "beam search method with a fixed random seed of 20, set the number of beams to 5,
    and consistently use the same three-shot examples for each rule".
  * The judge's reply, as printed in Figure 8 and the appendix figure, is plain text:
        Relevance: 2 marks / Equivalence: 1 mark / Specificity: 1 mark / Overall: 4 marks
    (Figure 8 actually prints "Relavance" -- the parser accepts both spellings.)

WHAT THE PAPER DOES NOT SPECIFY, AND WHAT WE CHOSE
--------------------------------------------------
  * The judge prompt itself is not published. JUDGE_SYSTEM_PROMPT is written from the
    paper's own criteria and scale text, word for word, plus a strict output format.
  * The rule text shown to the judge is data/prompt_templates.py::SAFETY_RULE_TEXTS --
    the exact wording our models were prompted with. Judging "adheres to the specific
    safety rule" against wording the model never saw would score the prompt, not the
    model.
  * Few-shot scores. The paper's examples were judged by humans; we have no human
    judgments. FEWSHOT_EXAMPLES uses the paper's five published, human-scored worked
    examples verbatim as anchors (Fig. 8 + appendix; 2 for rule_1, 1 each for rules
    2-4) and fills the remaining seven slots with references taken from the TRAIN split
    (the paper draws few-shot examples from train "to avoid information leakage from
    the test set") paired with hand-authored candidates. Those seven scores are OUR
    calibration, and each is tagged `source="authored:..."` so that is never forgotten.
  * The paper validated its judge against humans (Spearman 0.83, Pearson 0.91). We have
    no human labels to repeat that, so every judged item is written to
    llm_judge_details.json for inspection instead.
  * A reply that cannot be parsed scores 0/0/0 and is counted
    (reasoning_llm_judge_unparsed_*). That DEPRESSES the means, which is why the rate is
    reported next to them.
  * An empty candidate reasoning scores 0/0/0 without calling the model: an empty
    explanation is incapable on every criterion by definition, and asking an LLM to grade
    nothing only invites a hallucinated score.
"""
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.constants import RULES
from core.logging import get_logger
from data.prompt_templates import safety_rule_description
from evaluation.metrics_reasoning import TPReasonPair, collect_tp_reason_pairs

logger = get_logger(__name__)

KEY_PREFIX = "reasoning_llm_judge_"
JUDGE_CRITERIA = ("relevance", "equivalence", "specificity")
MAX_MARK = 2
MAX_TOTAL = MAX_MARK * len(JUDGE_CRITERIA)

# Defaults, overridden key-by-key by configs/base.yaml::llm_judge.
DEFAULT_JUDGE_CONFIG: Dict[str, Any] = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "num_beams": 5,           # paper
    "seed": 20,               # paper
    "max_new_tokens": 64,     # the four-line reply is ~25 tokens
    # 1, not larger: batched beam search pads prompts to a common length, and in bf16
    # the padding can nudge logits enough to flip a close beam. At batch 1 an item's
    # score depends only on its own text, so the SAME (reference, candidate) pair
    # scores identically across runs -- which is what a v1-vs-v2 comparison needs.
    "batch_size": 1,
    "dtype": "bfloat16",
    # false: a judge that cannot load (model not cached, OOM) logs an ERROR and its keys
    # are omitted, but every other metric is still written. True makes it fatal.
    "fail_hard": False,
}

# --- The paper's rubric, verbatim --------------------------------------------------
CRITERIA_DEFINITIONS = {
    "relevance": "Whether the explanation adheres to the specific safety rule.",
    "equivalence": (
        "Whether the explanation is talking about the same violation, in terms of object, "
        "and reason, as the ground truth."
    ),
    "specificity": (
        "Whether the explanation pinpoints the specific violator by describing the location "
        "or attribute."
    ),
}
SCALE_DEFINITION = (
    "0 indicates incapability for the criterion, 1 suggests attempting but not succeeding "
    "well, and 2 signifies acceptability for the criterion."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert construction site safety evaluator. You grade a candidate "
    "explanation of a safety rule violation against a reference explanation written by a "
    "human safety inspector. Both explanations concern the same safety rule and the same "
    "image; the reference explanation is the ground truth. You do not see the image.\n\n"
    "Score the candidate explanation on three criteria:\n"
    f"- Relevance: {CRITERIA_DEFINITIONS['relevance']}\n"
    f"- Equivalence: {CRITERIA_DEFINITIONS['equivalence']}\n"
    f"- Specificity: {CRITERIA_DEFINITIONS['specificity']}\n\n"
    f"Each criterion receives an integer mark from 0 to 2: {SCALE_DEFINITION} "
    "The overall mark is the sum of the three marks, from 0 to 6.\n\n"
    "Reply with exactly these four lines and nothing else:\n"
    "Relevance: <0-2> marks\n"
    "Equivalence: <0-2> marks\n"
    "Specificity: <0-2> marks\n"
    "Overall: <0-6> marks"
)


@dataclass(frozen=True)
class JudgeExample:
    reference: str
    candidate: str
    relevance: int
    equivalence: int
    specificity: int
    # "paper:..." = a human judgment published in the dataset paper (verbatim).
    # "authored:train/<image_id>" = reference from the train split, candidate and
    # scores written by us.
    source: str

    @property
    def total(self) -> int:
        return self.relevance + self.equivalence + self.specificity


# --- Three-shot blocks, exactly three per rule ---------------------------------------
# Paper anchors are copied verbatim (including their original wording choices). Across
# each rule's three examples the totals are spread, and across all twelve every mark
# 0/1/2 appears for every criterion, so no level of the scale is left un-anchored.
FEWSHOT_EXAMPLES: Dict[str, Tuple[JudgeExample, ...]] = {
    "rule_1": (
        JudgeExample(
            reference="Worker with a black cap and white shirt on the left is not wearing a hard hat.",
            candidate="The worker on the left is not wearing a hard hat, and his clothes do not cover his shoulders.",
            relevance=2, equivalence=2, specificity=2, source="paper:figure-8",
        ),
        JudgeExample(
            reference="The worker to the left of the truck is not wearing a hard hat.",
            candidate="A worker near the truck looks unsafe and should be more careful on site.",
            relevance=1, equivalence=0, specificity=1, source="authored:train/0005159",
        ),
        JudgeExample(
            reference="The worker with a camouflage uniform is not wearing a hard hat.",
            candidate="The construction worker is not wearing any visible PPE, such as a hard hat or high-visibility vest.",
            relevance=2, equivalence=1, specificity=1, source="paper:appendix-figure",
        ),
    ),
    "rule_2": (
        JudgeExample(
            reference="The worker standing on top of the ladder is not wearing a safety harness.",
            candidate="The worker at the top of the ladder has no safety harness while working at height.",
            relevance=2, equivalence=2, specificity=2, source="authored:train/0008429",
        ),
        JudgeExample(
            reference="Some workers on the rooftop in the upper left are not wearing safety harnesses.",
            candidate="Workers at the site are not wearing hard hats.",
            relevance=0, equivalence=0, specificity=0, source="authored:train/0005886",
        ),
        JudgeExample(
            reference="The person with a blue t-shirt and standing on top of the scaffold is not wearing a safety harness.",
            candidate="The worker on the scaffold is not using a safety harness while working at a height.",
            relevance=2, equivalence=2, specificity=1, source="paper:appendix-figure",
        ),
    ),
    "rule_3": (
        JudgeExample(
            reference="The edge of the excavation in the upper left is not guarded.",
            candidate="The edge of the excavation in the upper left has no guard rail or barrier.",
            relevance=2, equivalence=2, specificity=2, source="authored:train/0006692",
        ),
        JudgeExample(
            reference="The edge of the excavation in the lower right is not protected.",
            candidate="There are no workers visible in the image who are at a height of three meters or more.",
            relevance=0, equivalence=0, specificity=1, source="paper:appendix-figure",
        ),
        JudgeExample(
            reference="The left edge and right edge of the excavation are not guarded.",
            candidate="There is an open excavation on site without proper edge protection.",
            relevance=2, equivalence=1, specificity=0, source="authored:train/0005530",
        ),
    ),
    "rule_4": (
        JudgeExample(
            reference="The worker with a white hard hat is too close to the excavator and is in the blind spot.",
            candidate="A worker is in close proximity to an excavator, potentially in the operator's blind spot.",
            relevance=2, equivalence=2, specificity=2, source="paper:appendix-figure",
        ),
        JudgeExample(
            reference="The worker on the left is in the blind spot of the excavator.",
            candidate="The worker on the left is not wearing a high-visibility vest.",
            relevance=0, equivalence=0, specificity=2, source="authored:train/0005496",
        ),
        JudgeExample(
            reference="The worker on the left is in the blind spot and operation radius of the excavator.",
            candidate="A worker is standing near heavy machinery on the site.",
            relevance=1, equivalence=1, specificity=0, source="authored:train/0009230",
        ),
    ),
}


# ======================================================================================
# Prompt
# ======================================================================================

def _mark_word(n: int) -> str:
    return "mark" if n == 1 else "marks"


def format_judge_reply(relevance: int, equivalence: int, specificity: int) -> str:
    """The judge's reply format, as printed in the paper's figures."""
    total = relevance + equivalence + specificity
    return (
        f"Relevance: {relevance} {_mark_word(relevance)}\n"
        f"Equivalence: {equivalence} {_mark_word(equivalence)}\n"
        f"Specificity: {specificity} {_mark_word(specificity)}\n"
        f"Overall: {total} {_mark_word(total)}"
    )


def _item_message(rule: str, reference: str, candidate: str) -> str:
    return (
        f"Safety rule: {safety_rule_description(rule)}\n"
        f"Reference explanation (ground truth): {reference}\n"
        f"Candidate explanation: {candidate}"
    )


def build_judge_messages(rule: str, reference: str, candidate: str) -> List[Dict[str, str]]:
    """Chat messages for one judgment: system rubric, the rule's three-shot exchanges,
    then the item. Tokenizer-agnostic -- the chat template is applied at generation time,
    so this is testable without the model."""
    if rule not in FEWSHOT_EXAMPLES:
        raise ValueError(f"No few-shot examples for {rule!r}; known: {sorted(FEWSHOT_EXAMPLES)}")
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    for ex in FEWSHOT_EXAMPLES[rule]:
        messages.append({"role": "user", "content": _item_message(rule, ex.reference, ex.candidate)})
        messages.append({
            "role": "assistant",
            "content": format_judge_reply(ex.relevance, ex.equivalence, ex.specificity),
        })
    messages.append({"role": "user", "content": _item_message(rule, reference, candidate)})
    return messages


def rubric_fingerprint() -> str:
    """sha256 over the system prompt, rule texts and every few-shot block.

    Recorded with each run, so two sets of judge scores can be checked for having been
    produced under the same rubric before they are compared.
    """
    payload = {
        "system": JUDGE_SYSTEM_PROMPT,
        "rules": {r: safety_rule_description(r) for r in RULES},
        "fewshot": {
            r: [(e.reference, e.candidate, e.relevance, e.equivalence, e.specificity)
                for e in exs]
            for r, exs in FEWSHOT_EXAMPLES.items()
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ======================================================================================
# Parsing
# ======================================================================================

# Label alternatives per criterion. "relavance" is the paper's own Figure-8 spelling.
_LABELS = {
    "relevance": r"rel[ae]vance",
    "equivalence": r"equivalence",
    "specificity": r"specificity",
    "overall": r"overall",
}
# Label, optional markdown/punctuation, then an INTEGER not followed by a decimal part
# ("1.5" is not a valid mark and must not be read as 1).
_MARK_RE = {
    name: re.compile(
        rf"\b{label}\b\s*\**\s*[:=\-]?\s*\**\s*(-?\d+)(?![.,]\d)",
        re.IGNORECASE,
    )
    for name, label in _LABELS.items()
}


@dataclass
class ParsedJudgment:
    relevance: int
    equivalence: int
    specificity: int
    clamped: bool = False            # a mark was outside 0-2 and was clamped into range
    overall_reported: Optional[int] = None

    @property
    def total(self) -> int:
        return self.relevance + self.equivalence + self.specificity

    @property
    def overall_mismatch(self) -> bool:
        return self.overall_reported is not None and self.overall_reported != self.total


def parse_judge_output(text: Optional[str]) -> Optional[ParsedJudgment]:
    """Reads the three marks out of a judge reply.

    Tolerant of formatting (mark/marks, markdown bold, "=" or "-" separators, prose
    around the lines, the "Relavance" spelling). Strict about content: every criterion
    must be present as an integer, else None (unparseable). A mark outside 0-2 is clamped
    and flagged. "Overall" is read for a consistency check only -- the total is always
    the sum of the three criteria, never the reported line.
    """
    if not text or not str(text).strip():
        return None
    marks: Dict[str, int] = {}
    clamped = False
    for name in JUDGE_CRITERIA:
        m = _MARK_RE[name].search(text)
        if m is None:
            return None
        value = int(m.group(1))
        if value < 0 or value > MAX_MARK:
            clamped = True
            value = min(max(value, 0), MAX_MARK)
        marks[name] = value
    overall = _MARK_RE["overall"].search(text)
    return ParsedJudgment(
        relevance=marks["relevance"],
        equivalence=marks["equivalence"],
        specificity=marks["specificity"],
        clamped=clamped,
        overall_reported=int(overall.group(1)) if overall else None,
    )


# ======================================================================================
# Model
# ======================================================================================

_JUDGE_MODEL = None
_JUDGE_TOKENIZER = None
_JUDGE_MODEL_ID = None


def resolve_judge_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DEFAULT_JUDGE_CONFIG <- configs/base.yaml::llm_judge <- explicit overrides."""
    cfg = dict(DEFAULT_JUDGE_CONFIG)
    try:
        from core.config import load_base_config
        cfg.update(load_base_config().get("llm_judge") or {})
    except Exception as e:  # a config read problem must not silently change the rubric
        logger.warning(f"Could not read llm_judge from configs/base.yaml ({e}); using defaults.")
    cfg.update(overrides or {})
    return cfg


def _get_judge_model(cfg: Dict[str, Any]):
    """Loads the judge ONCE per process (lazy singleton, like _get_clip_model in
    evaluation/metrics_captioning.py) and reuses it for every item.

    local_files_only=True: SLURM compute nodes have no internet. A missing cache must
    fail immediately with the download command, not hang on a network timeout.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER, _JUDGE_MODEL_ID
    model_id = cfg["model_id"]
    if _JUDGE_MODEL is not None and _JUDGE_MODEL_ID == model_id:
        return _JUDGE_MODEL, _JUDGE_TOKENIZER

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The LLM judge needs a GPU (an 8B model in bf16 is impractical on CPU). "
            "Run evaluation on a GPU node, or drop --use_llm_judge."
        )

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    # transformers >= 5 renamed torch_dtype -> dtype; ARC pins 5.4.0.
    dtype_kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

    logger.info(f"Loading LLM judge {model_id} ({cfg.get('dtype')}) from the local HF cache...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, **{dtype_kwarg: dtype}
        )
    except OSError as e:
        raise RuntimeError(
            f"LLM judge model {model_id!r} is not in the local HuggingFace cache "
            f"(HF_HOME must point at the same cache the download used). On the login node: "
            f"export HF_HOME=$HOME/scratch/hf_cache && hf download {model_id} --exclude 'original/*'. "
            f"Underlying error: {e}"
        ) from e

    # Llama 3 ships no pad token. Decoder-only generation must pad on the LEFT, so the
    # newly generated tokens start at the same position for every row in a batch.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = model.to("cuda").eval()
    _JUDGE_MODEL, _JUDGE_TOKENIZER, _JUDGE_MODEL_ID = model, tokenizer, model_id
    logger.info("LLM judge loaded.")
    return model, tokenizer


def make_hf_generate_fn(cfg: Dict[str, Any]) -> Callable[[Sequence[List[Dict[str, str]]]], List[str]]:
    """Returns generate(list_of_message_lists) -> list_of_reply_texts backed by the model."""
    import torch
    from transformers import GenerationConfig, set_seed

    model, tokenizer = _get_judge_model(cfg)

    # Llama 3 Instruct ends an assistant turn with <|eot_id|>, which is not the
    # tokenizer's base eos (<|end_of_text|>). Without it, generation runs on past the
    # reply to max_new_tokens.
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id and eot not in eos_ids:
        eos_ids.append(eot)

    # A FRESH GenerationConfig rather than the checkpoint's own: Llama 3's
    # generation_config.json carries sampling settings (temperature 0.6, top_p 0.9) that
    # must not leak into a deterministic beam search.
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=int(cfg["num_beams"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    seed = int(cfg["seed"])

    def generate(message_batches: Sequence[List[Dict[str, str]]]) -> List[str]:
        # Reseeded per call so a batch's output never depends on how many batches ran
        # before it. (Beam search without sampling draws no random numbers; this pins
        # the paper's seed regardless.)
        set_seed(seed)
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_batches
        ]
        # add_special_tokens=False: the chat template already begins with
        # <|begin_of_text|>; tokenizing with specials would prepend a second BOS.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]

    return generate


# ======================================================================================
# Scoring
# ======================================================================================

def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-rule, micro and macro metrics from per-item records."""
    metrics: Dict[str, Any] = {}
    by_rule = {r: [x for x in records if x["rule"] == r] for r in RULES}

    def means(items):
        n = len(items)
        out = {c: (sum(x["scores"][c] for x in items) / n if n else 0.0) for c in JUDGE_CRITERIA}
        out["total"] = sum(x["scores"]["total"] for x in items) / n if n else 0.0
        return out

    measured = []
    for r in RULES:
        items = by_rule[r]
        m = means(items)
        for name, v in m.items():
            metrics[f"{KEY_PREFIX}{name}_{r}"] = v
        # Always emitted, including 0, so the sample size is never a blank cell.
        metrics[f"{KEY_PREFIX}scored_count_{r}"] = len(items)
        metrics[f"{KEY_PREFIX}unparsed_count_{r}"] = sum(1 for x in items if x["status"] == "unparsed")
        if items:
            measured.append(m)

    m = means(records)
    for name, v in m.items():
        metrics[f"{KEY_PREFIX}{name}_micro"] = v
    n_total = len(records)
    n_unparsed = sum(1 for x in records if x["status"] == "unparsed")
    metrics[f"{KEY_PREFIX}scored_count_micro"] = n_total
    metrics[f"{KEY_PREFIX}unparsed_count_micro"] = n_unparsed
    metrics[f"{KEY_PREFIX}unparsed_rate_micro"] = n_unparsed / n_total if n_total else 0.0
    metrics[f"{KEY_PREFIX}empty_candidate_count_micro"] = sum(
        1 for x in records if x["status"] == "empty_candidate")
    metrics[f"{KEY_PREFIX}clamped_count_micro"] = sum(1 for x in records if x.get("clamped"))

    # MACRO over rules that had at least one true positive to judge, with the divisor
    # published -- the same rule as reasoning_text_similarity_*_macro (see the macro
    # note in CLAUDE.md). A rule with no TP was not measured, which is not a score of 0.
    for name in list(JUDGE_CRITERIA) + ["total"]:
        vals = [mm[name] for mm in measured]
        metrics[f"{KEY_PREFIX}{name}_macro"] = sum(vals) / len(vals) if vals else 0.0
        metrics[f"{KEY_PREFIX}{name}_macro_n_rules"] = len(vals)
    return metrics


def score_reasoning_with_llm_judge(
    pred_violations: List[Optional[Dict[str, Any]]],
    gt_violations: List[Optional[Dict[str, Any]]],
    cfg: Optional[Dict[str, Any]] = None,
    generate_fn: Optional[Callable[[Sequence[List[Dict[str, str]]]], List[str]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Judges every true-positive reasoning. Returns (metrics, details, status).

    generate_fn is injectable so tests run without the model; when None the real model
    is loaded -- but only if at least one item actually needs judging.
    """
    cfg = resolve_judge_config(cfg)
    t0 = time.time()
    pairs_by_rule = collect_tp_reason_pairs(pred_violations, gt_violations, desc="LLM judge TP collection")
    pairs: List[TPReasonPair] = [p for r in RULES for p in pairs_by_rule[r]]

    records: List[Dict[str, Any]] = []
    pending: List[Tuple[int, List[Dict[str, str]]]] = []
    for pair in pairs:
        gt = gt_violations[pair.idx] or {}
        reference, candidate = _as_text(pair.gt_reason), _as_text(pair.pred_reason)
        rec = {
            "image_id": gt.get("image_id"),
            "index": pair.idx,
            "rule": pair.rule,
            "reference": reference,
            "candidate": candidate,
            "raw_output": None,
            "status": None,
            "scores": None,
            "clamped": False,
        }
        if not candidate:
            rec["status"] = "empty_candidate"
            rec["scores"] = {c: 0 for c in JUDGE_CRITERIA} | {"total": 0}
        else:
            pending.append((len(records), build_judge_messages(pair.rule, reference or "(none)", candidate)))
        records.append(rec)

    n_overall_mismatch = 0
    if pending:
        if generate_fn is None:
            generate_fn = make_hf_generate_fn(cfg)
        bs = max(1, int(cfg["batch_size"]))
        logger.info(f"LLM judge: scoring {len(pending)} reasoning(s) "
                    f"(beams={cfg['num_beams']}, seed={cfg['seed']}, batch={bs})...")
        for start in range(0, len(pending), bs):
            chunk = pending[start:start + bs]
            replies = generate_fn([msgs for _, msgs in chunk])
            if len(replies) != len(chunk):
                raise RuntimeError(f"generate_fn returned {len(replies)} replies for {len(chunk)} prompts")
            for (rec_idx, _), reply in zip(chunk, replies):
                rec = records[rec_idx]
                rec["raw_output"] = reply
                parsed = parse_judge_output(reply)
                if parsed is None:
                    rec["status"] = "unparsed"
                    rec["scores"] = {c: 0 for c in JUDGE_CRITERIA} | {"total": 0}
                    logger.warning(
                        f"LLM judge reply unparseable for image {rec['image_id']} {rec['rule']}; "
                        f"scored 0/0/0. Reply: {str(reply)[:200]!r}"
                    )
                else:
                    rec["status"] = "ok"
                    rec["clamped"] = parsed.clamped
                    rec["scores"] = {
                        "relevance": parsed.relevance,
                        "equivalence": parsed.equivalence,
                        "specificity": parsed.specificity,
                        "total": parsed.total,
                    }
                    n_overall_mismatch += int(parsed.overall_mismatch)
            done = min(start + bs, len(pending))
            if done % 50 < bs or done == len(pending):
                logger.info(f"LLM judge: {done}/{len(pending)} judged")

    metrics = _aggregate(records)
    metrics[f"{KEY_PREFIX}model_id"] = cfg["model_id"]   # string: provenance, never a metric row

    status = {
        "status": "ok",
        "model_id": cfg["model_id"],
        "num_beams": int(cfg["num_beams"]),
        "seed": int(cfg["seed"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "batch_size": int(cfg["batch_size"]),
        "dtype": cfg.get("dtype"),
        "rubric_sha256": rubric_fingerprint(),
        "fewshot_sources": {r: [e.source for e in exs] for r, exs in FEWSHOT_EXAMPLES.items()},
        "n_items": len(records),
        "n_model_calls": len(pending),
        "n_unparsed": metrics[f"{KEY_PREFIX}unparsed_count_micro"],
        "n_empty_candidate": metrics[f"{KEY_PREFIX}empty_candidate_count_micro"],
        "n_clamped": metrics[f"{KEY_PREFIX}clamped_count_micro"],
        "n_overall_line_mismatch": n_overall_mismatch,
        "total_histogram": {str(k): v for k, v in sorted(
            Counter(r["scores"]["total"] for r in records).items())},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return metrics, records, status


def run_llm_judge(
    pred_violations: List[Optional[Dict[str, Any]]],
    gt_violations: List[Optional[Dict[str, Any]]],
    cfg: Optional[Dict[str, Any]] = None,
    generate_fn=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Fail-soft wrapper used by the evaluator.

    On any error (model not cached, OOM, a broken reply loop) this logs an ERROR, returns
    NO metric keys and a status of "failed" -- so a judge problem can never cost the
    3004-image evaluation it runs inside. Set llm_judge.fail_hard: true to make it fatal.
    """
    resolved = resolve_judge_config(cfg)
    try:
        return score_reasoning_with_llm_judge(pred_violations, gt_violations, resolved, generate_fn)
    except Exception as e:
        if resolved.get("fail_hard"):
            raise
        logger.error(
            f"LLM judge FAILED and was skipped -- its reasoning_llm_judge_* keys are omitted "
            f"from metrics.json; every other metric is unaffected. Error: {e!r}",
            exc_info=True,
        )
        return {}, [], {"status": "failed", "model_id": resolved.get("model_id"), "error": repr(e)}
