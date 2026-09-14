"""
Reasoning evaluation.

Scores the "reason" field in safety violations using the standard captioning
metrics suite (BERTScore, METEOR, CIDEr, CLIPScore).
"""
from typing import Any, Dict, List, NamedTuple, Optional
from tqdm import tqdm

from core.constants import RULES
from core.logging import get_logger
from evaluation.metrics_captioning import compute_all_caption_metrics
# Shared "is this a violation?" predicate -- the same one the GRPO rewards use.
from rewards.reward_utils import _is_violation_present

logger = get_logger(__name__)

class TPReasonPair(NamedTuple):
    """One correctly identified violation whose reasoning can be scored."""
    idx: int            # position of the image in the prediction/reference lists
    rule: str           # "rule_1".."rule_4"
    pred_reason: Any    # RAW value as emitted -- may be "", None or non-str; callers sanitise
    gt_reason: Any      # RAW value from the reference
    image: Any          # the aligned image, or None when no images were supplied


def collect_tp_reason_pairs(
    pred_violations: List[Optional[Dict[str, Any]]],
    gt_violations: List[Optional[Dict[str, Any]]],
    images: Optional[List[Any]] = None,
    desc: str = "Reasoning Eval",
) -> Dict[str, List[TPReasonPair]]:
    """Per rule, every (prediction, reference) reasoning pair on a TRUE POSITIVE.

    SINGLE SOURCE OF TRUTH for which reasonings get scored. Both the text-similarity
    suite (batch_score_reasoning, below) and the LLM judge
    (evaluation/metrics_llm_judge.py) consume this, so
    ``reasoning_llm_judge_scored_count_rule_N`` can never drift from
    ``reasoning_text_similarity_scored_count_rule_N`` -- they are literally counts of
    the same list.

    A pair exists for rule r on image i when BOTH sides assert r (the shared
    _is_violation_present predicate, the same one the GRPO rewards use) AND both
    payloads are dicts, i.e. carry a `reason` field at all. This is the dataset
    paper's Stage-3 gate: "we only assess reasoning ... for correctly selected
    violations".

    Within an image, rules are visited in RULES order (the original iterated a set,
    whose order depends on the interpreter's hash seed). No metric value depends on
    that order; it is fixed so the pair lists are reproducible run to run.
    """
    per_rule: Dict[str, List[TPReasonPair]] = {r: [] for r in RULES}
    for i, (pred_dict, gt_dict) in enumerate(
        tqdm(zip(pred_violations, gt_violations), desc=desc, total=len(pred_violations))
    ):
        pred_dict = pred_dict or {}
        gt_dict = gt_dict or {}
        current_image = images[i] if images is not None else None

        # _is_violation_present also guards the .get() chain against a truthy non-dict
        # payload such as `true` or "yes", which previously raised AttributeError.
        pred_by_rule = {
            r: (pred_dict[f"{r}_violation"] or {}).get("reason", "")
            for r in RULES
            if _is_violation_present(pred_dict.get(f"{r}_violation"))
            and isinstance(pred_dict.get(f"{r}_violation"), dict)
        }
        gt_by_rule = {
            r: (gt_dict[f"{r}_violation"] or {}).get("reason", "")
            for r in RULES
            if _is_violation_present(gt_dict.get(f"{r}_violation"))
            and isinstance(gt_dict.get(f"{r}_violation"), dict)
        }

        for r in RULES:
            if r in pred_by_rule and r in gt_by_rule:
                per_rule[r].append(
                    TPReasonPair(i, r, pred_by_rule[r], gt_by_rule[r], current_image)
                )
    return per_rule


def batch_score_reasoning(
    pred_violations: List[Dict[str, Any]], 
    gt_violations: List[Dict[str, Any]],
    images: List[Any]  # <-- ADDED: Mandatory visual context
) -> Dict[str, float]:
    """Batched reasoning evaluation using the captioning metrics suite, broken down per rule."""
    if not pred_violations or not gt_violations:
        raise ValueError(
            "batch_score_reasoning requires non-empty predictions and references lists."
        )
    if images is None:
        raise ValueError(
            "batch_score_reasoning requires `images`; pass the image list aligned "
            "with pred_violations/gt_violations."
        )
    # ADDED: Ensure images match predictions and ground truth
    if len(pred_violations) != len(gt_violations) or len(pred_violations) != len(images):
        raise ValueError(
            "batch_score_reasoning: length mismatch between predictions, references, and images."
        )

    # Track globally for macro averages
    all_pred_reasons = []
    all_gt_reasons = []
    all_images = []

    # Track separately per rule
    rule_pred_reasons = {r: [] for r in RULES}
    rule_gt_reasons = {r: [] for r in RULES}
    rule_images = {r: [] for r in RULES}

    # The TP population comes from the shared helper (also used by the LLM judge).
    pairs_by_rule = collect_tp_reason_pairs(pred_violations, gt_violations, images)

    # The pooled lists interleave by image, as the original loop did.
    pooled = sorted(
        (pair for r in RULES for pair in pairs_by_rule[r]),
        key=lambda pair: (pair.idx, RULES.index(pair.rule)),
    )
    for pair in pooled:
        # Sanitize empty strings to avoid tokenizer crashes
        pred_reason = pair.pred_reason if pair.pred_reason and str(pair.pred_reason).strip() else "empty"
        gt_reason = pair.gt_reason if pair.gt_reason and str(pair.gt_reason).strip() else "empty"

        # Add to global trackers
        all_pred_reasons.append(pred_reason)
        all_gt_reasons.append(gt_reason)
        all_images.append(pair.image)

        # Add to rule trackers
        rule_pred_reasons[pair.rule].append(pred_reason)
        rule_gt_reasons[pair.rule].append(gt_reason)
        rule_images[pair.rule].append(pair.image)

    result = {}
    
    # 1. Compute global (micro/pooled) reasoning metrics
    if all_pred_reasons:
        logger.info(f"Computing global reasoning metrics over {len(all_pred_reasons)} valid reasons...")
        # ADDED: Pass images=all_images
        caption_res = compute_all_caption_metrics(all_pred_reasons, all_gt_reasons, images=all_images, include_spice=False)
        for k, v in caption_res.items():
            result[f"reasoning_text_similarity_{k}_micro"] = v
    else:
        result["reasoning_text_similarity_bertscore_precision_micro"] = 0.0
        result["reasoning_text_similarity_bertscore_recall_micro"] = 0.0
        result["reasoning_text_similarity_bertscore_f1_micro"] = 0.0
        result["reasoning_text_similarity_meteor_micro"] = 0.0
        result["reasoning_text_similarity_ciderd_micro"] = 0.0
        result["reasoning_text_similarity_clipscore_micro"] = 0.0
        result["reasoning_text_similarity_avg_words_per_caption_micro"] = 0.0
        result["reasoning_text_similarity_min_words_micro"] = 0.0
        result["reasoning_text_similarity_max_words_micro"] = 0.0
        
    # 2. Compute per-rule reasoning metrics
    for r in RULES:
        if rule_pred_reasons[r]:
            logger.info(f"Computing reasoning metrics for {r} over {len(rule_pred_reasons[r])} valid reasons...")
            rule_res = compute_all_caption_metrics(
                rule_pred_reasons[r], 
                rule_gt_reasons[r],
                images=rule_images[r], 
                include_spice=False
            )
            for k, v in rule_res.items():
                result[f"reasoning_text_similarity_{k}_{r}"] = v
        else:
            result[f"reasoning_text_similarity_bertscore_precision_{r}"] = 0.0
            result[f"reasoning_text_similarity_bertscore_recall_{r}"] = 0.0
            result[f"reasoning_text_similarity_bertscore_f1_{r}"] = 0.0
            result[f"reasoning_text_similarity_meteor_{r}"] = 0.0
            result[f"reasoning_text_similarity_ciderd_{r}"] = 0.0
            result[f"reasoning_text_similarity_clipscore_{r}"] = 0.0
            result[f"reasoning_text_similarity_avg_words_per_caption_{r}"] = 0.0
            result[f"reasoning_text_similarity_min_words_{r}"] = 0.0
            result[f"reasoning_text_similarity_max_words_{r}"] = 0.0
            # Emitted as an explicit 0 rather than left absent. Every reasoning
            # score is conditioned on the model's own true positives, so
            # scored_count is the sample size the score rests on and must always
            # be readable next to it -- a MISSING key showed up as a blank cell
            # in the comparison CSV, which reads as "unknown" rather than "zero
            # detections, nothing was measured".
            result[f"reasoning_text_similarity_scored_count_{r}"] = 0
            result[f"reasoning_text_similarity_total_count_{r}"] = 0
            
    # 3. Macro = mean over the rules that were actually MEASURABLE.
    #
    # Every reasoning metric here is TRUE-POSITIVE-CONDITIONED: a rule only has
    # reasoning text to score when the model correctly identified that rule on
    # at least one image. A rule with zero true positives was never measured,
    # and that is not the same fact as "it scored zero".
    #
    # This block used to read the per-rule key with a `.get(..., 0.0)` default
    # and divide by len(RULES) regardless. On the real vo-2b-sft run -- zero
    # rule_3 and zero rule_4 detections -- it reported
    #     bertscore_f1_macro = (0.820 + 0.743 + 0.0 + 0.0) / 4 = 0.391
    # for a model whose rule_1 bertscore of 0.820 was the highest of any run in
    # the comparison. Reported as 0.391 it looked like a catastrophic
    # regression below its own untrained baseline. It was an averaging artifact.
    # Same for clipscore_macro (0.339 vs 0.679) and ciderd_macro (1.264 vs 2.528).
    #
    # Membership is decided by whether the rule's bucket HAD DATA
    # (`rule_pred_reasons[r]`), never by testing the score against 0.0 -- a rule
    # that genuinely scored 0.0 on real text is a real measurement and keeps its
    # weight in the mean.
    #
    # `..._macro_n_rules` publishes the divisor, so a macro over 2 rules can
    # never again be silently compared against a macro over 4. It is counted per
    # metric, because a metric the captioning suite did not return for a rule
    # (e.g. ciderd under a skip flag) is missing for that rule alone.
    measured_rules = [r for r in RULES if rule_pred_reasons[r]]
    metrics_keys = ["bertscore_f1", "meteor", "ciderd", "clipscore"]
    for k in metrics_keys:
        rule_scores = [
            result[f"reasoning_text_similarity_{k}_{r}"]
            for r in measured_rules
            if f"reasoning_text_similarity_{k}_{r}" in result
        ]
        result[f"reasoning_text_similarity_{k}_macro"] = (
            sum(rule_scores) / len(rule_scores) if rule_scores else 0.0
        )
        result[f"reasoning_text_similarity_{k}_macro_n_rules"] = len(rule_scores)

    return result