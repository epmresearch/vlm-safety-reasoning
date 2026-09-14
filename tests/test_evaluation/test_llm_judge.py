"""LLM-as-a-judge reasoning evaluation (evaluation/metrics_llm_judge.py).

Every test here is CPU-only. The judge model is never loaded: generation is
replaced by an injected fake, and the one test that exercises the real loader
forces torch.cuda.is_available() -> False so it cannot pull 16 GB onto a GPU node.
"""
import hashlib
import io
import pathlib
import re
from unittest.mock import patch

import pytest
import yaml

from core.constants import RULES
from evaluation import metrics_llm_judge as J
from evaluation.metrics_llm_judge import (
    CRITERIA_DEFINITIONS, FEWSHOT_EXAMPLES, JUDGE_CRITERIA, JUDGE_SYSTEM_PROMPT, KEY_PREFIX,
    SCALE_DEFINITION, build_judge_messages, format_judge_reply, parse_judge_output,
    rubric_fingerprint, run_llm_judge, score_reasoning_with_llm_judge,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _v(reason, box=((0.1, 0.1, 0.3, 0.3),)):
    return {"reason": reason, "bounding_box": [list(b) for b in box]}


def _gt(image_id, **rules):
    d = {"image_id": image_id}
    for r in RULES:
        d[f"{r}_violation"] = rules.get(r)
    return d


def _pred(**rules):
    return {f"{r}_violation": rules.get(r) for r in RULES}


class FakeJudge:
    """Deterministic stand-in for the model. Scores by content so tests can predict marks."""

    def __init__(self, reply_for=None):
        self.calls = []            # every message list it was asked to judge
        self.reply_for = reply_for or (lambda cand: "Relevance: 2 marks\nEquivalence: 1 mark\nSpecificity: 1 mark\nOverall: 4 marks")

    def __call__(self, message_batches):
        out = []
        for msgs in message_batches:
            self.calls.append(msgs)
            final = msgs[-1]["content"]
            cand = final.split("Candidate explanation:", 1)[1].strip()
            out.append(self.reply_for(cand))
        return out


def _dataset():
    """4 images. rule_1: 2 TPs, rule_2: 1 TP, rule_3: a false positive only (no TP),
    rule_4: no TP. One prediction is a parse failure (None)."""
    gts = [
        _gt("img0", rule_1=_v("The worker on the left has no hard hat.")),
        _gt("img1", rule_1=_v("The man in blue has no hard hat."), rule_2=_v("The roofer has no harness.")),
        _gt("img2"),                                         # safe
        _gt("img3", rule_4=_v("The worker is in the excavator's blind spot.")),
    ]
    preds = [
        _pred(rule_1=_v("GOOD worker on the left without a hard hat")),
        _pred(rule_1=_v("BAD something is wrong"), rule_2=_v("GOOD roofer lacks a harness")),
        _pred(rule_3=_v("FP an edge is unguarded")),         # false positive: not judged
        None,                                                # parse failure: not judged
    ]
    return preds, gts


def _scores_by_prefix(cand):
    if cand.startswith("GOOD"):
        return "Relevance: 2 marks\nEquivalence: 2 marks\nSpecificity: 2 marks\nOverall: 6 marks"
    if cand.startswith("BAD"):
        return "Relevance: 0 marks\nEquivalence: 0 marks\nSpecificity: 1 mark\nOverall: 1 mark"
    return "I refuse to answer in the format."


# ---------------------------------------------------------------------------
# 1. Rule text: one source of truth, training prompt untouched
# ---------------------------------------------------------------------------

def test_training_prompts_unchanged_by_the_rule_text_refactor():
    """SAFETY_RULE_TEXTS was lifted out of _SAFETY_RULES so the judge can read one
    rule. The joined text feeds every unified/violations_only training and inference
    prompt, so it must stay byte-identical. These hashes were taken immediately before
    the refactor; a deliberate prompt edit must update them consciously."""
    from data import prompt_templates as P
    assert hashlib.sha256(P._SAFETY_RULES.encode()).hexdigest() == \
        "aa12be6ca92799f8b0e81a77d66c2d419136fff15bb87ddf574addbe0917c924"
    assert hashlib.sha256(P.VIOLATIONS_ONLY_PROMPT.encode()).hexdigest() == \
        "d6aa99afc1664be675831a496c82ee8855c6a21e3028e389d3be5ecd89a677f1"
    assert hashlib.sha256(P.UNIFIED_INSPECTION_PROMPT.encode()).hexdigest() == \
        "125e312d268c1c77864485f58c0071c649f38f207571fe5d016f6d7723635df1"
    assert "".join(P.SAFETY_RULE_TEXTS[r] for r in RULES) == P._SAFETY_RULES


def test_rule_description_is_the_prompt_wording_without_list_formatting():
    from data.prompt_templates import SAFETY_RULE_TEXTS, safety_rule_description
    for r in RULES:
        d = safety_rule_description(r)
        assert d.startswith(f"Rule {r[-1]} - ")
        assert not d.endswith("\n") and "   - " not in d
        assert d in SAFETY_RULE_TEXTS[r]


# ---------------------------------------------------------------------------
# 2. Few-shot blocks
# ---------------------------------------------------------------------------

def test_exactly_three_fewshot_examples_per_rule():
    assert set(FEWSHOT_EXAMPLES) == set(RULES)
    for r in RULES:
        assert len(FEWSHOT_EXAMPLES[r]) == 3, r


def test_fewshot_marks_are_integers_in_range_and_every_level_is_anchored():
    seen = {c: set() for c in JUDGE_CRITERIA}
    for exs in FEWSHOT_EXAMPLES.values():
        for e in exs:
            for c in JUDGE_CRITERIA:
                v = getattr(e, c)
                assert isinstance(v, int) and 0 <= v <= 2
                seen[c].add(v)
    # A mark level that no example demonstrates is a mark level the judge must guess at.
    for c, levels in seen.items():
        assert levels == {0, 1, 2}, f"{c} never demonstrates marks {sorted({0,1,2} - levels)}"


# The paper's five published, human-judged worked examples (Fig. 8 and the appendix
# figure). They anchor the scale so our judge is comparable to the paper's Table 8.
PAPER_ANCHORS = [
    ("rule_1", "Worker with a black cap and white shirt on the left is not wearing a hard hat.",
     "The worker on the left is not wearing a hard hat, and his clothes do not cover his shoulders.", (2, 2, 2)),
    ("rule_1", "The worker with a camouflage uniform is not wearing a hard hat.",
     "The construction worker is not wearing any visible PPE, such as a hard hat or high-visibility vest.", (2, 1, 1)),
    ("rule_2", "The person with a blue t-shirt and standing on top of the scaffold is not wearing a safety harness.",
     "The worker on the scaffold is not using a safety harness while working at a height.", (2, 2, 1)),
    ("rule_3", "The edge of the excavation in the lower right is not protected.",
     "There are no workers visible in the image who are at a height of three meters or more.", (0, 0, 1)),
    ("rule_4", "The worker with a white hard hat is too close to the excavator and is in the blind spot.",
     "A worker is in close proximity to an excavator, potentially in the operator's blind spot.", (2, 2, 2)),
]


@pytest.mark.parametrize("rule,reference,candidate,marks", PAPER_ANCHORS)
def test_paper_anchor_examples_present_verbatim_with_the_papers_scores(rule, reference, candidate, marks):
    matches = [e for e in FEWSHOT_EXAMPLES[rule] if e.reference == reference and e.candidate == candidate]
    assert len(matches) == 1, f"paper anchor missing from {rule}: {reference!r}"
    e = matches[0]
    assert (e.relevance, e.equivalence, e.specificity) == marks
    assert e.source.startswith("paper:")


def test_every_fewshot_example_declares_its_provenance():
    n_paper = n_authored = 0
    for exs in FEWSHOT_EXAMPLES.values():
        for e in exs:
            if e.source.startswith("paper:"):
                n_paper += 1
            else:
                # Authored examples take their reference from the TRAIN split (the paper
                # samples few-shot examples from train to avoid test leakage).
                assert re.fullmatch(r"authored:train/\d{7}", e.source), e.source
                n_authored += 1
    assert (n_paper, n_authored) == (5, 7)


# ---------------------------------------------------------------------------
# 3. Prompt
# ---------------------------------------------------------------------------

def test_system_prompt_carries_the_papers_rubric_verbatim():
    for c in JUDGE_CRITERIA:
        assert CRITERIA_DEFINITIONS[c] in JUDGE_SYSTEM_PROMPT
    assert SCALE_DEFINITION in JUDGE_SYSTEM_PROMPT
    assert "Overall" in JUDGE_SYSTEM_PROMPT


@pytest.mark.parametrize("rule", RULES)
def test_messages_are_system_then_three_shots_then_the_item(rule):
    from data.prompt_templates import safety_rule_description
    msgs = build_judge_messages(rule, "REFERENCE TEXT", "CANDIDATE TEXT")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    final = msgs[-1]["content"]
    assert safety_rule_description(rule) in final
    assert "REFERENCE TEXT" in final and "CANDIDATE TEXT" in final
    # The rule text is the item's own rule, never another rule's.
    for other in RULES:
        if other != rule:
            assert safety_rule_description(other) not in final


@pytest.mark.parametrize("rule", RULES)
def test_every_fewshot_reply_parses_back_to_its_own_marks(rule):
    """The reply format we demonstrate to the judge must be one our own parser reads."""
    msgs = build_judge_messages(rule, "r", "c")
    replies = [m["content"] for m in msgs if m["role"] == "assistant"]
    for reply, ex in zip(replies, FEWSHOT_EXAMPLES[rule]):
        p = parse_judge_output(reply)
        assert p is not None and not p.clamped and not p.overall_mismatch
        assert (p.relevance, p.equivalence, p.specificity) == (ex.relevance, ex.equivalence, ex.specificity)


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        build_judge_messages("rule_0", "r", "c")


def test_reply_format_matches_the_papers_figures():
    assert format_judge_reply(2, 1, 1) == (
        "Relevance: 2 marks\nEquivalence: 1 mark\nSpecificity: 1 mark\nOverall: 4 marks"
    )


def test_rubric_fingerprint_is_stable_and_sensitive():
    a = rubric_fingerprint()
    assert a == rubric_fingerprint() and len(a) == 64
    ex = FEWSHOT_EXAMPLES["rule_1"][0]
    tweaked = dict(FEWSHOT_EXAMPLES)
    tweaked["rule_1"] = (J.JudgeExample(ex.reference, ex.candidate, 1, ex.equivalence, ex.specificity, ex.source),) \
        + FEWSHOT_EXAMPLES["rule_1"][1:]
    with patch.object(J, "FEWSHOT_EXAMPLES", tweaked):
        assert rubric_fingerprint() != a


# ---------------------------------------------------------------------------
# 4. Parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,marks", [
    ("Relevance: 2 marks\nEquivalence: 1 mark\nSpecificity: 1 mark\nOverall: 4 marks", (2, 1, 1)),
    ("Relavance: 2 marks\nEquivalence: 2 marks\nSpecificity: 2 marks\nOverall: 6 marks", (2, 2, 2)),  # paper's typo
    ("**Relevance:** 2\n**Equivalence:** 0\n**Specificity:** 1", (2, 0, 1)),
    ("Sure.\nRelevance = 1 mark, Equivalence - 1, Specificity: 2 marks. Overall: 4", (1, 1, 2)),
    ("relevance: 0\nEQUIVALENCE: 0\nSpecificity: 0", (0, 0, 0)),
])
def test_parser_accepts_format_variants(text, marks):
    p = parse_judge_output(text)
    assert p is not None
    assert (p.relevance, p.equivalence, p.specificity) == marks
    assert p.total == sum(marks)


@pytest.mark.parametrize("text", [
    None, "", "   ",
    "Relevance: 2\nEquivalence: 2",                         # a criterion missing
    "Relevance: 1.5\nEquivalence: 1\nSpecificity: 1",       # not an integer mark
    "The explanation is good.",
])
def test_parser_rejects_unusable_replies(text):
    assert parse_judge_output(text) is None


def test_parser_clamps_out_of_range_marks_and_flags_it():
    p = parse_judge_output("Relevance: 3\nEquivalence: 1\nSpecificity: -1")
    assert (p.relevance, p.equivalence, p.specificity) == (2, 1, 0)
    assert p.clamped


def test_total_is_the_sum_of_criteria_not_the_reported_overall_line():
    p = parse_judge_output("Relevance: 2\nEquivalence: 2\nSpecificity: 2\nOverall: 5 marks")
    assert p.total == 6 and p.overall_mismatch


# ---------------------------------------------------------------------------
# 5. Scoring and aggregation (fake generation)
# ---------------------------------------------------------------------------

def test_scores_aggregate_per_rule_micro_and_macro():
    preds, gts = _dataset()
    fake = FakeJudge(_scores_by_prefix)
    m, details, status = score_reasoning_with_llm_judge(preds, gts, cfg={}, generate_fn=fake)

    # rule_1: GOOD (2,2,2) + BAD (0,0,1) -> means (1, 1, 1.5), total 3.5
    assert m[f"{KEY_PREFIX}scored_count_rule_1"] == 2
    assert m[f"{KEY_PREFIX}relevance_rule_1"] == 1.0
    assert m[f"{KEY_PREFIX}specificity_rule_1"] == 1.5
    assert m[f"{KEY_PREFIX}total_rule_1"] == 3.5
    # rule_2: one GOOD
    assert m[f"{KEY_PREFIX}total_rule_2"] == 6.0
    # rule_3 had only a false positive, rule_4 only a parse failure: nothing judged.
    for r in ("rule_3", "rule_4"):
        assert m[f"{KEY_PREFIX}scored_count_{r}"] == 0
        assert m[f"{KEY_PREFIX}total_{r}"] == 0.0

    # micro pools all 3 items: totals 6, 1, 6
    assert m[f"{KEY_PREFIX}scored_count_micro"] == 3
    assert m[f"{KEY_PREFIX}total_micro"] == pytest.approx(13 / 3)
    # macro averages only the MEASURED rules: (3.5 + 6) / 2, not / 4
    assert m[f"{KEY_PREFIX}total_macro"] == pytest.approx(4.75)
    assert m[f"{KEY_PREFIX}total_macro_n_rules"] == 2
    for c in JUDGE_CRITERIA:
        assert m[f"{KEY_PREFIX}{c}_macro_n_rules"] == 2

    # every false positive / parse failure stayed out of the judge entirely
    assert len(fake.calls) == 3
    assert not any("FP an edge" in msgs[-1]["content"] for msgs in fake.calls)
    assert status["status"] == "ok" and status["n_items"] == 3
    assert sum(status["total_histogram"].values()) == 3
    assert m[f"{KEY_PREFIX}model_id"] == status["model_id"]


def test_scored_counts_equal_the_text_similarity_suite_on_the_same_input():
    """Both families score the list from collect_tp_reason_pairs; their counts can
    never diverge. This is what makes the two directly comparable per rule."""
    from evaluation.metrics_reasoning import batch_score_reasoning
    preds, gts = _dataset()

    def fake_caption_metrics(p, g, images=None, include_spice=False):
        return {"bertscore_f1": 0.5, "meteor": 0.5, "ciderd": 1.0, "clipscore": 0.5,
                "scored_count": len(p), "total_count": len(p)}

    with patch("evaluation.metrics_reasoning.compute_all_caption_metrics", side_effect=fake_caption_metrics):
        text = batch_score_reasoning(preds, gts, images=[None] * len(preds))
    judge, _, _ = score_reasoning_with_llm_judge(preds, gts, cfg={}, generate_fn=FakeJudge())
    for r in RULES:
        assert judge[f"{KEY_PREFIX}scored_count_{r}"] == text[f"reasoning_text_similarity_scored_count_{r}"], r


def test_unparseable_reply_scores_zero_and_is_counted():
    preds, gts = _dataset()
    m, details, status = score_reasoning_with_llm_judge(
        preds, gts, cfg={}, generate_fn=FakeJudge(lambda cand: "no marks here"))
    assert m[f"{KEY_PREFIX}unparsed_count_micro"] == 3
    assert m[f"{KEY_PREFIX}unparsed_rate_micro"] == 1.0
    assert m[f"{KEY_PREFIX}total_micro"] == 0.0
    assert all(d["status"] == "unparsed" and d["raw_output"] == "no marks here" for d in details)
    assert status["n_unparsed"] == 3


def test_empty_candidate_scores_zero_without_asking_the_model():
    gts = [_gt("a", rule_1=_v("The worker has no hard hat."))]
    preds = [_pred(rule_1={"reason": "", "bounding_box": [[1, 1, 5, 5]]})]
    fake = FakeJudge()
    m, details, _ = score_reasoning_with_llm_judge(preds, gts, cfg={}, generate_fn=fake)
    assert fake.calls == []
    assert details[0]["status"] == "empty_candidate"
    assert m[f"{KEY_PREFIX}scored_count_rule_1"] == 1
    assert m[f"{KEY_PREFIX}total_rule_1"] == 0.0
    assert m[f"{KEY_PREFIX}empty_candidate_count_micro"] == 1


def test_no_true_positives_never_loads_the_model():
    gts = [_gt("a")]
    preds = [_pred(rule_2=_v("a false alarm"))]
    with patch.object(J, "make_hf_generate_fn", side_effect=AssertionError("model must not load")):
        m, details, status = score_reasoning_with_llm_judge(preds, gts, cfg={}, generate_fn=None)
    assert details == [] and status["n_model_calls"] == 0
    assert m[f"{KEY_PREFIX}total_macro"] == 0.0
    assert m[f"{KEY_PREFIX}total_macro_n_rules"] == 0


def test_batch_size_does_not_change_scores():
    preds, gts = _dataset()
    a, _, _ = score_reasoning_with_llm_judge(preds, gts, cfg={"batch_size": 1}, generate_fn=FakeJudge(_scores_by_prefix))
    b, _, _ = score_reasoning_with_llm_judge(preds, gts, cfg={"batch_size": 2}, generate_fn=FakeJudge(_scores_by_prefix))
    assert a == b


def test_details_record_everything_needed_to_audit_a_judgment():
    preds, gts = _dataset()
    _, details, _ = score_reasoning_with_llm_judge(preds, gts, cfg={}, generate_fn=FakeJudge(_scores_by_prefix))
    first = details[0]
    assert set(first) >= {"image_id", "index", "rule", "reference", "candidate", "raw_output", "status", "scores"}
    assert first["image_id"] == "img0" and first["rule"] == "rule_1"
    assert first["reference"] == "The worker on the left has no hard hat."
    assert first["scores"] == {"relevance": 2, "equivalence": 2, "specificity": 2, "total": 6}


# ---------------------------------------------------------------------------
# 6. Failure handling
# ---------------------------------------------------------------------------

def test_judge_failure_is_soft_by_default():
    preds, gts = _dataset()

    def broken(_):
        raise RuntimeError("CUDA out of memory")

    metrics, details, status = run_llm_judge(preds, gts, cfg={}, generate_fn=broken)
    assert metrics == {} and details == []
    assert status["status"] == "failed" and "out of memory" in status["error"]


def test_judge_failure_is_fatal_when_fail_hard():
    preds, gts = _dataset()
    with pytest.raises(RuntimeError):
        run_llm_judge(preds, gts, cfg={"fail_hard": True},
                      generate_fn=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))


def test_reply_count_mismatch_is_an_error_not_a_silent_misalignment():
    preds, gts = _dataset()
    metrics, _, status = run_llm_judge(preds, gts, cfg={"batch_size": 2}, generate_fn=lambda batch: ["x"])
    assert metrics == {} and status["status"] == "failed"


def test_real_loader_refuses_cpu_and_the_wrapper_degrades_cleanly():
    preds, gts = _dataset()
    J._JUDGE_MODEL = None
    with patch("torch.cuda.is_available", return_value=False):
        metrics, _, status = run_llm_judge(preds, gts, cfg={})
    assert metrics == {}
    assert status["status"] == "failed" and "GPU" in status["error"]


# ---------------------------------------------------------------------------
# 7. Evaluator gating
# ---------------------------------------------------------------------------

def _raw(obj):
    import json
    return "```json\n" + json.dumps(obj) + "\n```"


def test_evaluator_omits_judge_keys_when_the_flag_is_off():
    from PIL import Image
    from evaluation.evaluator import run_full_evaluation
    preds, gts = _dataset()
    raw = [_raw(p) if p else "garbage" for p in preds]
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True), \
         patch("evaluation.metrics_reasoning.compute_all_caption_metrics", return_value={"scored_count": 1}), \
         patch("evaluation.metrics_llm_judge.run_llm_judge", side_effect=AssertionError("must not run")):
        res = run_full_evaluation(raw, gts, images=[Image.new("RGB", (4, 4))] * 4, task="violations_only")
    assert not any(k.startswith(KEY_PREFIX) for k in res["metrics"])
    assert res["llm_judge_status"] is None and res["llm_judge_details"] is None


def test_evaluator_adds_judge_keys_alongside_existing_reasoning_metrics():
    from PIL import Image
    from evaluation.evaluator import run_full_evaluation
    preds, gts = _dataset()
    raw = [_raw(p) if p else "garbage" for p in preds]

    def judged(pv, gv):
        return score_reasoning_with_llm_judge(pv, gv, cfg={}, generate_fn=FakeJudge(_scores_by_prefix))

    with patch("evaluation.metrics_captioning._check_java_available", return_value=True), \
         patch("evaluation.metrics_reasoning.compute_all_caption_metrics",
               side_effect=lambda p, g, images=None, include_spice=False: {"bertscore_f1": 0.5, "scored_count": len(p)}), \
         patch("evaluation.metrics_llm_judge.run_llm_judge", side_effect=judged):
        res = run_full_evaluation(raw, gts, images=[Image.new("RGB", (4, 4))] * 4,
                                  task="violations_only", use_llm_judge=True)
    m = res["metrics"]
    assert m[f"{KEY_PREFIX}total_rule_2"] == 6.0
    # the existing families are still there, untouched
    assert "reasoning_text_similarity_bertscore_f1_macro" in m
    assert "violation_identification_f1_micro" in m
    assert res["llm_judge_status"]["status"] == "ok"


@pytest.mark.parametrize("task", ["object_only", "caption_only"])
def test_evaluator_skips_the_judge_silently_for_tasks_without_violations(task):
    from evaluation.evaluator import run_full_evaluation
    raw = [_raw({"excavator": [], "rebar": [], "worker_with_white_hard_hat": []})] if task == "object_only" \
        else ["A quiet construction site."]
    refs = [{"image_id": "a", "caption": "A site.", "excavator": [], "rebar": [], "worker_with_white_hard_hat": []}]
    from PIL import Image
    with patch("evaluation.metrics_captioning._check_java_available", return_value=True), \
         patch("evaluation.evaluator.compute_all_caption_metrics", return_value={}), \
         patch("evaluation.metrics_llm_judge.run_llm_judge", side_effect=AssertionError("must not run")):
        res = run_full_evaluation(raw, refs, images=[Image.new("RGB", (4, 4))], task=task, use_llm_judge=True)
    assert not any(k.startswith(KEY_PREFIX) for k in res["metrics"])
    assert res["llm_judge_status"] is None


# ---------------------------------------------------------------------------
# 8. Wiring: CLI, config, phase scripts, comparison tooling
# ---------------------------------------------------------------------------

def test_config_replicates_the_papers_decoding():
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))["llm_judge"]
    assert cfg["model_id"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg["num_beams"] == 5
    assert cfg["seed"] == 20
    resolved = J.resolve_judge_config()
    assert resolved["num_beams"] == 5 and resolved["seed"] == 20


def test_generation_is_deterministic_beam_search_in_source():
    """The real generate path cannot run here; pin the load-bearing lines instead."""
    src = (REPO / "evaluation" / "metrics_llm_judge.py").read_text(encoding="utf-8")
    assert "do_sample=False" in src
    assert 'num_beams=int(cfg["num_beams"])' in src
    assert "set_seed(seed)" in src
    assert "local_files_only=True" in src
    assert 'padding_side = "left"' in src
    assert "add_special_tokens=False" in src      # chat template already adds <|begin_of_text|>
    assert "<|eot_id|>" in src                    # Llama 3 turn terminator


def test_run_evaluation_exposes_the_flag_and_threads_it_through():
    src = (REPO / "experiments" / "run_evaluation.py").read_text(encoding="utf-8")
    assert '"--use_llm_judge", action="store_true"' in src
    assert "use_llm_judge=args.use_llm_judge" in src
    assert "llm_judge_details.json" in src and "llm_judge_status.json" in src


def test_run_evaluation_logs_only_numeric_values_to_wandb():
    """String provenance keys (the base64 outcome vector, the judge model id) must not
    reach wandb.log: that call runs inside the SFT/GRPO jobs after metrics.json is
    written, and an exception there would block every afterok dependent."""
    src = (REPO / "experiments" / "run_evaluation.py").read_text(encoding="utf-8")
    assert "if not isinstance(v, str)" in src


@pytest.mark.parametrize("script", ["hpc_baseline.sh", "hpc_sft.sh", "hpc_grpo.sh"])
def test_phase_scripts_request_the_judge_in_their_eval_step(script):
    text = (REPO / "scripts" / script).read_text(encoding="utf-8")
    eval_block = text[text.index("python -m experiments.run_evaluation"):]
    assert "--use_llm_judge" in eval_block.split('--task "$TASK"')[0]


def test_comparison_tooling_surfaces_the_judge():
    from experiments.results_lib import (
        MACRO_N_RULES_KEY, SUPPORT_KEYS, UNBOUNDED_HEADLINE_KEYS, BOUNDED_HEADLINE_KEYS,
    )
    from experiments.results_charts import METRIC_Y_CEILING, PER_RULE_METRIC_SPECS
    key = f"{KEY_PREFIX}total_macro"
    assert key in UNBOUNDED_HEADLINE_KEYS["reasoning"]
    assert key not in BOUNDED_HEADLINE_KEYS["reasoning"], "0-6 must never share the 0-1 axis"
    assert MACRO_N_RULES_KEY[key] == f"{KEY_PREFIX}total_macro_n_rules"
    assert f"{KEY_PREFIX}scored_count_rule_1" in SUPPORT_KEYS["reasoning"]
    assert METRIC_Y_CEILING[key] == 6
    judge_specs = {s[2]: s[6] for s in PER_RULE_METRIC_SPECS if s[2].startswith(KEY_PREFIX)}
    assert judge_specs[f"{KEY_PREFIX}total_{{rule}}"] == 6
    for c in JUDGE_CRITERIA:
        assert judge_specs[f"{KEY_PREFIX}{c}_{{rule}}"] == 2


def test_win_tally_excludes_divisors_rates_and_failures_but_keeps_judge_scores():
    from experiments.results_lib import is_non_comparable_key
    for k in [f"{KEY_PREFIX}total_macro_n_rules", f"{KEY_PREFIX}unparsed_rate_micro",
              f"{KEY_PREFIX}scored_count_rule_1", "violation_pred_positive_rate",
              "violation_gt_positive_rate", "violation_prediction_failure_rate",
              "reasoning_text_similarity_bertscore_f1_macro_n_rules"]:
        assert is_non_comparable_key(k), k
    for k in [f"{KEY_PREFIX}total_macro", f"{KEY_PREFIX}relevance_rule_1",
              "violation_identification_f1_micro"]:
        assert not is_non_comparable_key(k), k


def test_judge_charts_render_on_a_fixed_axis_and_only_when_present(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from experiments.results_charts import chart_per_rule_metrics, _bar

    cols = {("violations_only", "2b", p, "v2") for p in ("baseline", "sft", "grpo")}
    lut = {}
    written = chart_per_rule_metrics(lut, cols, tmp_path)
    assert not any("judge_" in p.name for p in written), "no judge keys -> no judge charts"

    for c in cols:
        for r in RULES:
            lut[(*c, f"{KEY_PREFIX}total_{r}")] = 4.2
            for crit in JUDGE_CRITERIA:
                lut[(*c, f"{KEY_PREFIX}{crit}_{r}")] = 1.4
    written = chart_per_rule_metrics(lut, cols, tmp_path)
    judge = [p for p in written if "judge_" in p.name]
    assert judge and all(p.exists() for p in judge)

    captured = {}
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    def spy(*a, **k):
        fig, ax = real_subplots(*a, **k)
        captured["ax"] = ax
        return fig, ax

    with patch("experiments.results_charts.plt.subplots", side_effect=spy):
        _bar(["rule_1"], [("SFT", [4.2])], "t", "y", tmp_path / "x.png", y_bounded=6)
    assert captured["ax"].get_ylim() == pytest.approx((0, 6.3))
