"""
Shared utility module for GRPO reward functions.

This module provides common parsing, validation, embedding, and similarity computation
functions used across different reward components for GRPO training.
"""
import json
import math
import re
import functools
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

import torch

from evaluation.output_parser import strip_fences
from data.schemas import UnifiedOutput
from data.box_utils import normalize_boxes, clean_boxes, scale_1000_to_01, compute_mask_union_iou
from core.constants import GROUNDING_CLASSES, RULES
from core.logging import get_logger

logger = get_logger(__name__)

# Global singleton for embedding model
_EMBED_MODEL = None

import copy

@functools.lru_cache(maxsize=128)
def _strict_parse_cached(text: str) -> Optional[dict]:
    """
    Strictly parse JSON from text and cache the result.
    """
    try:
        from evaluation.output_parser import parse_model_output, validate_unified_output
        parsed = parse_model_output(text)
        if parsed is None or not isinstance(parsed, dict):
            return None
        
        # Validate using Pydantic schema via output_parser
        if validate_unified_output(parsed) is None:
            return None
            
        return parsed
    except Exception as e:
        return None

def _strict_parse(text: Union[str, List[Dict]]) -> Optional[dict]:
    """
    Wrapper around cached parser to ensure we never return a mutable reference 
    from the cache, which could corrupt downstream reward functions.
    """
    if isinstance(text, list):
        text = text[-1].get("content", "") if text else ""
    elif not isinstance(text, str):
        text = str(text)
        
    res = _strict_parse_cached(text)
    return copy.deepcopy(res) if res is not None else None

# ---------------------------------------------------------------------------
# Tunable reward constants, read from the task YAML
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def reward_constant(task: str, key: str, default: float) -> Any:
    """Reads a tunable reward constant from configs/tasks/<task>.yaml.

    Cached, because reward functions are called once per completion per step and
    a YAML read per call would dominate the reward budget.

    Returns ``default`` when the key is absent, so a task YAML that says nothing
    keeps exactly the behaviour it had before these knobs existed.
    """
    from core.config import load_task_config
    try:
        cfg = load_task_config(task)
    except Exception:
        return default
    value = cfg.get(key)
    return default if value is None else value


def grounding_tn_constant(task: str, cls: str) -> float:
    """Per-class true-negative credit for object grounding.

    Accepts either a scalar (applies to every class) or a per-class mapping in the
    task YAML under ``grounding_tn_constant``. Defaults to 0.15, the historical
    global value, so `unified` is unchanged unless it opts in.

    WHY THIS IS TUNABLE. The true-negative credit sets the break-even detection
    quality for a class: emitting boxes for class k is positive-EV only when

        p_k * E[IoU_k]  >  c * (1 - p_k)        i.e.   E[IoU_k] > c * (1 - p_k) / p_k

    where p_k is that class's prevalence in the GRPO pool. At the historical flat
    c = 0.15 and the measured pool prevalences (excavator 0.361, rebar 0.088,
    worker_with_white_hard_hat 0.115) the break-even IoUs are 0.27, 1.55 and 1.15.
    Two of those exceed 1.0, so *never emitting those classes is strictly dominant
    regardless of how good the detector is* — the reward actively trains the model
    to stop detecting them. See scripts/validate_rewards.py, which fails the build
    if any class's break-even exceeds a configured ceiling.
    """
    value = reward_constant(task, "grounding_tn_constant", 0.15)
    if isinstance(value, dict):
        return float(value.get(cls, value.get("default", 0.15)))
    return float(value)


def _is_substantive_violation(v: Any) -> bool:
    """True if a PREDICTED violation carries any actual content.

    A violation object is *present* (see _is_violation_present) as soon as it
    exists at all — that is the prompt contract and it is deliberately generous.
    But presence alone should not earn full identification credit, because
    ``{"reason": "", "bounding_box": []}`` is a contentless assertion: it names no
    location and gives no justification, yet it scored a perfect F-beta = 1.0 on
    reward_violation_id, the most heavily weighted component of the
    violations_only task (0.40). Grounding and reasoning are TP-conditioned, so
    neither of them penalised it either.

    Substance = a non-empty reason OR at least one bounding box. Ground truth is
    always substantive, so this is only ever applied to predictions.
    """
    if not _is_violation_present(v):
        return False
    if not isinstance(v, dict):
        # A bare `true` asserts a violation with no content whatsoever.
        return False
    reason = v.get("reason")
    has_reason = isinstance(reason, str) and bool(reason.strip())
    boxes = v.get("bounding_box")
    has_box = bool(boxes) if isinstance(boxes, (list, tuple)) else False
    return has_reason or has_box


def _is_violation_present(v: Any) -> bool:
    """
    Determine if a violation value represents an actual violation.

    SINGLE SOURCE OF TRUTH — used by every GRPO reward and by
    evaluation/metrics_{violations,reasoning}.py, so training and evaluation can
    never disagree about what counts as a violation.

    The rule follows the prompt contract literally: the prompt says "If NOT violated,
    output null", so **only null (or an absent key) means safe**. Emitting a violation
    object at all is the model asserting a violation, even if it then fails to say
    where or why.

    That matters because a bare `true` is rewritten by structural_repair.py:959 into
    {"reason": "", "bounding_box": []}. Treating that shape as safe would invert the
    model's own answer — crediting an unsubstantiated alarm as "correctly identified a
    safe site". It also left a small reward-hacking surface: an empty violation object
    scored as safe collected the true-negative reward on safe images.

    A contentless assertion still earns nothing downstream — grounding IoU against an
    empty box list is 0.0, and an empty reason scores ~0 — so the model gets credit for
    exactly what it asserted and nothing for what it withheld.

    A fully empty dict {} is the one exception: it carries no keys and therefore no
    assertion. structural_repair.py:1007 already normalizes it to null.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        # {} carries no assertion; any keyed violation object does.
        return bool(v)
    if isinstance(v, str):
        stripped = v.strip().lower()
        if stripped in ("false", "none", "null", "n/a", ""):
            return False
        return True
    return bool(v)

def _get_embed_model():
    """Lazy singleton loader for SentenceTransformer, forced to CPU to save VRAM."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # Force CPU to avoid VRAM conflicts with the training model
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _EMBED_MODEL

def _embed_texts(texts: List[str]) -> torch.Tensor:
    """Batch-embeds texts using the singleton model. Returns normalized embeddings."""
    if not texts:
        return torch.empty((0, 384))
    model = _get_embed_model()
    # output shape: (N, D)
    embeddings = model.encode(texts, convert_to_tensor=True, device="cpu")
    # Normalize for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D normalized embeddings."""
    if a.dim() == 1 and b.dim() == 1:
        return torch.dot(a, b).item()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def _cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity across two batches of embeddings (shape: N x D)."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)

def _get_ngrams(tokens: List[str], n: int) -> set:
    """Helper to extract n-grams from a list of tokens."""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _ngram_f1(pred: str, ref: str, n_range: tuple = (1, 2)) -> float:
    """Computes unigram+bigram set F1 between pred and ref."""
    if not pred or not ref:
        return 0.0
    
    pred_tokens = re.findall(r'\w+', pred.lower())
    ref_tokens = re.findall(r'\w+', ref.lower())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    pred_ngrams = set()
    ref_ngrams = set()
    
    for n in range(n_range[0], n_range[1] + 1):
        pred_ngrams.update(_get_ngrams(pred_tokens, n))
        ref_ngrams.update(_get_ngrams(ref_tokens, n))
        
    if not pred_ngrams or not ref_ngrams:
        return 0.0
        
    intersection = pred_ngrams & ref_ngrams
    if not intersection:
        return 0.0
        
    precision = len(intersection) / len(pred_ngrams)
    recall = len(intersection) / len(ref_ngrams)
    
    return 2 * precision * recall / (precision + recall)

def _has_repetition_pathology(parsed: dict, threshold: int = 5) -> bool:
    """Detects repeated boxes indicating a generation loop artifact."""
    all_boxes = []
    
    # Check GROUNDING_CLASSES
    for cls in GROUNDING_CLASSES:
        if cls in parsed:
            val = parsed[cls]
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, list):
                        all_boxes.append(tuple(item))
            elif isinstance(val, dict):
                boxes = val.get("bounding_box", [])
                if isinstance(boxes, list):
                    for b in boxes:
                        if isinstance(b, list):
                            all_boxes.append(tuple(b))

    # Check RULES (violation fields use f"{rule}_violation" key pattern)
    for rule in RULES:
        key = f"{rule}_violation"
        val = parsed.get(key)
        if isinstance(val, dict):
            boxes = val.get("bounding_box", [])
            if isinstance(boxes, list):
                for b in boxes:
                    if isinstance(b, list):
                        all_boxes.append(tuple(b))
                            
    if not all_boxes:
        return False
        
    counts = Counter(all_boxes)
    if counts.most_common(1)[0][1] > threshold:
        return True
        
    return False

def _safe_reward(fn):
    """Decorator that wraps a reward function, returning 0.0 on exception."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback
            logger.warning(
                f"Error in reward function {fn.__name__}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return 0.0
    return wrapper

def _safe_batch_reward(fn):
    """Decorator that wraps a batched reward function, returning [0.0]*N on exception."""
    @functools.wraps(fn)
    def wrapper(completions, ground_truths, *args, **kwargs):
        try:
            return fn(completions, ground_truths, *args, **kwargs)
        except Exception as e:
            import traceback
            logger.warning(
                f"Error in batch reward function {fn.__name__}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return [0.0] * len(completions)
    return wrapper

# ---------------------------------------------------------------------------
# Task-aware strict parsing
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def _strict_parse_with_schema_cached(text: str, schema_name: str) -> Optional[dict]:
    """Parse JSON and validate against the schema identified by schema_name.
    
    Unlike _strict_parse_cached (which always validates against UnifiedOutput),
    this function uses the schema registry to validate against the correct
    schema for the given task, and parse_output_for_task so that plain-text tasks
    (caption_only) are parsed with their own contract rather than as JSON.
    """
    from evaluation.output_parser import parse_output_for_task, validate_output_for_task

    # Resolved outside the try below: an unregistered task must raise, not read as
    # a schema failure.
    from core.tasks import get_task_spec
    get_task_spec(schema_name)

    try:
        parsed = parse_output_for_task(text, task=schema_name)
        if parsed is None or not isinstance(parsed, dict):
            return None
        # Validate against the appropriate schema
        if validate_output_for_task(parsed, task=schema_name) is None:
            return None
        return parsed
    except Exception:
        return None


def _strict_parse_for_task(text: Union[str, List[Dict]], task: str = "unified") -> Optional[dict]:
    """Task-aware wrapper around cached parser.
    
    For task='unified', produces identical results to _strict_parse().
    For every other task, parses with that task's wire format and validates
    against that task's schema from data/schemas.py::SCHEMA_REGISTRY.
    """
    if isinstance(text, list):
        text = text[-1].get("content", "") if text else ""
    elif not isinstance(text, str):
        text = str(text)
    res = _strict_parse_with_schema_cached(text, task)
    return copy.deepcopy(res) if res is not None else None
