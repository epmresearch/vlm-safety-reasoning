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
        stripped = strip_fences(text)
        if not stripped:
            return None
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            return None
        
        # Validate using Pydantic schema
        UnifiedOutput(**parsed)
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

def _is_violation_present(v: Any) -> bool:
    """
    Determine if a violation value represents an actual violation.
    Handles various formats: None, empty dict, falsey dicts, bool, string, etc.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        if not v:
            return False
        # Check if violation dict has meaningful content
        boxes = v.get("bounding_box")
        has_boxes = isinstance(boxes, list) and len(boxes) > 0
        reason = v.get("reason")
        has_reason = isinstance(reason, str) and bool(reason.strip())
        return has_boxes or has_reason
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

def _has_repetition_pathology(parsed: dict, threshold: int = 3) -> bool:
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
