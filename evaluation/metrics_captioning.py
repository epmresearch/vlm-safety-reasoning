"""
Metrics for image captioning evaluation.

CIDEr-D, METEOR, and SPICE use the OFFICIAL pycocoevalcap toolkit
(Java-based PTBTokenizer + Java jars).
Requires: `apt-get install -y default-jre` in the Colab session before use.
Only needed for evaluation notebooks — not for training/inference notebooks.
"""

import os
import re
import subprocess
from contextlib import contextmanager

_SPICE_CACHE_NOISE_PATTERN = re.compile(
    r"Could not cache item to.*|Caption may be too long", re.IGNORECASE
)


@contextmanager
def _suppress_spice_cache_noise():
    """Filters SPICE's jar-internal 'Could not cache item ... Caption may be
    too long' lines out of the subprocess output.

    Root cause: SPICE's bundled jar (spice-1.0.jar) uses the raw caption text
    as the cache filename inside its `cache/` dir (instead of hashing it), and
    Linux filesystems cap filenames at 255 bytes. Our captions routinely
    exceed that, so every such caption prints this two-line warning. It only
    means that ONE caption's cache entry was skipped -- the SPICE score
    itself is still computed correctly (see Spice.compute_score in
    pycocoevalcap/spice/spice.py -- the cache dir is passed via '-cache' but
    failure to write there doesn't fail the job or corrupt results). This is
    purely cosmetic log noise, unrelated to evaluation/spice_cache.py (which
    only caches the CoreNLP model files, not this per-caption cache).

    Everything else printed by the jar (including real errors) still passes
    through, and a non-zero exit code still raises CalledProcessError exactly
    like the un-patched subprocess.check_call would.
    """
    original_check_call = subprocess.check_call

    def patched_check_call(cmd, *args, **kwargs):
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
        proc = subprocess.Popen(cmd, *args, **kwargs)
        assert proc.stdout is not None
        for line in proc.stdout:
            if not _SPICE_CACHE_NOISE_PATTERN.search(line):
                print(line, end="")
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return returncode

    subprocess.check_call = patched_check_call
    try:
        yield
    finally:
        subprocess.check_call = original_check_call

from typing import Dict, List, Any
import logging

from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Official PTBTokenizer wrapper (shared by CIDEr-D, METEOR, SPICE)
# ---------------------------------------------------------------------------

def _ptb_tokenize_pairs(predictions: List[str], references: List[str]):
    """Tokenizes predictions and references using the OFFICIAL PTBTokenizer
    (Penn Treebank tokenization via Stanford CoreNLP), exactly as the coco-caption
    toolkit does before computing CIDEr-D / METEOR / SPICE.

    Returns:
        (res, gts): two dicts of {str_id: [tokenized_string, ...]}, in the
        exact format pycocoevalcap's scorers expect.
    """
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ids = [str(i) for i in range(len(predictions))]
    gts_input = {img_id: [{"caption": ref}] for img_id, ref in zip(ids, references)}
    res_input = {img_id: [{"caption": pred}] for img_id, pred in zip(ids, predictions)}

    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts_input)
    res_tokenized = tokenizer.tokenize(res_input)

    return res_tokenized, gts_tokenized


def _check_java_available() -> bool:
    """Quick check that `java` is on PATH before attempting a Java-backed metric."""
    import shutil
    return shutil.which("java") is not None


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        import transformers
        from bert_score import score

        transformers.utils.logging.set_verbosity_error()

        P, R, F1 = score(predictions, references, lang="en", verbose=False, rescale_with_baseline=True)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except ImportError:
        logger.warning("bert_score or transformers not installed. Skipping BERTScore.")
        return {}


# ---------------------------------------------------------------------------
# METEOR — OFFICIAL Meteor-1.5 jar
# ---------------------------------------------------------------------------

def compute_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Computes METEOR using the OFFICIAL Meteor-1.5 jar bundled with
    pycocoevalcap (exact/stem/synonym/paraphrase matching), matching the
    tool the paper cites in footnote 3. Requires a JRE on PATH.
    """
    if not _check_java_available():
        logger.warning("Java not found on PATH. Skipping official METEOR. "
                        "Run `!apt-get install -y default-jre` first.")
        return {}

    scorer = None
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        res, gts = _ptb_tokenize_pairs(predictions, references)
        scorer = Meteor()
        score, _ = scorer.compute_score(gts, res)
        return {"meteor": float(score)}
    except Exception as e:
        logger.warning(f"Failed to compute official METEOR: {e}")
        return {}
    finally:
        if scorer is not None:
            try:
                del scorer  # triggers Meteor.__del__, which kills the JVM subprocess
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CIDEr-D — official pycocoevalcap Cider class + PTBTokenizer
# ---------------------------------------------------------------------------

def compute_cider(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Computes CIDEr-D using pycocoevalcap's Cider class, fed with
    PTBTokenizer output (matching the official coco-caption pipeline the
    paper cites), instead of NLTK-tokenized strings.
    """
    if not _check_java_available():
        logger.warning("Java not found on PATH. Skipping official CIDEr-D (needs PTBTokenizer). "
                        "Run `!apt-get install -y default-jre` first.")
        return {}

    try:
        from pycocoevalcap.cider.cider import Cider

        res, gts = _ptb_tokenize_pairs(predictions, references)
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        return {"ciderd": float(score)}
    except Exception as e:
        logger.warning(f"Failed to compute official CIDEr-D: {e}")
        return {}


# ---------------------------------------------------------------------------
# SPICE — new. Scene-graph F1 via Stanford CoreNLP.
# ---------------------------------------------------------------------------

def compute_spice(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Computes SPICE (scene-graph tuple F1) using pycocoevalcap's Spice class.

    On first-ever call, this downloads ~2GB of Stanford CoreNLP models into
    the pycocoevalcap package directory. Use evaluation/spice_cache.py to
    persist that download to Drive across Colab sessions.

    Also returns per-category sub-scores (Object/Relation/Attribute/Color/
    Cardinality/Size F1), actual reason to use SPICE over the
    other caption metrics.
    """
    if not _check_java_available():
        logger.warning("Java not found on PATH. Skipping SPICE. "
                        "Run `!apt-get install -y default-jre` first.")
        return {}

    try:
        from pycocoevalcap.spice.spice import Spice

        res, gts = _ptb_tokenize_pairs(predictions, references)
        scorer = Spice()
        with _suppress_spice_cache_noise():
            score, scores = scorer.compute_score(gts, res)

        result = {"spice": float(score)}

        sub_categories = ["Object", "Relation", "Attribute", "Color", "Cardinality", "Size"]
        for cat in sub_categories:
            cat_scores = [
                s[cat]["f"] for s in scores
                if cat in s and s[cat]["f"] == s[cat]["f"]  # filters NaN (NaN != NaN)
            ]
            if cat_scores:
                result[f"spice_{cat.lower()}_f1"] = sum(cat_scores) / len(cat_scores)

        return result
    except Exception as e:
        logger.warning(f"Failed to compute SPICE: {e}")
        return {}


# ---------------------------------------------------------------------------
# CLIPScore
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None

def _get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model_id = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_processor

def compute_clipscore(predictions: List[str], images: List[Any], batch_size: int = 32) -> Dict[str, float]:
    """Computes CLIPScore between predictions and images.

    Uses the standard 2.5 * max(cos_sim, 0) scaling from Hessel et al. 2021.
    Text-image pairs are validated and aligned before batching to guarantee
    that no indexing mismatch can occur if an image fails to load.

    Args:
        predictions: List of caption strings.
        images: List of PIL Images, file paths, or other image-like objects.
        batch_size: Number of pairs to process per forward pass.

    Returns:
        Dict with 'clipscore' key, or empty dict on failure.
    """
    if not images or len(predictions) != len(images):
        return {}

    try:
        import torch
        from PIL import Image

        logger.info("Computing CLIPScore (batched)...")
        model, processor = _get_clip_model()
        device = model.device

        # Pre-validate and pair predictions with images upfront.
        # This guarantees text[i] always corresponds to image[i] even
        # when invalid images are dropped.
        valid_pairs = []
        for pred, img in zip(predictions, images):
            if not isinstance(img, Image.Image):
                if isinstance(img, str):
                    try:
                        img = Image.open(img).convert("RGB")
                    except Exception:
                        continue
                else:
                    continue
            valid_pairs.append((pred, img))

        if not valid_pairs:
            return {"clipscore": 0.0}

        scores = []
        for i in range(0, len(valid_pairs), batch_size):
            batch = valid_pairs[i:i + batch_size]
            batch_preds = [p for p, _ in batch]
            batch_imgs = [img for _, img in batch]

            inputs = processor(text=batch_preds, images=batch_imgs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Element-wise dot product: text_embeds[i] · image_embeds[i]
            # This is correct because CLIP encodes text and images independently
            # through separate encoders, so the i-th text embed corresponds to
            # the i-th image embed as long as inputs are aligned (guaranteed above).
            cos_sims = torch.sum(image_embeds * text_embeds, dim=-1)
            for cos_sim in cos_sims:
                scores.append(max(0.0, 2.5 * cos_sim.item()))

        return {"clipscore": sum(scores) / len(scores) if scores else 0.0}
    except Exception as e:
        logger.warning(f"Failed to compute CLIPScore: {e}")
        return {}


# ---------------------------------------------------------------------------
# Long-context CLIPScore
#
# Standard CLIP (openai/clip-vit-base-patch32) has a HARD 77-token text limit and
# compute_clipscore passes truncation=True, so it silently sees only the first ~55
# words of a caption. Ground-truth captions average 48.5 words and run much longer,
# which is why CLIPScore was essentially flat baseline-vs-SFT (0.7781 -> 0.7732)
# while every content-overlap metric moved sharply: the metric never saw the part of
# the caption the models actually differ on.
#
# We keep the standard number for historical comparability and add a second one from
# a long-context model, reported under a separate key so the two are never confused.
# ---------------------------------------------------------------------------

# The 77-token limit is a property of CLIP's TEXT TOWER, not of the score. So rather
# than swapping in a different model, we keep the SAME encoder and stop throwing the
# caption away: split it into windows that each fit inside the limit, score every
# window against the image, and average.
#
# Why not a long-context CLIP variant: jinaai/jina-clip-v1 was tried first and is
# incompatible with transformers 5.x. Its remote `eva_model.py` does
#     x.item() for x in torch.linspace(0, drop_path_rate, depth)
# inside __init__, and transformers 5 builds models under a meta-device
# TorchFunctionMode, where `.item()` raises
#     RuntimeError: Tensor.item() cannot be called on meta tensors
# There is no from_pretrained flag that disables that meta init, and
# low_cpu_mem_usage=False does not help. Any model shipping third-party remote code
# carries that class of risk on a pinned transformers.
#
# Chunking has real advantages over a second model here:
#   * zero new dependencies, zero extra download, no remote code to break;
#   * works offline immediately -- the CLIP weights are already cached;
#   * DIRECTLY comparable to `clipscore`, because it is the same encoder and the
#     same 2.5 * max(0, cos) formula. The only difference is how much of the caption
#     was read, which is exactly the quantity in question.
#
# The tradeoff, stated plainly: chunk-mean cannot see coherence ACROSS windows. It
# answers "is every part of this caption supported by the image?" rather than "is
# this caption, as a whole, the best description?". For catching a model that
# describes the first sentence correctly and then drifts, that is the more useful
# question anyway.

# CLIP's context is 77 positions including BOS and EOS, leaving 75 for content.
_CLIP_CONTENT_TOKENS = 75


def _chunk_by_token_budget(text: str, processor, budget: int = _CLIP_CONTENT_TOKENS) -> List[str]:
    """Splits text into the fewest word-aligned chunks that each fit in `budget`
    CLIP text tokens.

    Word-aligned rather than token-aligned so every chunk is readable text the
    encoder sees the same way it would see a short caption. A single word longer
    than the budget (never happens with real captions, but be safe) is emitted on
    its own and left for the encoder to truncate.
    """
    words = str(text).split()
    if not words:
        return []

    tok = getattr(processor, "tokenizer", processor)

    def n_tokens(s: str) -> int:
        # add_special_tokens=False so the count is content only.
        return len(tok(s, add_special_tokens=False)["input_ids"])

    # Fast path: the whole caption already fits, so this is identical to clipscore.
    if n_tokens(text) <= budget:
        return [str(text)]

    chunks, current = [], []
    for word in words:
        candidate = current + [word]
        if current and n_tokens(" ".join(candidate)) > budget:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def compute_long_clipscore(predictions: List[str], images: List[Any],
                           batch_size: int = 32) -> Dict[str, float]:
    """CLIPScore over the WHOLE caption, via token-budgeted chunking.

    Same model and same 2.5 * max(0, cosine) convention as compute_clipscore, so the
    two numbers sit on one scale and the difference between them is precisely the
    part of the caption standard CLIPScore never read.

    Per caption: split into <=75-token windows, score each against the image, take
    the mean. Then average over captions.

    Returns {} on any failure, so a broken metric is visibly ABSENT from
    metrics.json rather than reported as a score of 0.0.
    """
    if os.environ.get("VLM_DISABLE_LONG_CLIP", "").strip() not in ("", "0", "false", "False"):
        logger.info("VLM_DISABLE_LONG_CLIP is set — skipping the chunked CLIPScore.")
        return {}

    try:
        import torch
        from PIL import Image

        model, processor = _get_clip_model()
        device = model.device

        # Pair up, dropping anything unusable, exactly as compute_clipscore does.
        valid = []
        for pred, img in zip(predictions, images):
            if not isinstance(img, Image.Image):
                if isinstance(img, str):
                    try:
                        img = Image.open(img).convert("RGB")
                    except Exception:
                        continue
                else:
                    continue
            if not (pred and str(pred).strip()):
                continue
            valid.append((str(pred), img))

        if not valid:
            return {}

        per_caption_scores = []
        chunk_counts = []
        truncated_captions = 0

        with torch.no_grad():
            for text, img in valid:
                chunks = _chunk_by_token_budget(text, processor)
                if not chunks:
                    continue
                chunk_counts.append(len(chunks))
                truncated_captions += len(chunks) > 1

                chunk_scores = []
                for i in range(0, len(chunks), batch_size):
                    window = chunks[i:i + batch_size]
                    inputs = processor(
                        text=window,
                        images=[img] * len(window),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                    t = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
                    v = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
                    cos = torch.sum(t * v, dim=-1)
                    chunk_scores.extend(max(0.0, 2.5 * float(c)) for c in cos)

                if chunk_scores:
                    per_caption_scores.append(sum(chunk_scores) / len(chunk_scores))

        if not per_caption_scores:
            return {}

        n = len(per_caption_scores)
        return {
            "long_clipscore": sum(per_caption_scores) / n,
            "long_clipscore_scored_count": n,
            # How many captions standard CLIPScore was silently truncating, and by
            # how much. If avg_chunks is ~1.0 the two CLIPScores should agree and the
            # 77-token limit was never binding on this data.
            "long_clipscore_truncated_caption_count": truncated_captions,
            "long_clipscore_avg_chunks_per_caption": (
                sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0
            ),
        }
    except Exception as e:
        logger.warning(f"Failed to compute chunked (long) CLIPScore: {e}")
        return {}


# ---------------------------------------------------------------------------
# Caption Length Stats
# ---------------------------------------------------------------------------

def compute_caption_length_stats(predictions: List[str]) -> Dict[str, float]:
    """Average word count of generated captions (predictions only)."""
    if not predictions:
        return {}
    word_counts = [len(str(p).split()) for p in predictions]
    if not word_counts:
        return {}
    return {
        "avg_words_per_caption": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_caption_metrics(
    predictions: List[str],
    references: List[str],
    images: List[Any],
    include_spice: bool = True,
    spice_only: bool = False,
    prefix: str = "",
) -> Dict[str, float]:
    """Computes all captioning metrics: BERTScore, METEOR, CIDEr-D, SPICE,
    and CLIPScore.

    Args:
        include_spice: Set False for short, non-descriptive text (e.g. one-line
            violation "reason" strings in reasoning eval) — SPICE's scene-graph
            parsing is designed for full descriptive captions and is both
            slow (extra JVM spin-up) and not meaningful on short phrases.
    """
    if not predictions or not references:
        raise ValueError(
            "compute_all_caption_metrics requires non-empty predictions and references lists."
        )
    if len(predictions) != len(references):
        raise ValueError(
            f"compute_all_caption_metrics: length mismatch — "
            f"{len(predictions)} predictions vs {len(references)} references."
        )
    if images is None:
        raise ValueError(
            "CLIPScore requires images. You explicitly passed images=None to the caption metrics suite."
        )

    # ---------------------------------------------------------------------------
    # Blank-prediction policy
    #
    # This used to rewrite any blank prediction (or reference) to the literal string
    # "empty" and hand it to the graders. That is worse than it looks: a completely
    # failed generation earned a real, nonzero BERTScore/METEOR for the word
    # "empty", so a model that emitted nothing scored as a mediocre captioner rather
    # than a broken one — and the failure was invisible in the caption metrics.
    #
    # Instead: blanks are EXCLUDED from the graders and reported as their own rate.
    # This mirrors the convention the repo already uses for violations
    # (violation_prediction_failure_rate), so both failure modes read the same way.
    # It also means the graders never receive an empty string, which is what made
    # BERTScore fragile here in the first place.
    #
    # Read the two numbers together: `bertscore_f1` is quality GIVEN a caption was
    # produced, and `blank_prediction_rate` is how often one wasn't. A model cannot
    # hide a high blank rate behind a good conditional score, because both ship.
    total = len(predictions)
    kept_preds, kept_refs, kept_images = [], [], []
    blank_pred = blank_ref = 0
    for i, (p, r) in enumerate(zip(predictions, references)):
        p_blank = not (p and str(p).strip())
        r_blank = not (r and str(r).strip())
        blank_pred += p_blank
        blank_ref += r_blank
        if p_blank or r_blank:
            continue
        kept_preds.append(str(p))
        kept_refs.append(str(r))
        if images is not None and i < len(images):
            kept_images.append(images[i])

    results = {
        "blank_prediction_count": blank_pred,
        "blank_prediction_rate": (blank_pred / total) if total else 0.0,
        "blank_reference_count": blank_ref,
        "scored_count": len(kept_preds),
        "total_count": total,
    }

    if blank_pred:
        logger.warning(
            f"{blank_pred}/{total} predictions were blank and are EXCLUDED from the "
            f"caption metrics (reported as blank_prediction_rate="
            f"{results['blank_prediction_rate']:.4f}). Every metric below is "
            "conditional on a caption having been produced."
        )
    if blank_ref:
        logger.warning(
            f"{blank_ref}/{total} REFERENCES were blank — a data problem, not a model "
            "problem. Those pairs are excluded too."
        )

    if not kept_preds:
        logger.error(
            "Every prediction or reference was blank; no caption metric can be "
            "computed. Returning only the blank-rate counters."
        )
        if prefix:
            results = {f"{prefix}{k}": v for k, v in results.items()}
        return results

    if not spice_only:
        results.update(compute_bertscore(kept_preds, kept_refs))
        results.update(compute_meteor(kept_preds, kept_refs))
        results.update(compute_cider(kept_preds, kept_refs))
        results.update(compute_caption_length_stats(kept_preds))
        results.update(compute_clipscore(kept_preds, kept_images))
        # Second, long-context CLIPScore alongside the standard 77-token one.
        # Omitted entirely (no key) if the model is unavailable, rather than
        # reported as 0.0 — an absent metric must not look like a bad score.
        results.update(compute_long_clipscore(kept_preds, kept_images))

    if include_spice:
        results.update(compute_spice(kept_preds, kept_refs))

    if prefix:
        results = {f"{prefix}{k}": v for k, v in results.items()}

    return results