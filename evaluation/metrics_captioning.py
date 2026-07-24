"""
Metrics for image captioning evaluation.

CIDEr-D, METEOR, and SPICE use the OFFICIAL pycocoevalcap toolkit
(Java-based PTBTokenizer + Java jars).
Requires: `apt-get install -y default-jre` in the Colab session before use.
Only needed for evaluation notebooks — not for training/inference notebooks.
"""

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

    clean_preds = [p if p and str(p).strip() else "empty" for p in predictions]
    clean_refs = [r if r and str(r).strip() else "empty" for r in references]

    results = {}
    
    if not spice_only:
        results.update(compute_bertscore(clean_preds, clean_refs))
        results.update(compute_meteor(clean_preds, clean_refs))
        results.update(compute_cider(clean_preds, clean_refs))
        results.update(compute_caption_length_stats(clean_preds))
        results.update(compute_clipscore(clean_preds, images))

    if include_spice:
        results.update(compute_spice(clean_preds, clean_refs))

    if prefix:
        results = {f"{prefix}{k}": v for k, v in results.items()}

    return results