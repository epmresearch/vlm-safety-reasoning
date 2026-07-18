"""
Metrics for image captioning evaluation.

CIDEr-D, METEOR, and SPICE use the OFFICIAL pycocoevalcap toolkit
(Java-based PTBTokenizer + Java jars).
Requires: `apt-get install -y default-jre` in the Colab session before use.
Only needed for evaluation notebooks — not for training/inference notebooks.
"""
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
        from transformers import RobertaTokenizer
        from bert_score import score

        transformers.utils.logging.set_verbosity_error()

        if not hasattr(RobertaTokenizer, "build_inputs_with_special_tokens"):
            RobertaTokenizer.build_inputs_with_special_tokens = lambda self, t0, t1=None: [self.cls_token_id] + t0 + [self.sep_token_id]

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
        return {"cider": float(score)}
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

def compute_clipscore(predictions: List[str], images: List[Any]) -> Dict[str, float]:
    if not images or len(predictions) != len(images):
        return {}

    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image

        logger.info("Loading CLIP model for CLIPScore...")
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        scores = []
        for p, img in zip(predictions, images):
            if not isinstance(img, Image.Image):
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                else:
                    continue

            inputs = processor(text=[p], images=img, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            cos_sim = torch.matmul(image_embeds, text_embeds.t()).item()
            scores.append(max(0.0, 2.5 * cos_sim))

        return {"clipscore": sum(scores) / len(scores) if scores else 0.0}
    except Exception as e:
        logger.warning(f"Failed to compute CLIPScore: {e}")
        return {}


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_caption_metrics(
    predictions: List[str],
    references: List[str],
    images: List[Any] = None,
    include_spice: bool = True,
) -> Dict[str, float]:
    """Computes all captioning metrics: BERTScore, METEOR, CIDEr-D, SPICE,
    and optionally CLIPScore.

    Args:
        include_spice: Set False for short, non-descriptive text (e.g. one-line
            violation "reason" strings in reasoning eval) — SPICE's scene-graph
            parsing is designed for full descriptive captions and is both
            slow (extra JVM spin-up) and not meaningful on short phrases.
    """
    if not predictions or not references or len(predictions) != len(references):
        return {}

    clean_preds = [p if p and str(p).strip() else "empty" for p in predictions]
    clean_refs = [r if r and str(r).strip() else "empty" for r in references]

    results = {}
    results.update(compute_bertscore(clean_preds, clean_refs))
    results.update(compute_meteor(clean_preds, clean_refs))
    results.update(compute_cider(clean_preds, clean_refs))
    if include_spice:
        results.update(compute_spice(clean_preds, clean_refs))

    if images:
        results.update(compute_clipscore(clean_preds, images))

    return results