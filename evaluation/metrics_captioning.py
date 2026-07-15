"""
Metrics for image captioning evaluation.
"""
from typing import Dict, List
import logging

from core.logging import get_logger

logger = get_logger(__name__)

def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes BERTScore for a batch of predictions.
    Requires bert-score package.
    """
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore.")
        return {}

def compute_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes METEOR score.
    Requires nltk package.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        scores = []
        for p, r in zip(predictions, references):
            scores.append(meteor_score([r.split()], p.split()))
            
        return {"meteor": sum(scores) / len(scores) if scores else 0.0}
    except ImportError:
        logger.warning("nltk not installed. Skipping METEOR.")
        return {}

def compute_cider(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes CIDEr score.
    Requires pycocoevalcap package.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        
        # pycocoevalcap expects dict of list of strings
        res = {i: [p] for i, p in enumerate(predictions)}
        gts = {i: [r] for i, r in enumerate(references)}
        
        score, _ = scorer.compute_score(gts, res)
        return {"cider": float(score)}
    except ImportError:
        logger.warning("pycocoevalcap not installed. Skipping CIDEr.")
        return {}

def compute_all_caption_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes all captioning metrics: BERTScore, METEOR, CIDEr.
    """
    if not predictions or not references or len(predictions) != len(references):
        return {}
        
    # Sanitize empty strings to avoid tokenizer crashes in underlying metric libraries (like bert_score)
    clean_preds = [p if p and str(p).strip() else "empty" for p in predictions]
    clean_refs = [r if r and str(r).strip() else "empty" for r in references]
        
    results = {}
    results.update(compute_bertscore(clean_preds, clean_refs))
    results.update(compute_meteor(clean_preds, clean_refs))
    results.update(compute_cider(clean_preds, clean_refs))
    
    return results
