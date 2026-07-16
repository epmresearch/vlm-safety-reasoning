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
        P, R, F1 = score(predictions, references, lang="en", verbose=False, rescale_with_baseline=True)
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
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Download required wordnet and tokenizer assets
        for dataset in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
            try:
                if 'punkt' in dataset:
                    nltk.data.find(f'tokenizers/{dataset}')
                else:
                    nltk.data.find(f'corpora/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
            
        scores = []
        for p, r in zip(predictions, references):
            # Use proper NLP tokenization instead of whitespace splitting
            ref_tokens = word_tokenize(r)
            pred_tokens = word_tokenize(p)
            
            # NLTK meteor_score expects references to be a list of tokenized sentences (list of lists)
            # and hypothesis to be a single tokenized sentence (list).
            scores.append(meteor_score([ref_tokens], pred_tokens))
            
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
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Ensure punkt is available for tokenization
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
        scorer = Cider()
        
        # Instead of relying on PTBTokenizer (which requires Java), we can pre-tokenize 
        # using NLTK and join with spaces. CIDEr's internal .split() will then correctly 
        # separate punctuation, perfectly fixing the bug without needing Java!
        res = {i: [" ".join(word_tokenize(p))] for i, p in enumerate(predictions)}
        gts = {i: [" ".join(word_tokenize(r))] for i, r in enumerate(references)}
        
        score, _ = scorer.compute_score(gts, res)
        return {"cider": float(score)}
    except ImportError:
        logger.warning("pycocoevalcap not installed. Skipping CIDEr.")
        return {}

def compute_clipscore(predictions: List[str], images: List[Any]) -> Dict[str, float]:
    """
    Computes CLIPScore between predictions and images.
    Requires transformers and torch.
    """
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
                # If it's a string path, try to open it
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
            clipscore = max(0.0, 2.5 * cos_sim)
            scores.append(clipscore)
            
        return {"clipscore": sum(scores) / len(scores) if scores else 0.0}
    except Exception as e:
        logger.warning(f"Failed to compute CLIPScore: {e}")
        return {}

def compute_all_caption_metrics(predictions: List[str], references: List[str], images: List[Any] = None) -> Dict[str, float]:
    """
    Computes all captioning metrics: BERTScore, METEOR, CIDEr, and optionally CLIPScore.
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
    
    if images:
        results.update(compute_clipscore(clean_preds, images))
    
    return results
