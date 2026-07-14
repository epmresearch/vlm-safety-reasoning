"""
Stratified error analysis module.
Computes metrics grouped by metadata fields like weather, lighting, etc.
"""
from typing import Dict, List, Any
import pandas as pd

from evaluation.evaluator import run_full_evaluation
from core.constants import METADATA_FIELDS, METADATA_VALUES
from core.logging import get_logger

logger = get_logger(__name__)

def run_stratified_analysis(raw_predictions: List[str], references: List[Dict[str, Any]], metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Runs evaluation stratified by metadata fields.
    metadata: list of dictionaries containing metadata for each sample (e.g., {"weather": "rainy", "lighting": "poor"})
    """
    logger.info("Starting stratified error analysis...")
    
    if len(raw_predictions) != len(references) or len(raw_predictions) != len(metadata):
        logger.error("Mismatched lengths for predictions, references, and metadata.")
        return {}
        
    results_by_stratum = {}
    
    for field in METADATA_FIELDS:
        if field not in METADATA_VALUES:
            continue
            
        for value in METADATA_VALUES[field]:
            # Filter samples for this stratum
            indices = [i for i, meta in enumerate(metadata) if meta.get(field) == value]
            
            if not indices:
                continue
                
            stratum_preds = [raw_predictions[i] for i in indices]
            stratum_refs = [references[i] for i in indices]
            
            logger.info(f"Evaluating stratum: {field}={value} (n={len(indices)})")
            
            try:
                eval_result = run_full_evaluation(stratum_preds, stratum_refs)
                stratum_key = f"{field}_{value}"
                results_by_stratum[stratum_key] = eval_result["metrics"]
            except Exception as e:
                logger.error(f"Error evaluating stratum {field}={value}: {e}")
                
    return results_by_stratum