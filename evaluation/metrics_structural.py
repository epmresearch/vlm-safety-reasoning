"""
Metrics for structural evaluation (e.g., JSON validity).
"""
from typing import List, Dict, Any, Optional

from evaluation.output_parser import parse_model_output, validate_unified_output
from core.logging import get_logger

logger = get_logger(__name__)

def compute_structural_metrics(raw_outputs: List[str]) -> Dict[str, float]:
    """
    Computes JSON validity and schema adherence metrics.
    raw_outputs: list of raw string responses from the model.
    """
    if not raw_outputs:
        return {}
        
    total = len(raw_outputs)
    valid_json_count = 0
    valid_schema_count = 0
    
    for raw_str in raw_outputs:
        parsed = parse_model_output(raw_str)
        if parsed is not None:
            valid_json_count += 1
            validated = validate_unified_output(parsed)
            if validated is not None:
                valid_schema_count += 1
                
    return {
        "structural_json_validity_rate": valid_json_count / total,
        "structural_schema_adherence_rate": valid_schema_count / total,
        "structural_valid_json_count": valid_json_count,
        "structural_valid_schema_count": valid_schema_count,
        "structural_total_samples_count": total,
    }
