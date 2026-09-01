"""
Metrics for structural evaluation (e.g., JSON validity).
"""
from typing import List, Dict, Any, Optional

from evaluation.output_parser import parse_output_for_task, validate_output_for_task
from core.logging import get_logger

logger = get_logger(__name__)

def compute_structural_metrics(raw_outputs: List[str], task: str = "unified") -> Dict[str, float]:
    """
    Computes JSON validity and schema adherence metrics.

    Args:
        raw_outputs: list of raw string responses from the model.
        task: Task name. Selects both the wire format used to parse the raw output
            and the schema it is validated against.

    Metric key names are identical for every task so the comparison tables and
    plots stay uniform. For a PLAIN-TEXT task (caption_only) there is no JSON to
    validate, so ``structural_json_validity_rate`` means "the completion yielded a
    usable caption payload" — i.e. non-empty prose, or prose recoverable from a
    stray fence / {"caption": ...} wrapper — rather than "parsed as JSON".
    """
    if not raw_outputs:
        raise ValueError(
            "compute_structural_metrics requires a non-empty list of raw outputs."
        )
        
    total = len(raw_outputs)
    valid_json_count = 0
    valid_schema_count = 0
    
    for raw_str in raw_outputs:
        parsed = parse_output_for_task(raw_str, task=task)
        if parsed is not None:
            valid_json_count += 1
            validated = validate_output_for_task(parsed, task=task)
            if validated is not None:
                valid_schema_count += 1
                
    return {
        "structural_json_validity_rate": valid_json_count / total,
        "structural_schema_adherence_rate": valid_schema_count / total,
        "structural_valid_json_count": valid_json_count,
        "structural_valid_schema_count": valid_schema_count,
        "structural_total_samples_count": total,
    }
