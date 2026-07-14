"""
LLM-as-a-Judge reasoning evaluation.

Scores the "reason" field in safety violations using Meta-Llama-3-8B-Instruct
loaded locally in 4-bit.

Dimensions (0-2 each, total 0-6):
- Relevance: Addresses the specific rule.
- Equivalence: Identifies same violator as ground truth.
- Specificity: Describes specific visual evidence.
"""
from typing import Any, Dict, List, Tuple
import torch

from core.logging import get_logger

logger = get_logger(__name__)

SCORING_PROMPT = """You are an expert safety inspector evaluator.
Evaluate the predicted reasoning for a safety violation against the ground truth.

Rule ID: {rule_id}

Ground Truth Reasoning:
{gt_reason}

Predicted Reasoning:
{pred_reason}

Rate the Predicted Reasoning on three dimensions, assigning a score of 0, 1, or 2 for each:
1. Relevance: Does it address the specific rule? (0=No, 1=Somewhat, 2=Yes)
2. Equivalence: Does it identify the same core violation/violator as the ground truth? (0=No, 1=Partial, 2=Yes)
3. Specificity: Does it provide specific visual evidence? (0=Vague, 1=Basic, 2=Detailed)

Respond ONLY with a JSON object in this exact format:
{{"relevance": <int>, "equivalence": <int>, "specificity": <int>}}
"""

def load_judge_model() -> Tuple[Any, Any]:
    """Loads Meta-Llama-3-8B-Instruct locally in 4-bit for evaluation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading Llama-3-8B Judge model in 4-bit...")
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Judge model: {e}")
        return None, None

def score_reasoning(
    judge_model: Any, 
    judge_tokenizer: Any, 
    prediction_reason: str, 
    gt_reason: str, 
    rule_id: str
) -> Dict[str, float]:
    """Scores a single reasoning string on a scale of 0-6."""
    if judge_model is None or judge_tokenizer is None:
        return {"relevance": 0.0, "equivalence": 0.0, "specificity": 0.0, "total": 0.0}
        
    prompt = SCORING_PROMPT.format(
        rule_id=rule_id, 
        gt_reason=gt_reason, 
        pred_reason=prediction_reason
    )
    
    messages = [
        {"role": "system", "content": "You are a precise evaluator that outputs ONLY valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text = judge_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    
    try:
        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = judge_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        import json
        import re
        
        # Extract JSON if fenced
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            scores = json.loads(json_str)
            
            rel = float(scores.get("relevance", 0))
            eq = float(scores.get("equivalence", 0))
            spec = float(scores.get("specificity", 0))
            total = rel + eq + spec
            
            return {
                "relevance": rel, 
                "equivalence": eq, 
                "specificity": spec, 
                "total": total
            }
    except Exception as e:
        logger.error(f"Judge model evaluation failed: {e}")
        
    return {"relevance": 0.0, "equivalence": 0.0, "specificity": 0.0, "total": 0.0}

def batch_score_reasoning(
    judge_model: Any, 
    judge_tokenizer: Any, 
    pred_violations: List[List[Dict[str, Any]]], 
    gt_violations: List[List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Batched reasoning evaluation for correctly identified violations."""
    if not judge_model:
        return {"reasoning_score_avg": 0.0}
        
    total_scores = []
    
    from tqdm import tqdm
    for pred_list, gt_list in tqdm(zip(pred_violations, gt_violations), desc="Judge Eval", total=len(pred_violations)):
        pred_list = pred_list or []
        gt_list = gt_list or []
        
        pred_by_rule = {v.get("rule_id"): v.get("reason", "") for v in pred_list if v.get("rule_id")}
        gt_by_rule = {v.get("rule_id"): v.get("reason", "") for v in gt_list if v.get("rule_id")}
        
        common_rules = set(pred_by_rule.keys()) & set(gt_by_rule.keys())
        
        for r in common_rules:
            pred_reason = pred_by_rule[r]
            gt_reason = gt_by_rule[r]
            
            scores = score_reasoning(judge_model, judge_tokenizer, pred_reason, gt_reason, r)
            total_scores.append(scores["total"])
            
    if not total_scores:
        return {"reasoning_score_avg": 0.0}
        
    return {"reasoning_score_avg": sum(total_scores) / len(total_scores)}
