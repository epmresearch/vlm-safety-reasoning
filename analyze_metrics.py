import json
from pathlib import Path

TIERS = ['2b', '4b', '8b']
PHASES = ['baseline', 'sft', 'grpo']
KEYS = [
    'violation_identification_f1_macro', 
    'violation_identification_f1_rule_1', 
    'violation_identification_f1_rule_2', 
    'violation_identification_f1_rule_3', 
    'violation_identification_f1_rule_4',
    'violation_identification_iou_conditioned_f1_macro',
    'structural_schema_adherence_rate',
    'reasoning_text_similarity_bertscore_f1_macro'
]

RESULTS_DIR = Path('evaluation_results')

print(f"{'Model':<15} | {'Macro F1':<8} | {'Rule1 F1':<8} | {'Rule2 F1':<8} | {'Rule3 F1':<8} | {'Rule4 F1':<8} | {'Schema%':<7} | {'BERT F1':<7}")
print('-'*100)

for tier in TIERS:
    for phase in PHASES:
        p = RESULTS_DIR / f'vo_{phase}_{tier}' / 'metrics.json'
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            vals = []
            for k in KEYS:
                v = d.get(k, 0)
                vals.append(f'{v:.3f}' if isinstance(v, float) else str(v))
            
            print(f"{tier.upper()+' '+phase.upper():<15} | {vals[0]:<8} | {vals[1]:<8} | {vals[2]:<8} | {vals[3]:<8} | {vals[4]:<8} | {vals[6]:<7} | {vals[7]:<7}")
        else:
            print(f"{tier.upper()+' '+phase.upper():<15} | MISSING")
