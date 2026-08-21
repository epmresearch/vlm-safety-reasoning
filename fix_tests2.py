import re

with open("tests/test_evaluation/test_evaluator_vo.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("metrics, merged = run_full_evaluation(", "res = run_full_evaluation(")
content = content.replace("metrics.keys()", "res['metrics'].keys()")
content = content.replace("\"violation_rule_1_f1\" in metrics", "\"violation_rule_1_f1\" in res['metrics']")
content = content.replace("\"parse_success_rate\" in metrics", "\"parse_success_rate\" in res['metrics']")


with open("tests/test_evaluation/test_evaluator_vo.py", "w", encoding="utf-8") as f:
    f.write(content)
