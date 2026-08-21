import re

with open("preprocessing/structural_repair.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update repair_and_validate signature
content = re.sub(
    r"def repair_and_validate\(raw_str: str, duplicate_box_threshold: int = 3\) -> Dict\[str, Any\]:",
    r"def repair_and_validate(raw_str: str, duplicate_box_threshold: int = 3, task: str = \"unified\") -> Dict[str, Any]:",
    content
)

# 2. Update UnifiedOutput usage in repair_and_validate
content = re.sub(
    r"validated_raw = UnifiedOutput\(\*\*strict_parsed\)",
    r"from data.schemas import get_output_schema\n            schema_cls = get_output_schema(task)\n            validated_raw = schema_cls(**strict_parsed)",
    content
)

content = re.sub(
    r"validated_fixed = UnifiedOutput\(\*\*fixed\)",
    r"from data.schemas import get_output_schema\n        schema_cls = get_output_schema(task)\n        validated_fixed = schema_cls(**fixed)",
    content
)

# 3. Update process_jsonl signature
content = re.sub(
    r"def process_jsonl\(\n    input_path: str, output_path: str, report_path: str,\n    broken_path: str, manifest_path: str,\n    duplicate_box_threshold: int = 3,\n\):",
    r"def process_jsonl(\n    input_path: str, output_path: str, report_path: str,\n    broken_path: str, manifest_path: str,\n    duplicate_box_threshold: int = 3,\n    task: str = \"unified\",\n):",
    content
)

# 4. Update repair_and_validate call in process_jsonl
content = re.sub(
    r"result = repair_and_validate\(raw, duplicate_box_threshold=duplicate_box_threshold\)",
    r"result = repair_and_validate(raw, duplicate_box_threshold=duplicate_box_threshold, task=task)",
    content
)

# 5. Update CLI arguments
content = re.sub(
    r"args = parser\.parse_args\(\)",
    r"parser.add_argument(\"--task\", default=\"unified\", help=\"Task name: 'unified' or 'violations_only'\")\n    args = parser.parse_args()",
    content
)

# 6. Update process_jsonl call in CLI
content = re.sub(
    r"duplicate_box_threshold=args\.duplicate_box_threshold,\n    \)",
    r"duplicate_box_threshold=args.duplicate_box_threshold,\n        task=args.task,\n    )",
    content
)

with open("preprocessing/structural_repair.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated structural_repair.py")
