# ============================================================
# Cell 1: Setup Google Drive & Environment
# ============================================================
from google.colab import drive
import os

drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/vlm-finetuning-project1/vlm-safety-reasoning"

# ============================================================
# Cell 2: Setup Paths
# ============================================================
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ============================================================
# Cell 3: Generate Summary Comparison
# ============================================================
import pandas as pd
from core.constants import DEFAULT_MODEL_TIER
from core.io import get_drive_path
from models.model_loader import get_model_info
import json

short_name = get_model_info(DEFAULT_MODEL_TIER)["short_name"]

def load_metrics(variant: str):
    path = get_drive_path("results", short_name, variant) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

base_metrics = load_metrics("baseline")
sft_metrics = load_metrics("unified-sft-v1")

rows = []
for label, m in [("Base", base_metrics), ("SFT", sft_metrics)]:
    if not m:
        continue
    rows.append({
        "Model": label,
        "Valid_JSON_%": m.get("structural", {}).get("valid_json_ratio", 0) * 100,
        "Complete_Format_%": m.get("structural", {}).get("complete_format_ratio", 0) * 100,
        "Caption_BERTScore": m.get("captioning", {}).get("bert_f1", 0),
        "Caption_CLIPScore": m.get("captioning", {}).get("clip_score", 0),
        "Grounding_IoU": m.get("grounding", {}).get("mean_iou", 0),
        "Violation_F1": m.get("violations", {}).get("f1", 0),
    })

df = pd.DataFrame(rows)
print("=== Results Summary ===")
print(df.to_string(index=False))

# ============================================================
# Cell 4: Generate Paper Tables
# ============================================================
from evaluation.report_generator import generate_paper_tables

# Note: this requires run_full_evaluation output formatting.
print("Generating comprehensive latex tables...")
# You can customize this function in report_generator.py to parse the metrics.json
# and dump out formatted latex strings.

# ============================================================
# Cell 5: Generate Paper Figures
# ============================================================
# If you have scripts/generate_figures.py updated:
# !python scripts/generate_figures.py
print("To generate visual plots, run: python scripts/generate_figures.py")
