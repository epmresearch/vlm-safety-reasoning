"""
Builds Base vs SFT comparison tables across all three model sizes and tasks.
"""
import pandas as pd

from core.constants import PROJECT1_TASKS
from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)
SIZES = ["small", "medium", "large"]


def _load_eval_csv(task: str, filename: str) -> pd.DataFrame:
    path = get_drive_path("results", task, filename)
    if not path.exists():
        logger.warning(f"Missing results file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"model": label, "avg_score": None, "avg_json_valid": None, "avg_latency_ms": None, "n": 0}
    return {
        "model": label, "avg_score": df["score"].mean(),
        "avg_json_valid": df["json_valid"].mean(),
        "avg_latency_ms": df["latency_ms"].mean(), "n": len(df),
    }


def compare_results_multisize(variant_name: str = "sft_v1"):
    all_tables = {}
    for task in PROJECT1_TASKS:
        rows = []
        for size in SIZES:
            base_df = _load_eval_csv(task, f"{size}-base_baseline_eval.csv")
            sft_df = _load_eval_csv(task, f"{size}-{variant_name}_eval.csv")
            rows.append(_summarize(base_df, f"{size}-Base"))
            rows.append(_summarize(sft_df, f"{size}-SFT"))

        comparison_df = pd.DataFrame(rows)
        out_path = get_drive_path("results", task, "comparison_table_multisize.csv")
        ensure_dir(out_path.parent)
        comparison_df.to_csv(out_path, index=False)
        all_tables[task] = comparison_df

        logger.info(f"[{task}] comparison table saved to {out_path}")
        print(f"\n=== {task} ===")
        print(comparison_df.to_string(index=False))

    return all_tables


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant_name", default="sft_v1")
    args = parser.parse_args()
    compare_results_multisize(variant_name=args.variant_name)


if __name__ == "__main__":
    main()