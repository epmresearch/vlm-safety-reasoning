"""
Builds the Base vs SFT vs SFT+GSPO comparison table for a task.
Usage: python experiments/compare_results.py --task rule_violation
"""
import argparse
import pandas as pd

from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def load_eval_csv(task: str, filename: str) -> pd.DataFrame:
    path = get_drive_path("results", task, filename)
    if not path.exists():
        logger.warning(f"Missing results file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    baseline_df = load_eval_csv(args.task, "baseline_eval.csv")
    sft_df = load_eval_csv(args.task, "sft_v1_eval.csv")
    grpo_df = load_eval_csv(args.task, "grpo_v1_eval.csv")

    def summarize(df: pd.DataFrame, label: str) -> dict:
        if df.empty:
            return {"model": label, "avg_score": None, "avg_json_valid": None, "avg_latency_ms": None}
        return {
            "model": label,
            "avg_score": df["score"].mean(),
            "avg_json_valid": df["json_valid"].mean(),
            "avg_latency_ms": df["latency_ms"].mean(),
        }

    summary_rows = [
        summarize(baseline_df, "Base"),
        summarize(sft_df, "SFT"),
        summarize(grpo_df, "SFT+GSPO/GRPO"),
    ]

    comparison_df = pd.DataFrame(summary_rows)
    out_path = get_drive_path("results", args.task, "comparison_table.csv")
    ensure_dir(out_path.parent)
    comparison_df.to_csv(out_path, index=False)

    logger.info(f"Comparison table saved to {out_path}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()