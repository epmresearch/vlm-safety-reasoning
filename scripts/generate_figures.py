"""
Reads multi-size results CSVs from Drive and produces paper-ready figures.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def plot_comparison(task: str) -> None:
    path = get_drive_path("results", task, "comparison_table_multisize.csv")
    if not path.exists():
        logger.warning(f"No comparison table found at {path}. Run compare_results_multisize first.")
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["model"], df["avg_score"])
    ax.set_ylabel("Average Score")
    ax.set_title(f"{task}: Base vs SFT across model sizes")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fig_dir = get_drive_path("figures")
    ensure_dir(fig_dir)
    fig.savefig(fig_dir / f"{task}_multisize_comparison.png", dpi=200)
    fig.savefig(fig_dir / f"{task}_multisize_comparison.pdf")
    logger.info(f"Saved figures for {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    plot_comparison(args.task)