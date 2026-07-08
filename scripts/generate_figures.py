"""
Reads results CSVs from Drive and produces paper-ready PNG + PDF figures.
Usage: python scripts/generate_figures.py --task rule_violation
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from core.io import get_drive_path, ensure_dir
from core.logging import get_logger

logger = get_logger(__name__)


def plot_comparison(task: str) -> None:
    path = get_drive_path("results", task, "comparison_table.csv")
    if not path.exists():
        logger.warning(f"No comparison table found at {path}. Run compare_results.py first.")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["model"], df["avg_score"])
    ax.set_ylabel("Average Score")
    ax.set_title(f"{task}: Base vs SFT vs SFT+GSPO/GRPO")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    fig_dir = get_drive_path("figures")
    ensure_dir(fig_dir)
    png_path = fig_dir / f"{task}_comparison.png"
    pdf_path = fig_dir / f"{task}_comparison.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    logger.info(f"Saved figures: {png_path}, {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    plot_comparison(args.task)