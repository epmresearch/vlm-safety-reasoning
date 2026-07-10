"""
Baseline (zero-shot, no fine-tuning) evaluation across all three model sizes
and all three Project 1 tasks.
"""
from typing import Optional

from core.config import load_task_config
from core.constants import PROJECT1_TASKS
from core.eval_sampling import get_eval_split
from data.loader import load_construction_dataset
from models.registry import REGISTRY
from evaluation.evaluator import ModelEvaluator
from evaluation.error_analyzer import save_failure_report
from evaluation.rule_violation_report import save_rule_precision_recall_table
from core.logging import get_logger

logger = get_logger(__name__)


def run_baseline_multisize(max_samples: Optional[int] = None):
    dataset = load_construction_dataset()
    eval_split = get_eval_split(dataset["test"], max_samples)

    size_model_ids = [mid for mid in REGISTRY.keys() if mid.endswith("-base")]
    logger.info(f"Running baseline for model sizes: {size_model_ids} on {len(eval_split)} images")

    for model_id in size_model_ids:
        for task in PROJECT1_TASKS:
            task_cfg = load_task_config(task)
            logger.info(f"=== Baseline: model={model_id}, task={task} ===")

            evaluator = ModelEvaluator(model_id=model_id, task=task, task_cfg=task_cfg)
            results = evaluator.run(eval_split, run_name=f"{model_id}-baseline")
            evaluator.save_results(results, filename=f"{model_id}_baseline_eval.csv")
            save_failure_report(results, task, filename=f"{model_id}_baseline_failures.csv")

            if task == "rule_violation":
                save_rule_precision_recall_table(
                    results, model_id, filename=f"{model_id}_baseline_rule_pr_table.csv"
                )

    logger.info("All baseline runs complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Evaluate on a random subsample (recommended while validating). "
                              "Omit for the full 3,004-image test split.")
    args = parser.parse_args()
    run_baseline_multisize(max_samples=args.max_samples)


if __name__ == "__main__":
    main()