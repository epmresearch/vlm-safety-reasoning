"""
Entry point: runs baseline (no fine-tuning) inference + evaluation for a task.
Usage: python experiments/run_baseline.py --task rule_violation --model_id rule_violation-base
"""
import argparse

from core.config import load_task_config
from data.loader import load_construction_dataset
from evaluation.evaluator import ModelEvaluator
from evaluation.error_analyzer import save_failure_report
from core.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    task_cfg = load_task_config(args.task)
    dataset = load_construction_dataset()

    evaluator = ModelEvaluator(model_id=args.model_id, task=args.task, task_cfg=task_cfg)
    results = evaluator.run(dataset["test"], run_name="baseline")
    evaluator.save_results(results, filename="baseline_eval.csv")
    save_failure_report(results, args.task, filename="baseline_failures.csv")

    logger.info("Baseline run complete.")


if __name__ == "__main__":
    main()