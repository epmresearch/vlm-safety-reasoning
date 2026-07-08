"""
Entry point: runs GRPO/GSPO training on top of an SFT checkpoint, then evaluates.
Usage: python experiments/run_grpo.py --task rule_violation --model_id rule_violation-sft_v1
"""
import argparse

from models.grpo_trainer import run_grpo
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
    parser.add_argument("--variant_name", default="grpo_v1")
    args = parser.parse_args()

    run_grpo(args.task, args.model_id, args.variant_name)

    new_model_id = f"{args.task}-{args.variant_name}"
    task_cfg = load_task_config(args.task)
    dataset = load_construction_dataset()

    evaluator = ModelEvaluator(model_id=new_model_id, task=args.task, task_cfg=task_cfg)
    results = evaluator.run(dataset["test"], run_name=args.variant_name)
    evaluator.save_results(results, filename=f"{args.variant_name}_eval.csv")
    save_failure_report(results, args.task, filename=f"{args.variant_name}_failures.csv")

    logger.info("GRPO run + evaluation complete.")


if __name__ == "__main__":
    main()