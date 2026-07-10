"""
Multi-task SFT training + evaluation across all three model sizes.
"""
from typing import List, Optional

from core.config import load_task_config
from core.constants import PROJECT1_TASKS
from core.eval_sampling import get_eval_split
from data.loader import load_construction_dataset
from models.registry import REGISTRY
from models.sft_trainer import run_sft_multitask
from evaluation.evaluator import ModelEvaluator
from evaluation.error_analyzer import save_failure_report
from evaluation.rule_violation_report import save_rule_precision_recall_table
from core.logging import get_logger

logger = get_logger(__name__)


def run_sft_multisize(variant_name: str = "sft_v1", sizes: Optional[List[str]] = None,
                       max_samples: Optional[int] = None):
    task_cfgs = {t: load_task_config(t) for t in PROJECT1_TASKS}
    dataset = load_construction_dataset()
    eval_split = get_eval_split(dataset["test"], max_samples)  # SAME subsample as baseline

    base_model_ids = [mid for mid in REGISTRY.keys() if mid.endswith("-base")]
    if sizes:
        base_model_ids = [mid for mid in base_model_ids if mid.split("-")[0] in sizes]

    for base_model_id in base_model_ids:
        logger.info(f"=== Multi-task SFT: {base_model_id} ===")
        run_sft_multitask(base_model_id, task_cfgs, variant_name=variant_name)

        size_name = base_model_id.split("-")[0]
        new_model_id = f"{size_name}-{variant_name}"

        for task in PROJECT1_TASKS:
            task_cfg = task_cfgs[task]
            logger.info(f"=== Evaluating {new_model_id} on {task} ===")
            evaluator = ModelEvaluator(model_id=new_model_id, task=task, task_cfg=task_cfg)
            results = evaluator.run(eval_split, run_name=new_model_id)
            evaluator.save_results(results, filename=f"{new_model_id}_eval.csv")
            save_failure_report(results, task, filename=f"{new_model_id}_failures.csv")

            if task == "rule_violation":
                save_rule_precision_recall_table(
                    results, new_model_id, filename=f"{new_model_id}_rule_pr_table.csv"
                )

    logger.info("All multi-task SFT runs complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant_name", default="sft_v1")
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    run_sft_multisize(variant_name=args.variant_name, sizes=args.sizes, max_samples=args.max_samples)


if __name__ == "__main__":
    main()