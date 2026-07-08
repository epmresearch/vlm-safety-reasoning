"""
Thin wrapper around Weights & Biases so training/eval scripts don't
call the wandb API directly (makes it easy to swap loggers later).
"""
import os
from typing import Any, Dict, Optional

import wandb

from core.config import load_base_config
from core.logging import get_logger

logger = get_logger(__name__)


def init_run(study_name: str, run_name: str, config: Optional[Dict[str, Any]] = None):
    base_cfg = load_base_config()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.warning("WANDB_API_KEY not set in environment — assuming already logged in.")

    run = wandb.init(
        project=base_cfg["wandb_project"],
        entity=base_cfg.get("wandb_entity"),
        group=study_name,
        name=run_name,
        config=config or {},
        reinit=True,
    )
    return run


def log_evaluation_results(results: Dict[str, Any], run=None) -> None:
    target = run or wandb
    target.log(results)


def log_artifact(local_path: str, artifact_name: str, artifact_type: str, run=None) -> None:
    target = run or wandb
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(local_path)
    (run or wandb.run).log_artifact(artifact)
    logger.info(f"Logged artifact '{artifact_name}' ({artifact_type}) from {local_path}")


def finish_run(run=None) -> None:
    (run or wandb).finish()