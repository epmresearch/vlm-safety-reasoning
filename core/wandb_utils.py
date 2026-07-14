"""
Thin wrapper around Weights & Biases so training/eval scripts don't
call the wandb API directly (makes it easy to swap loggers later).

Supports:
  - Run initialization with optional resume
  - Metric and artifact logging
  - Checkpoint artifact upload
"""
import os
from typing import Any, Dict, Optional

import wandb

from core.config import load_base_config
from core.logging import get_logger

logger = get_logger(__name__)


def init_run(
    study_name: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
):
    """Initializes a W&B run.

    Args:
        study_name: Group name for organizing runs.
        run_name: Human-readable run name.
        config: Hyperparameters to log.
        run_id: Existing run ID for resumption.
        resume: Resume mode ("must", "allow", "never", or None).

    Returns:
        The wandb Run object.
    """
    base_cfg = load_base_config()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        logger.warning(
            "WANDB_API_KEY not set in environment — assuming already logged in."
        )

    kwargs = {
        "project": base_cfg["wandb_project"],
        "entity": base_cfg.get("wandb_entity"),
        "group": study_name,
        "name": run_name,
        "config": config or {},
        "reinit": True,
    }
    if run_id:
        kwargs["id"] = run_id
    if resume:
        kwargs["resume"] = resume

    run = wandb.init(**kwargs)
    return run


def log_evaluation_results(results: Dict[str, Any], run=None) -> None:
    """Logs evaluation metrics to W&B."""
    target = run or wandb
    target.log(results)


def log_artifact(
    local_path: str,
    artifact_name: str,
    artifact_type: str,
    run=None,
) -> None:
    """Logs a file as a W&B artifact.

    Args:
        local_path: Path to the file to upload.
        artifact_name: Name for the artifact.
        artifact_type: Type (e.g., "model", "dataset", "results").
        run: Optional W&B run (uses current run if None).
    """
    active_run = run or wandb.run
    if active_run is None:
        logger.warning("No active W&B run — skipping artifact logging.")
        return

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(local_path)
    active_run.log_artifact(artifact)
    logger.info(
        f"Logged artifact '{artifact_name}' ({artifact_type}) from {local_path}"
    )


def log_checkpoint_artifact(
    checkpoint_dir: str,
    artifact_name: str,
    run=None,
) -> None:
    """Logs an entire checkpoint directory as a W&B artifact.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        artifact_name: Name for the artifact.
        run: Optional W&B run.
    """
    active_run = run or wandb.run
    if active_run is None:
        logger.warning("No active W&B run — skipping checkpoint artifact.")
        return

    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_dir(checkpoint_dir)
    active_run.log_artifact(artifact)
    logger.info(f"Logged checkpoint artifact '{artifact_name}' from {checkpoint_dir}")


def finish_run(run=None) -> None:
    """Finishes the W&B run."""
    target = run or wandb
    target.finish()