"""
Custom TrainerCallbacks for crash-safe, resumable, well-logged SFT runs.
"""
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from core.logging import get_logger

logger = get_logger(__name__)


class SaveBestModelCallback(TrainerCallback):
    """Copies the adapter to a STABLE `best/` directory the instant eval improves.

    This is independent of `save_total_limit` checkpoint rotation and independent
    of the run finishing — if Colab dies at step 951, whatever was best as of the
    last eval (e.g. step 900) is already sitting in `best/` on Drive.
    """

    def __init__(self, best_dir: str, metric_name: str = "eval_loss", greater_is_better: bool = False):
        self.best_dir = Path(best_dir)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_value: Optional[float] = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", {})
        current = metrics.get(self.metric_name)
        if current is None:
            return

        improved = (
            self.best_value is None
            or (self.greater_is_better and current > self.best_value)
            or (not self.greater_is_better and current < self.best_value)
        )
        if not improved:
            return

        self.best_value = current
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None:
            return

        self.best_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"New best {self.metric_name}={current:.4f} at step {state.global_step} "
            f"-> saving to {self.best_dir}"
        )
        model.save_pretrained(str(self.best_dir))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(self.best_dir))

        tmp_path = self.best_dir / "best_info.tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "metric_name": self.metric_name,
                "metric_value": current,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        os.replace(str(tmp_path), str(self.best_dir / "best_info.json"))


class ManifestUpdateCallback(TrainerCallback):
    """Writes training_state.json on every save/eval, including the W&B run id.

    This is what makes resume-after-disconnect actually work: the wandb_run_id is
    persisted the moment the run starts, not after training completes.
    """

    def __init__(self, checkpoint_dir: str, static_fields: dict):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.static_fields = static_fields
        self.state_path = self.checkpoint_dir / "training_state.json"

    def _write(self, state: TrainerState, status: str):
        payload = dict(self.static_fields)
        payload.update({
            "status": status,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        import tempfile
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(str(tmp_path), str(self.state_path))

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(state, status="in_progress")

    def on_save(self, args, state, control, **kwargs):
        self._write(state, status="in_progress")

    def on_evaluate(self, args, state, control, **kwargs):
        self._write(state, status="in_progress")

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, status="completed")


class GPUMemoryLoggingCallback(TrainerCallback):
    """Logs allocated/reserved CUDA memory to W&B every N steps — cheap early warning
    for creeping fragmentation before it becomes an OOM."""

    def __init__(self, every_n_steps: int = 20):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0 or state.global_step % self.every_n_steps != 0:
            return
        try:
            import torch
            import wandb
            if torch.cuda.is_available() and wandb.run is not None:
                wandb.log({
                    "gpu/allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu/reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu/max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }, step=state.global_step)
        except Exception:
            pass