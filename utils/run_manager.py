"""
Experiment lifecycle and temporary data manager.
Implements DEC-20260905-06 storage tier and rolling checkpoint protocols.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_manager")


class ExperimentRun:
    """
    Manages an isolated experiment run directory structure:
    runs/{stage}_{timestamp}_{tag}/
        ├── checkpoints/
        ├── staging/
        ├── logs/
        └── audit_report.json
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.staging_dir = run_dir / "staging"
        self.logs_dir = run_dir / "logs"
        self.audit_report_path = run_dir / "audit_report.json"

        # Initialize directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric: float | None = None
        self._step_checkpoints: list[tuple[int, Path]] = []

    def save_audit_report(self, report_dict: dict[str, Any]) -> Path:
        """Save quality check audit report into the run folder."""
        self.audit_report_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved audit report to {self.audit_report_path}")
        return self.audit_report_path

    def save_rolling_checkpoint(
        self,
        checkpoint_data: Any,
        step: int,
        metric: float | None = None,
        mode: str = "min",
        max_to_keep: int = 2,
    ) -> dict[str, Path]:
        """
        Save rolling checkpoint with automatic pruning to prevent disk bloat.
        Maintains:
            - last_adapter / last_checkpoint
            - best_adapter / best_checkpoint (if metric provided)
            - at most max_to_keep historical step checkpoints
        """
        import torch

        saved_paths: dict[str, Path] = {}

        # 1. Save last checkpoint
        last_path = self.checkpoints_dir / "last_checkpoint.pt"
        torch.save(checkpoint_data, last_path)
        saved_paths["last"] = last_path

        # 2. Check if this is the best checkpoint so far
        is_best = False
        if metric is not None:
            if self._best_metric is None:
                is_best = True
            elif mode == "min" and metric < self._best_metric:
                is_best = True
            elif mode == "max" and metric > self._best_metric:
                is_best = True

            if is_best:
                self._best_metric = metric
                best_path = self.checkpoints_dir / "best_checkpoint.pt"
                torch.save(checkpoint_data, best_path)
                saved_paths["best"] = best_path
                logger.info(f"New best checkpoint at step {step} with metric {metric:.4f}")

        # 3. Save numbered step checkpoint
        step_path = self.checkpoints_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save(checkpoint_data, step_path)
        self._step_checkpoints.append((step, step_path))
        saved_paths["step"] = step_path

        # 4. Prune older step checkpoints exceeding max_to_keep
        while len(self._step_checkpoints) > max_to_keep:
            old_step, old_path = self._step_checkpoints.pop(0)
            if old_path.exists():
                try:
                    old_path.unlink()
                    logger.debug(f"Pruned old checkpoint: {old_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to prune old checkpoint {old_path}: {e}")

        return saved_paths

    def clean_staging(self) -> None:
        """Remove temporary staging files after task completion."""
        if self.staging_dir.exists():
            for item in self.staging_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info(f"Cleaned staging directory: {self.staging_dir}")


def create_run(
    stage_name: str,
    tag: str = "run",
    base_dir: str | Path = "runs",
) -> ExperimentRun:
    """
    Factory function to create a new timestamped ExperimentRun.
    Example output directory: runs/stage1_20260905_150530_sample/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage = stage_name.replace(" ", "_").lower()
    safe_tag = tag.replace(" ", "_").lower()
    dir_name = f"{safe_stage}_{timestamp}_{safe_tag}"

    base_path = Path(base_dir).resolve()
    run_dir = base_path / dir_name
    return ExperimentRun(run_dir)
