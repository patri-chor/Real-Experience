"""Distillation and trajectory collection package."""

from p1.src.distill.trajectory_collector import (
    is_trajectory_qualified,
    run_expert_and_collect,
)

__all__ = [
    "is_trajectory_qualified",
    "run_expert_and_collect",
]
