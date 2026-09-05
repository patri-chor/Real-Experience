"""Distillation and trajectory collection package."""

from p1.src.distill.trajectory_collector import (
    is_trajectory_qualified,
    run_expert_and_collect,
)
from p1.src.distill.sft_trainer import train_student_model
from p1.src.distill.student_tester import (
    InternalizedStudentModel,
    analyze_internalized_cot,
)

__all__ = [
    "is_trajectory_qualified",
    "run_expert_and_collect",
    "train_student_model",
    "InternalizedStudentModel",
    "analyze_internalized_cot",
]
