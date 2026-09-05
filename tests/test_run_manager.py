"""
Unit tests for ExperimentRun and rolling checkpoint management.
Verifies DEC-20260905-06 data lifecycle guarantees.
"""

from pathlib import Path
import tempfile
import torch
import pytest

from utils.run_manager import ExperimentRun, create_run


def test_create_run_and_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        run = create_run("stage1", tag="unit_test", base_dir=tmpdir)
        assert run.run_dir.exists()
        assert run.checkpoints_dir.exists()
        assert run.staging_dir.exists()
        assert run.logs_dir.exists()
        assert "stage1_" in run.run_dir.name
        assert "_unit_test" in run.run_dir.name


def test_save_audit_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        run = create_run("stage1", tag="audit_test", base_dir=tmpdir)
        report_data = {"status": "PASSED", "verified_samples": 50}
        saved_path = run.save_audit_report(report_data)
        assert saved_path.exists()
        assert "PASSED" in saved_path.read_text(encoding="utf-8")


def test_rolling_checkpoint_pruning():
    with tempfile.TemporaryDirectory() as tmpdir:
        run = create_run("stage2", tag="checkpoint_test", base_dir=tmpdir)
        
        # Save 4 sequential checkpoints with max_to_keep=2
        # Metrics: 10.0 (step 1), 8.0 (step 2), 9.0 (step 3), 7.0 (step 4)
        run.save_rolling_checkpoint({"weight": 1}, step=1, metric=10.0, mode="min", max_to_keep=2)
        run.save_rolling_checkpoint({"weight": 2}, step=2, metric=8.0, mode="min", max_to_keep=2)
        run.save_rolling_checkpoint({"weight": 3}, step=3, metric=9.0, mode="min", max_to_keep=2)
        run.save_rolling_checkpoint({"weight": 4}, step=4, metric=7.0, mode="min", max_to_keep=2)
        
        # Last should be step 4
        last_ckpt = torch.load(run.checkpoints_dir / "last_checkpoint.pt")
        assert last_ckpt["weight"] == 4
        
        # Best should be step 4 (lowest metric = 7.0)
        best_ckpt = torch.load(run.checkpoints_dir / "best_checkpoint.pt")
        assert best_ckpt["weight"] == 4
        
        # Only max_to_keep=2 numbered steps should remain (steps 3 and 4)
        step_files = list(run.checkpoints_dir.glob("checkpoint_step_*.pt"))
        assert len(step_files) == 2
        file_names = {f.name for f in step_files}
        assert "checkpoint_step_000003.pt" in file_names
        assert "checkpoint_step_000004.pt" in file_names
        # Older steps (1 and 2) must have been pruned automatically
        assert "checkpoint_step_000001.pt" not in file_names
        assert "checkpoint_step_000002.pt" not in file_names
