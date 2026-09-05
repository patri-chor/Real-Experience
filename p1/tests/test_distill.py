"""Pytest suite for Task 04: SFT Trainer and Internalized Student Model."""

import json
from pathlib import Path
import pytest

from p1.src.distill.sft_trainer import train_student_model
from p1.src.distill.student_tester import (
    InternalizedStudentModel,
    analyze_internalized_cot,
)
from p1.src.memory.data_builder import build_test_cases
from p1.src.expert.evaluator import verify_correctness


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """Run SFT training into a temporary directory once for module tests."""
    temp_dir = tmp_path_factory.mktemp("student_model_test")
    dataset_file = temp_dir / "test_distill.jsonl"

    # 写入最小规范测试数据
    sample_records = [
        {
            "problem": "What is 1 + 1?",
            "category": "simple",
            "trajectory": "Final Answer: 2",
            "final_answer": "2",
        },
        {
            "problem": "A car travels from City X to City Y at a speed of 50 km/h...",
            "category": "similar",
            "trajectory": "【分析旧题】...\n【提取逻辑】...\n【修改方案】...\nFinal Answer: 60 km/h",
            "final_answer": "60 km/h",
        },
    ]
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_student_model(dataset_path=str(dataset_file), output_dir=str(temp_dir))
    return temp_dir


def test_sft_trainer_output_integrity(trained_model_dir):
    """Verify mock_model.json is correctly generated with required schema."""
    model_json_path = trained_model_dir / "mock_model.json"
    assert model_json_path.exists()

    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "InternalizedStudent-0.5B"
    assert data["total_samples"] == 2
    assert "internalized_weights" in data
    assert len(data["internalized_weights"]) == 2


def test_internalized_student_simple_inference(trained_model_dir):
    """Verify student model correctly handles simple questions directly."""
    student = InternalizedStudentModel(model_dir=str(trained_model_dir))
    resp = student.generate("What is 1 + 1?")

    assert "Final Answer: 2" in resp
    ans = student.extract_final_answer(resp)
    assert verify_correctness(ans, "2") is True


def test_internalized_student_similar_inference_and_cot(trained_model_dir):
    """Verify student model autonomously recalls and executes 3-phase cognitive reuse."""
    student = InternalizedStudentModel(model_dir=str(trained_model_dir))
    prompt = (
        "A car travels from City X to City Y at a speed of 50 km/h and returns "
        "at a speed of 75 km/h. If the distance between City X and City Y is 150 km, "
        "what is the average speed of the car for the entire round trip?"
    )
    trajectory = student.generate(prompt)

    # 验证自发回忆与核心步骤
    cot = analyze_internalized_cot(trajectory)
    assert cot["spontaneous_recall"] is True
    assert cot["analyze_old"] is True
    assert cot["extract_logic"] is True
    assert cot["modify_solution"] is True
    assert cot["is_valid_internalized_cot"] is True
    assert "分析旧题" in cot["matched_steps"]
    assert "提取逻辑" in cot["matched_steps"]
    assert "修改方案" in cot["matched_steps"]

    # 验证答案准确性
    assert verify_correctness(cot["final_answer"], "60 km/h") is True


def test_analyze_internalized_cot_negative_cases():
    """Verify cot analyzer correctly rejects incomplete trajectories."""
    bad_traj_1 = "直接计算得出结果为 60 km/h。Final Answer: 60 km/h"
    res_1 = analyze_internalized_cot(bad_traj_1)
    assert res_1["is_valid_internalized_cot"] is False

    bad_traj_2 = "【分析旧题】原题为往返题。\nFinal Answer: 60"
    res_2 = analyze_internalized_cot(bad_traj_2)
    assert res_2["is_valid_internalized_cot"] is False
