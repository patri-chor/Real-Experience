"""Trajectory collector module for filtering and exporting SFT distillation data."""

import json
from pathlib import Path
from typing import Dict, Any, List

from p1.src.expert.evaluator import (
    evaluate_invocation,
    evaluate_reuse_and_modify,
    verify_correctness,
)

# Reference ground-truth answers for test cases
REFERENCE_ANSWERS: Dict[str, str] = {
    "simple_001": "2",
    "simple_002": "5",
    "similar_001": "60 km/h",
    "similar_002": "10/3",
    "similar_003": "29",
    "unrelated_001": "证明完成",
    "unrelated_002": "证明完成",
}


def is_trajectory_qualified(
    result: Dict[str, Any],
    case: Dict[str, Any],
) -> bool:
    """Verify whether an execution trajectory satisfies distillation quality criteria.

    Filtering rules:
    1. Answer accuracy: verify_correctness must return True against expected ground truth.
    2. Invocation timing: evaluate_invocation must conform to category policy.
    3. 4-phase cognitive reuse: for 'similar' cases, evaluate_reuse_and_modify must return True.

    Args:
        result: Dictionary returned by expert.solve (problem, trajectory, is_memory_called, final_answer).
        case: Test case dictionary from build_test_cases.

    Returns:
        True if the trajectory meets all quality standards and should be collected, False otherwise.
    """
    category = case.get("category", "")
    case_id = case.get("id", "")
    expected_ans = case.get("expected_answer") or REFERENCE_ANSWERS.get(case_id, "")

    # 1. 验证答案准确性
    pred_ans = result.get("final_answer", "")
    if not verify_correctness(pred_ans, expected_ans):
        return False

    # 2. 验证工具调用时机
    is_memory_called = result.get("is_memory_called", False)
    if not evaluate_invocation(is_memory_called, expected_action=category):
        return False

    # 3. 针对 similar 题型，必须验证 4 步认知复用与修改
    if category == "similar":
        target_id = case.get("target_memory_id", "")
        trajectory = result.get("trajectory", "")
        if not evaluate_reuse_and_modify(trajectory, old_problem_id=target_id):
            return False

    return True


def run_expert_and_collect(
    expert,
    test_cases_dict: Dict[str, List[Dict[str, Any]]],
    output_jsonl: str,
) -> None:
    """Execute expert reasoning over test cases, filter high-quality CoT trajectories, and save to JSONL.

    Args:
        expert: Initialized ExperienceExpert instance.
        test_cases_dict: Dictionary mapping category to list of test case dictionaries.
        output_jsonl: File path where qualified JSONL records will be written.
    """
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected_records: List[Dict[str, Any]] = []
    total_cases = 0

    print("[TrajectoryCollector] 开始执行测试集推理与高质量轨迹收集...")

    for category, cases in test_cases_dict.items():
        for case in cases:
            total_cases += 1
            case_id = case.get("id", f"case_{total_cases}")
            problem_text = case.get("problem_text", "")

            # 专家执行推理
            result = expert.solve(problem_text)

            # 严格质量过滤
            qualified = is_trajectory_qualified(result, case)

            if qualified:
                record = {
                    "problem": result["problem"],
                    "category": category,
                    "trajectory": result["trajectory"],
                    "final_answer": result["final_answer"],
                }
                collected_records.append(record)
                print(f"  - [{case_id}] ({category}) 判定合格 -> 已收录")
            else:
                print(f"  - [{case_id}] ({category}) 未达到质量门禁标准 -> 已过滤")

    # 写入 JSONL 文件
    with open(out_path, "w", encoding="utf-8") as f:
        for item in collected_records:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")

    print(
        f"[TrajectoryCollector] 轨迹收集完成: 评估总数 {total_cases} 条，"
        f"成功收集 {len(collected_records)} 条，已写入目标文件: {out_path.resolve()}"
    )
