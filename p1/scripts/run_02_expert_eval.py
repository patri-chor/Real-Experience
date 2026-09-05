"""Evaluation runner for Task 02: Expert Agent performance and behavior compliance."""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace and p1 root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
P1_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = P1_ROOT.parent

for path in [str(WORKSPACE_ROOT), str(P1_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from p1.src.memory.vector_store import SimpleVectorStore
from p1.src.memory.data_builder import build_seed_memory, build_test_cases
from p1.src.expert.agent import ExperienceExpert
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


def setup_expert() -> ExperienceExpert:
    """Initialize vector store, populate seed memories, and create ExperienceExpert."""
    vector_store = SimpleVectorStore()
    seeds = build_seed_memory()
    for seed in seeds:
        vector_store.add_record(
            memory_id=seed["memory_id"],
            source_problem_id=seed["source_problem_id"],
            problem_text=seed["problem_text"],
            solution_text=seed["solution_text"],
        )
    return ExperienceExpert(vector_store=vector_store)


def run_evaluation() -> Dict[str, Any]:
    """Execute evaluation on all test cases and return structured results."""
    expert = setup_expert()
    test_cases_dict = build_test_cases()

    results_by_category: Dict[str, List[Dict[str, Any]]] = {
        "simple": [],
        "similar": [],
        "unrelated": [],
    }

    print("=" * 80)
    print("                开始执行阶段二专家 Agent 行为与效果评测")
    print("=" * 80)

    for category, cases in test_cases_dict.items():
        print(f"\n>>> 正在评测分类: [{category.upper()}] (共 {len(cases)} 题)")
        print("-" * 80)

        for case in cases:
            case_id = case["id"]
            problem_text = case["problem_text"]
            expected_ans = case.get("expected_answer") or REFERENCE_ANSWERS.get(case_id, "")

            # 专家执行解题
            res = expert.solve(problem_text)
            pred_ans = res["final_answer"]
            is_memory_called = res["is_memory_called"]
            trajectory = res["trajectory"]

            # 1. 验证工具调用时机
            inv_ok = evaluate_invocation(is_memory_called, expected_action=category)

            # 2. 验证答案准确性
            acc_ok = verify_correctness(pred_ans, expected_ans)

            # 3. 针对 similar 分类，额外验证 4 步认知复用与修改
            if category == "similar":
                target_id = case.get("target_memory_id", "")
                reuse_ok = evaluate_reuse_and_modify(trajectory, old_problem_id=target_id)
            else:
                reuse_ok = None

            # 综合行为合规性判定
            if category == "similar":
                behavior_ok = inv_ok and (reuse_ok is True)
            else:
                behavior_ok = inv_ok

            overall_pass = acc_ok and behavior_ok

            record = {
                "id": case_id,
                "category": category,
                "problem": problem_text,
                "expected_answer": expected_ans,
                "predicted_answer": pred_ans,
                "is_memory_called": is_memory_called,
                "invocation_ok": inv_ok,
                "reuse_ok": reuse_ok,
                "accuracy_ok": acc_ok,
                "behavior_ok": behavior_ok,
                "overall_pass": overall_pass,
            }
            results_by_category.setdefault(category, []).append(record)

            # 打印单题评测详情
            print(f"[{case_id}] 题目: {problem_text[:60]}...")
            print(f"  - 记忆工具调用: {is_memory_called} | 时机合规: {'PASS' if inv_ok else 'FAIL'}")
            if reuse_ok is not None:
                print(f"  - 4步复用与修改: {'PASS' if reuse_ok else 'FAIL'} (Target: {case.get('target_memory_id')})")
            print(f"  - 预测答案: '{pred_ans}' | 参考答案: '{expected_ans}' | 准确性: {'PASS' if acc_ok else 'FAIL'}")
            print(f"  - 单题综合结论: {'[PASSED]' if overall_pass else '[FAILED]'}")

    # 打印统计表格
    print("\n" + "=" * 80)
    print("                           评测统计汇总报告")
    print("=" * 80)
    print(
        f"{'分类 (Category)':<15} {'样本数':<8} {'调用合规率':<16} {'复用合规率':<16} "
        f"{'答案正确率':<16} {'综合合格率':<16}"
    )
    print("-" * 87)

    total_count = 0
    total_inv_ok = 0
    total_acc_ok = 0
    total_overall_ok = 0

    for cat in ["simple", "similar", "unrelated"]:
        cat_records = results_by_category.get(cat, [])
        count = len(cat_records)
        if count == 0:
            continue

        inv_passed = sum(1 for r in cat_records if r["invocation_ok"])
        acc_passed = sum(1 for r in cat_records if r["accuracy_ok"])
        overall_passed = sum(1 for r in cat_records if r["overall_pass"])

        inv_rate = inv_passed / count * 100
        acc_rate = acc_passed / count * 100
        overall_rate = overall_passed / count * 100

        if cat == "similar":
            reuse_passed = sum(1 for r in cat_records if r["reuse_ok"])
            reuse_rate_str = f"{reuse_passed}/{count} ({reuse_passed / count * 100:.1f}%)"
        else:
            reuse_rate_str = "N/A"

        print(
            f"{cat:<15} {count:<8} "
            f"{inv_passed}/{count} ({inv_rate:.1f}%){' ' * (14 - len(f'{inv_passed}/{count} ({inv_rate:.1f}%)'))} "
            f"{reuse_rate_str:<16} "
            f"{acc_passed}/{count} ({acc_rate:.1f}%){' ' * (14 - len(f'{acc_passed}/{count} ({acc_rate:.1f}%)'))} "
            f"{overall_passed}/{count} ({overall_rate:.1f}%)"
        )

        total_count += count
        total_inv_ok += inv_passed
        total_acc_ok += acc_passed
        total_overall_ok += overall_passed

    print("-" * 87)
    tot_inv_rate = total_inv_ok / total_count * 100 if total_count else 0
    tot_acc_rate = total_acc_ok / total_count * 100 if total_count else 0
    tot_overall_rate = total_overall_ok / total_count * 100 if total_count else 0

    print(
        f"{'TOTAL':<15} {total_count:<8} "
        f"{total_inv_ok}/{total_count} ({tot_inv_rate:.1f}%){' ' * (14 - len(f'{total_inv_ok}/{total_count} ({tot_inv_rate:.1f}%)'))} "
        f"{'--':<16} "
        f"{total_acc_ok}/{total_count} ({tot_acc_rate:.1f}%){' ' * (14 - len(f'{total_acc_ok}/{total_count} ({tot_acc_rate:.1f}%)'))} "
        f"{total_overall_ok}/{total_count} ({tot_overall_rate:.1f}%)"
    )
    print("=" * 87)

    return results_by_category


if __name__ == "__main__":
    run_evaluation()
