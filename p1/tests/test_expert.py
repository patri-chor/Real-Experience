"""Comprehensive pytest suite for Task 02: ExperienceExpert and evaluator."""

import pytest
from p1.src.memory.vector_store import SimpleVectorStore
from p1.src.memory.data_builder import build_seed_memory, build_test_cases
from p1.src.expert.prompts import EXPERT_SYSTEM_PROMPT, SEARCH_MEMORY_TOOL_DEFINITION
from p1.src.expert.agent import ExperienceExpert, MockLLMClient
from p1.src.expert.evaluator import (
    evaluate_invocation,
    evaluate_reuse_and_modify,
    verify_correctness,
)


@pytest.fixture
def populated_vector_store():
    store = SimpleVectorStore()
    seeds = build_seed_memory()
    for s in seeds:
        store.add_record(
            memory_id=s["memory_id"],
            source_problem_id=s["source_problem_id"],
            problem_text=s["problem_text"],
            solution_text=s["solution_text"],
        )
    return store


@pytest.fixture
def expert_agent(populated_vector_store):
    return ExperienceExpert(vector_store=populated_vector_store)


def test_prompt_requirements():
    """Verify EXPERT_SYSTEM_PROMPT contains all required guidelines."""
    assert "search_memory" in EXPERT_SYSTEM_PROMPT
    assert "分析旧题" in EXPERT_SYSTEM_PROMPT
    assert "提取可复用逻辑" in EXPERT_SYSTEM_PROMPT
    assert "识别新旧差异" in EXPERT_SYSTEM_PROMPT
    assert "修改旧方案" in EXPERT_SYSTEM_PROMPT
    assert SEARCH_MEMORY_TOOL_DEFINITION["function"]["name"] == "search_memory"


def test_tool_execution(expert_agent):
    """Verify _execute_tool returns formatted memory records."""
    out = expert_agent._execute_tool("search_memory", {"query": "average speed round trip", "top_k": 1})
    assert "mem_001" in out
    assert "seed_speed_round_trip" in out
    assert "solution_text" in out


def test_solve_simple_cases(expert_agent):
    """Verify simple questions trigger direct reasoning without memory invocation."""
    test_cases = build_test_cases()["simple"]
    for case in test_cases:
        res = expert_agent.solve(case["problem_text"])
        assert res["is_memory_called"] is False
        assert evaluate_invocation(res["is_memory_called"], expected_action="simple") is True
        assert verify_correctness(res["final_answer"], case["expected_answer"]) is True


def test_solve_similar_cases(expert_agent):
    """Verify similar questions trigger memory tool, execute 4-step CoT, and solve accurately."""
    test_cases = build_test_cases()["similar"]
    expected_answers = {
        "similar_001": "60 km/h",
        "similar_002": "10/3",
        "similar_003": "29",
    }
    for case in test_cases:
        res = expert_agent.solve(case["problem_text"])
        assert res["is_memory_called"] is True
        assert evaluate_invocation(res["is_memory_called"], expected_action="similar") is True

        # 检查轨迹中的 4 步认知链与旧题标识
        target_id = case["target_memory_id"]
        assert evaluate_reuse_and_modify(res["trajectory"], old_problem_id=target_id) is True

        # 检查答案准确度
        exp_ans = expected_answers[case["id"]]
        assert verify_correctness(res["final_answer"], exp_ans) is True


def test_solve_unrelated_cases(expert_agent):
    """Verify unrelated questions do not force false reuse."""
    test_cases = build_test_cases()["unrelated"]
    for case in test_cases:
        res = expert_agent.solve(case["problem_text"])
        assert "独立" in res["trajectory"] or "声明" in res["trajectory"]


def test_evaluator_positive_and_negative():
    """Verify evaluator edge cases and boundary rejection."""
    # 1. evaluate_invocation
    assert evaluate_invocation(True, "similar") is True
    assert evaluate_invocation(False, "similar") is False
    assert evaluate_invocation(False, "simple") is True
    assert evaluate_invocation(True, "simple") is False

    # 2. evaluate_reuse_and_modify
    mock_good_traj = (
        "分析旧题: 对应 seed_speed_round_trip\n"
        "提取可复用逻辑: 平均速度公式\n"
        "识别新旧差异: 距离由 180 改为 150\n"
        "修改旧方案并验证: 代入计算并验算通过"
    )
    assert evaluate_reuse_and_modify(mock_good_traj, "seed_speed_round_trip") is True
    mock_bad_traj = "直接套用旧题 seed_speed_round_trip 答案，未作差异修改。"
    assert evaluate_reuse_and_modify(mock_bad_traj, "seed_speed_round_trip") is False

    # 3. verify_correctness
    assert verify_correctness("60 km/h", "60") is True
    assert verify_correctness("Final Answer: $81.00", "81") is True
    assert verify_correctness("3.33333 hours", "10/3") is True
    assert verify_correctness("25", "29") is False
