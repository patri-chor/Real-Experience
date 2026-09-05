"""Evaluation suite for memory invocation timing, trajectory reuse, and answer correctness."""

import re
from fractions import Fraction
from typing import Optional


def evaluate_invocation(is_memory_called: bool, expected_action: str) -> bool:
    """Verify whether the memory tool was invoked at the correct timing.

    Args:
        is_memory_called: Actual boolean indicating whether memory tool was called.
        expected_action: Expected policy or category name ("similar", "simple", "call", "no_call", etc.).

    Returns:
        True if action conforms to expectation, False otherwise.
    """
    action = expected_action.strip().lower()

    # 复杂/相似类题型：强制要求必须调用
    if action in ("similar", "call", "must_call", "invoke", "true", "yes"):
        return is_memory_called is True

    # 简单/直接类题型：强制要求绝不调用
    elif action in ("simple", "no_call", "skip", "direct", "false", "no"):
        return is_memory_called is False

    # 无关类题型：若配置为 unrelated，检查是否为 False（避免对无关问题发生工具幻觉）
    elif action in ("unrelated",):
        return is_memory_called is False

    raise ValueError(f"Unknown expected_action specification: '{expected_action}'")


def evaluate_reuse_and_modify(trajectory: str, old_problem_id: str) -> bool:
    """Verify whether the reasoning trajectory properly executes the 4-phase reuse loop.

    Checks for:
    1. Reference to historical problem ID or memory seed.
    2. Phase 1: Analyzing the old problem (分析旧题 / 历史案例).
    3. Phase 2: Extracting reusable logic (提取可复用逻辑 / 公式 / 核心逻辑).
    4. Phase 3: Identifying differences (识别新旧差异 / 参数差异 / 区别).
    5. Phase 4: Adapting and verifying (修改旧方案 / 调整计算 / 验算).

    Args:
        trajectory: The full reasoning trajectory string.
        old_problem_id: Expected source problem ID or memory ID (e.g. "seed_speed_round_trip", "mem_001").

    Returns:
        True if all required stages are present in trajectory, False otherwise.
    """
    if not trajectory or not old_problem_id:
        return False

    traj_lower = trajectory.lower()
    pid_lower = old_problem_id.lower()

    # 1. 验证是否准确关联并引用了对应旧题来源标识
    has_id_reference = pid_lower in traj_lower
    if not has_id_reference:
        # 兼容性检查：若传入的是 seed_xxx，检查对应 mem_xxx 或相似关键词
        alias_map = {
            "seed_speed_round_trip": "mem_001",
            "seed_collaborative_work": "mem_002",
            "seed_consecutive_sum": "mem_003",
            "seed_sequential_discount": "mem_004",
        }
        mapped_id = alias_map.get(old_problem_id, "")
        if mapped_id and mapped_id.lower() in traj_lower:
            has_id_reference = True

    if not has_id_reference:
        return False

    # 2. 验证阶段 1：分析旧题
    has_phase1 = any(k in trajectory for k in ["分析旧题", "旧题", "历史案例", "原题分析"])

    # 3. 验证阶段 2：提取可复用逻辑
    has_phase2 = any(k in trajectory for k in ["提取可复用逻辑", "可复用", "核心逻辑", "通用公式", "解法骨架"])

    # 4. 验证阶段 3：识别新旧差异
    has_phase3 = any(k in trajectory for k in ["识别新旧差异", "差异", "不同点", "参数不同", "新条件"])

    # 5. 验证阶段 4：修改旧方案并验证
    has_phase4 = any(k in trajectory for k in ["修改旧方案", "修改旧解法", "代入新参数", "调整计算", "验算", "验证"])

    return has_phase1 and has_phase2 and has_phase3 and has_phase4


def verify_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """Verify correctness of predicted answer against ground truth.

    Supports:
    - Direct normalized string match.
    - Numerical value equality (handling floats, fractions, currencies, and units like km/h, hours).
    - Substring containment for verbose answers.

    Args:
        predicted_answer: Model output answer string.
        ground_truth: Reference expected answer.

    Returns:
        True if answer is mathematically or semantically equivalent, False otherwise.
    """
    if predicted_answer is None or ground_truth is None:
        return False

    p_clean = predicted_answer.strip().lower()
    g_clean = ground_truth.strip().lower()

    # 1. 直接字符串完全匹配
    if p_clean == g_clean:
        return True

    # 2. 数值与分数解析比对
    p_nums = _extract_numbers(p_clean)
    g_nums = _extract_numbers(g_clean)

    if p_nums and g_nums:
        # 取最后一个数字作为最终结论比对
        val_pred = p_nums[-1]
        val_gt = g_nums[-1]
        # 绝对容差或相对容差比对
        if abs(val_pred - val_gt) < 1e-3:
            return True
        if abs(val_gt) > 1e-6 and abs(val_pred - val_gt) / abs(val_gt) < 1e-3:
            return True

    # 3. 剥离常用单位后的文本包含与匹配
    def _strip_units(s: str) -> str:
        s = re.sub(r"[\$,\*\`]", "", s)
        for unit in ["km/h", "hours", "hour", "km", "miles", "小时"]:
            s = s.replace(unit, "")
        return s.strip()

    p_stripped = _strip_units(p_clean)
    g_stripped = _strip_units(g_clean)
    if p_stripped == g_stripped:
        return True

    return g_stripped in p_stripped if g_stripped else False


def _extract_numbers(text: str) -> list[float]:
    """Helper to extract numbers including fractions like 10/3."""
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    results = []
    for t in tokens:
        try:
            if "/" in t:
                results.append(float(Fraction(t)))
            else:
                results.append(float(t))
        except Exception:
            continue
    return results
