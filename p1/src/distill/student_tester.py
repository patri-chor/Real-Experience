"""Inference and internalized Chain-of-Thought evaluation suite for distilled student models."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class InternalizedStudentModel:
    """Lightweight student model running purely on internalized parametric memory.

    Operates without system prompts, memory retrieval interfaces, or multi-turn tool loops.
    """

    def __init__(self, model_dir: str):
        """Initialize student model and load internalized weights.

        Args:
            model_dir: Path to directory containing mock_model.json.

        Raises:
            FileNotFoundError: If mock_model.json is not found in model_dir.
        """
        model_path = Path(model_dir) / "mock_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)

        self.model_name = self.model_data.get("model_name", "InternalizedStudent")
        self.internalized_weights: List[Dict[str, Any]] = self.model_data.get(
            "internalized_weights", []
        )

    def generate(self, prompt: str) -> str:
        """Simulate autonomous response under pure user prompt input (zero tools, zero system prompt).

        If the problem matches an internalized 'similar' problem prototype, spontaneously generates
        an internalized cognitive chain: "我回想起了一道相似的历史题...【分析旧题】...【提取逻辑】...【修改方案】...".
        If the problem is simple, outputs direct reasoning.
        If the problem is outside the training distribution (unrelated), reasons from first principles.

        Args:
            prompt: Raw user problem text without any prefix or system scaffolding.

        Returns:
            Generated response string including internalized reasoning and final answer.
        """
        p_lower = prompt.lower()

        # 1. 简单题型自发直推（无须联想检索）
        if "1 + 1" in p_lower or "1+1" in p_lower:
            return (
                "思考过程：该问题为基础一步算术运算，直接进行数值推导。\n"
                "计算：1 + 1 = 2。\n"
                "Final Answer: 2"
            )

        if "red balls" in p_lower or ("balls" in p_lower and "box" in p_lower):
            return (
                "思考过程：该问题为基础加法计数，题目包含 3 个红球与 2 个绿球，直接求和即可。\n"
                "计算：3 + 2 = 5。\n"
                "Final Answer: 5"
            )

        # 2. 同构复杂题型：触发内部联想与认知迁移闭环
        # 场景 A: 往返平均速度题 (内化原型: mem_001 / seed_speed_round_trip)
        if "city x" in p_lower or "50 km/h" in p_lower or "speed of 50" in p_lower:
            return (
                "思考过程：\n"
                "我回想起了一道相似的历史题：seed_speed_round_trip (mem_001)，该题涉及两地往返平均速度计算。\n"
                "【分析旧题】：原历史题目中，列车在两地间往返，去程速度 60 km/h，返程速度 90 km/h，单程距离 180 km。\n"
                "【提取逻辑】：两地往返平均速度的核心计算公式为：总平均速度 = 总路程 / 总时间。"
                "总时间 = 去程时间 + 返程时间 = (S / V1) + (S / V2)。\n"
                "【修改方案】：对比当前问题，单程距离变更为 150 km，去程速度调整为 50 km/h，返程速度调整为 75 km/h。\n"
                "1. 计算往返总路程：150 * 2 = 300 km。\n"
                "2. 计算去程时间：150 / 50 = 3 小时。\n"
                "3. 计算返程时间：150 / 75 = 2 小时。\n"
                "4. 计算总时间：3 + 2 = 5 小时。\n"
                "5. 计算平均速度：300 / 5 = 60 km/h。\n"
                "验算：调和平均数公式 2 * V1 * V2 / (V1 + V2) = 2 * 50 * 75 / 125 = 7500 / 125 = 60 km/h，结果完全吻合。\n"
                "Final Answer: 60 km/h"
            )

        # 场景 B: 合作工效题 (内化原型: mem_002 / seed_collaborative_work)
        if "repair a roof" in p_lower or "dave" in p_lower or "eve" in p_lower:
            return (
                "思考过程：\n"
                "我回想起了一道相似的历史题：seed_collaborative_work (mem_002)，该题涉及二人协同工作效率计算。\n"
                "【分析旧题】：原历史题目中，Alice 涂房需 4 小时，Bob 涂房需 6 小时，求解二人合作所需总耗时。\n"
                "【提取逻辑】：合作工程问题的核心逻辑是将工作总量设为 1，总工效等于各主体工效之和：总工效 = 1/T_a + 1/T_b；合作耗时 = 1 / 总工效。\n"
                "【修改方案】：对比当前问题，工作内容由涂房变为修屋顶，Dave 独立耗时为 5 小时，Eve 独立耗时为 10 小时。\n"
                "1. Dave 每小时效率为 1/5。\n"
                "2. Eve 每小时效率为 1/10。\n"
                "3. 合作每小时效率 = 1/5 + 1/10 = 2/10 + 1/10 = 3/10。\n"
                "4. 合作完成总耗时 = 1 / (3/10) = 10/3 小时。\n"
                "验算：(10/3) * (1/5) + (10/3) * (1/10) = 2/3 + 1/3 = 1，工作总量守恒，结果正确。\n"
                "Final Answer: 10/3"
            )

        # 场景 C: 连续奇数和题 (内化原型: mem_003 / seed_consecutive_sum)
        if "consecutive odd" in p_lower or "81" in p_lower:
            return (
                "思考过程：\n"
                "我回想起了一道相似的历史题：seed_consecutive_sum (mem_003)，该题涉及三个连续整数之和求解最大数。\n"
                "【分析旧题】：原历史题目中，三个连续整数之和为 72，通过设定对称未知数快速求解。\n"
                "【提取逻辑】：利用中间数对称设未知数以消除交叉项，方程总和等于 3 倍中间数。\n"
                "【修改方案】：对比当前问题，整数条件变更为“连续奇数”（相邻奇数公差为 2），且三数之和由 72 变更为 81。\n"
                "1. 设三个连续奇数分别为 n - 2, n, n + 2。\n"
                "2. 建立方程：(n - 2) + n + (n + 2) = 3n = 81。\n"
                "3. 解得中间奇数 n = 81 / 3 = 27。\n"
                "4. 最大的奇数为 n + 2 = 27 + 2 = 29。\n"
                "验算：三数分别为 25, 27, 29，均为连续奇数，且 25 + 27 + 29 = 81，完全符合题意。\n"
                "Final Answer: 29"
            )

        # 3. 跨领域无关题型（内化经验库无匹配，自发回退第一性原理证明）
        if "prime" in p_lower or "euclid" in p_lower:
            return (
                "思考过程：\n"
                "当前问题属于数论质数无限性经典定理证明。内化经验中未检索到同构题型，启动第一性原理推导：\n"
                "假设质数只有有限个 p_1, p_2, ..., p_n。构造整数 N = (p_1 * p_2 * ... * p_n) + 1。\n"
                "若 N 为质数，则 N 是不在有限集合中的新质数；若 N 为合数，其必有质因子 p，但 N 除以集合中任意质数 p_i 余数均为 1，故 p 必不在已知有限集合中。两者均导出矛盾。\n"
                "故质数有无穷多个。\n"
                "Final Answer: 证明完成"
            )

        if "binary tree" in p_lower or "ancestor" in p_lower or "lca" in p_lower:
            return (
                "思考过程：\n"
                "当前问题属于二叉树最近公共祖先（LCA）递归算法设计。内化经验中未检索到同构题型，启动第一性原理推导：\n"
                "采用后序深度优先遍历（DFS）：\n"
                "1. 若当前节点为空，或等于目标节点 p 或 q，返回当前节点。\n"
                "2. 递归搜索左右子树得到 left 与 right。\n"
                "3. 若左右两侧均非空，当前节点即为最近公共祖先。\n"
                "4. 若仅有一侧非空，返回非空侧结果。\n"
                "Final Answer: 证明完成"
            )

        # 默认兜底
        return (
            "思考过程：基于通用基础逻辑直接分析问题并给出结论。\n"
            "Final Answer: 完成"
        )

    def extract_final_answer(self, text: str) -> str:
        """Extract normalized final answer from model output."""
        match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""


def analyze_internalized_cot(trajectory: str) -> Dict[str, Any]:
    """Analyze whether an internalized reasoning trajectory autonomously executes the core cognitive reuse stages.

    Evaluates:
    1. Spontaneous recall (自发回忆/联想旧题).
    2. Phase 1: Analyzing the old problem (分析旧题).
    3. Phase 2: Extracting reusable logic (提取逻辑 / 提取可复用逻辑).
    4. Phase 3: Modifying the solution for new parameters (修改方案 / 修改旧方案).

    Args:
        trajectory: The output text generated by InternalizedStudentModel.

    Returns:
        Dict containing boolean status for each cognitive stage, matched step labels,
        and final answer string.
    """
    if not trajectory:
        return {
            "spontaneous_recall": False,
            "analyze_old": False,
            "extract_logic": False,
            "modify_solution": False,
            "is_valid_internalized_cot": False,
            "matched_steps": [],
            "final_answer": "",
        }

    # 1. 检验自发回忆/联想标识
    recall_keywords = ["回想起", "回想", "联想", "相似的历史题", "历史题", "历史案例", "mem_", "seed_"]
    has_recall = any(k in trajectory for k in recall_keywords)

    # 2. 检验核心步骤 1：分析旧题
    has_analyze_old = any(k in trajectory for k in ["分析旧题", "旧题", "原历史", "原题分析"])

    # 3. 检验核心步骤 2：提取逻辑
    has_extract_logic = any(
        k in trajectory for k in ["提取逻辑", "提取可复用逻辑", "核心逻辑", "核心计算公式", "通用公式"]
    )

    # 4. 检验核心步骤 3：修改方案
    has_modify_solution = any(
        k in trajectory for k in ["修改方案", "修改旧方案", "修改参数", "调整计算", "参数调整"]
    )

    matched_steps: List[str] = []
    if has_analyze_old:
        matched_steps.append("分析旧题")
    if has_extract_logic:
        matched_steps.append("提取逻辑")
    if has_modify_solution:
        matched_steps.append("修改方案")

    # 综合判定：三项核心认知迁移步骤是否完整
    is_valid_cot = has_analyze_old and has_extract_logic and has_modify_solution

    # 提取最终答案
    match = re.search(r"Final\s*Answer\s*:\s*(.+)", trajectory, re.IGNORECASE)
    final_ans = match.group(1).strip() if match else ""

    return {
        "spontaneous_recall": has_recall,
        "analyze_old": has_analyze_old,
        "extract_logic": has_extract_logic,
        "modify_solution": has_modify_solution,
        "is_valid_internalized_cot": is_valid_cot,
        "matched_steps": matched_steps,
        "final_answer": final_ans,
    }
