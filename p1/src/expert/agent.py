"""ExperienceExpert agent implementation supporting multi-turn tool calling and mock LLM simulation."""

from typing import Dict, Any, List, Optional
import json
import re
from p1.src.expert.prompts import EXPERT_SYSTEM_PROMPT, SEARCH_MEMORY_TOOL_DEFINITION


class ToolCall:
    """Lightweight abstraction for a model-requested tool invocation."""

    def __init__(self, call_id: str, name: str, arguments: Dict[str, Any]):
        self.id = call_id
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


class ModelResponse:
    """Lightweight representation of an LLM generation response."""

    def __init__(
        self,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class MockLLMClient:
    """Deterministic LLM simulator implementing multi-turn tool calling and CoT generation.

    Emulates model behaviors across:
    1. 'simple': direct answer without tool calls.
    2. 'similar': Turn 1 emits search_memory tool call; Turn 2 receives memory and
       emits the 4-phase CoT (分析旧题 -> 提取可复用逻辑 -> 识别新旧差异 -> 修改旧方案并验证).
    3. 'unrelated': recognizes irrelevance and reasons from first principles.
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        user_problem = ""
        last_tool_result = ""
        turn_count = 0

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                user_problem = msg.get("content", "")
            elif role == "tool":
                last_tool_result = msg.get("content", "")
            elif role == "assistant":
                turn_count += 1

        prob_lower = user_problem.lower()

        # 1. 简单题分类处理：直接作答，不调用任何工具
        if self._is_simple_problem(prob_lower):
            content = self._solve_simple(user_problem)
            return ModelResponse(content=content, tool_calls=[])

        # 2. 如果上一条是工具执行结果（Turn 2）：进行经验复用或独立推导
        if last_tool_result:
            content = self._generate_turn2_cot(user_problem, last_tool_result)
            return ModelResponse(content=content, tool_calls=[])

        # 3. 复杂题/相似题/未知题首次处理（Turn 1）：触发 search_memory 工具调用
        if turn_count == 0:
            call = ToolCall(
                call_id="call_mem_001",
                name="search_memory",
                arguments={"query": user_problem, "top_k": 1},
            )
            thought = (
                "内部思考：当前问题包含多步骤逻辑与变量推导，属于复杂问题。"
                "根据准则，尝试调用 search_memory 检索历史相似案例以提高解题精度与复用经验。"
            )
            return ModelResponse(content=thought, tool_calls=[call])

        # 兜底
        return ModelResponse(
            content=f"思考完毕。\nFinal Answer: 未知问题推导完成", tool_calls=[]
        )

    def _is_simple_problem(self, text: str) -> bool:
        keywords = ["1 + 1", "1+1", "red balls", "in a box", "how many balls", "trivial"]
        return any(k in text for k in keywords)

    def _solve_simple(self, text: str) -> str:
        if "1 + 1" in text or "1+1" in text:
            return (
                "内部思考：该问题为基础一步算术题，无复杂逻辑，直接进行快速计算即可，无需检索外部记忆。\n"
                "计算：1 + 1 = 2。\n"
                "Final Answer: 2"
            )
        elif "red balls" in text or "balls" in text:
            return (
                "内部思考：该问题为基础加法计数，题目给出 3 个红球和 2 个绿球，直接求和即可，无需调用记忆工具。\n"
                "计算过程：3 + 2 = 5。\n"
                "Final Answer: 5"
            )
        return (
            "内部思考：判定为简单直接问题，独立推导作答。\n"
            "Final Answer: 42"
        )

    def _generate_turn2_cot(self, user_problem: str, tool_result: str) -> str:
        prob_lower = user_problem.lower()

        # 场景 A: 往返平均速度题 (similar_001 -> mem_001)
        if "city x" in prob_lower or "50 km/h" in prob_lower or "speed of 50" in prob_lower:
            return (
                "内部思考：已获取检索结果。\n"
                "【分析旧题】：检索到的历史案例为 source_problem_id: seed_speed_round_trip (mem_001)。"
                "旧题为两地往返平均速度问题（原距离 180 km，速度分别为 60 km/h 与 90 km/h）。\n"
                "【提取可复用逻辑】：两地往返总平均速度公式为：平均速度 = 总路程 / 总时间。"
                "总时间 = 去程时间 + 返程时间 = (S / V1) + (S / V2)。\n"
                "【识别新旧差异】：旧题单程路程为 180 km，速度为 60 与 90 km/h；"
                "新问题单程路程变更为 150 km，去程速度为 50 km/h，返程速度为 75 km/h。\n"
                "【修改旧方案并验证】：\n"
                "1. 计算总往返路程：150 * 2 = 300 km。\n"
                "2. 计算去程时间：150 / 50 = 3 小时。\n"
                "3. 计算返程时间：150 / 75 = 2 小时。\n"
                "4. 计算总耗时：3 + 2 = 5 小时。\n"
                "5. 计算平均速度：300 / 5 = 60 km/h。\n"
                "验算：调和平均数公式 2 * V1 * V2 / (V1 + V2) = 2 * 50 * 75 / 125 = 7500 / 125 = 60 km/h，结果完全吻合。\n"
                "Final Answer: 60 km/h"
            )

        # 场景 B: 合作修屋顶题 (similar_002 -> mem_002)
        elif "roof" in prob_lower or "dave" in prob_lower or "worker eve" in prob_lower:
            return (
                "内部思考：已获取检索结果。\n"
                "【分析旧题】：检索到的历史案例为 source_problem_id: seed_collaborative_work (mem_002)。"
                "旧题为二人合作刷漆效率题（Alice 4h，Bob 6h）。\n"
                "【提取可复用逻辑】：工程合作问题的核心逻辑是将工作总量视为 1，"
                "总工效 = 工人 A 工效 + 工人 B 工效 = 1/T_a + 1/T_b；总耗时 = 1 / 总工效。\n"
                "【识别新旧差异】：旧题主体为刷墙，时间为 4h 和 6h；"
                "新问题主体为修屋顶，Dave 独立耗时 5h，Eve 独立耗时 10h，工程参数发生改变。\n"
                "【修改旧方案并验证】：\n"
                "1. Dave 每小时效率为 1/5。\n"
                "2. Eve 每小时效率为 1/10。\n"
                "3. 合作每小时效率 = 1/5 + 1/10 = 2/10 + 1/10 = 3/10。\n"
                "4. 合作总时间 = 1 / (3/10) = 10/3 小时（约 3.33 小时）。\n"
                "验算：(10/3) * (1/5) + (10/3) * (1/10) = 2/3 + 1/3 = 1，工作量守恒，结果正确。\n"
                "Final Answer: 10/3"
            )

        # 场景 C: 连续奇数和题 (similar_003 -> mem_003)
        elif "consecutive odd" in prob_lower or "81" in prob_lower:
            return (
                "内部思考：已获取检索结果。\n"
                "【分析旧题】：检索到的历史案例为 source_problem_id: seed_consecutive_sum (mem_003)。"
                "旧题为三个连续整数和为 72，求最大整数。\n"
                "【提取可复用逻辑】：设中间项对称表示各数，令其对称分布于均值两侧消除交叉项快速求解。\n"
                "【识别新旧差异】：旧题为连续整数（差为 1，和为 72）；"
                "新问题变更为“连续奇数”（相邻奇数公差为 2），且三数之和变更为 81。\n"
                "【修改旧方案并验证】：\n"
                "1. 设三个连续奇数分别为 n - 2, n, n + 2。\n"
                "2. 建立方程：(n - 2) + n + (n + 2) = 3n = 81。\n"
                "3. 解得中间奇数 n = 81 / 3 = 27。\n"
                "4. 最大的奇数为 n + 2 = 27 + 2 = 29。\n"
                "验算：三数为 25, 27, 29，均为连续奇数，25 + 27 + 29 = 81，完全符合题意。\n"
                "Final Answer: 29"
            )

        # 场景 D: 无关题目处理 (欧几里得素数、二叉树 LCA 等)
        else:
            return (
                "内部思考：检查检索结果，返回的案例为数学应用题，与当前题目完全无语义和领域关联（相关度极低）。\n"
                "声明：未检索到相关历史案例，废弃无关检索结果，启动独立逻辑推理。\n"
                "【独立推导】：基于数学/算法第一性原理逐步推演分析...\n"
                "已完成严密论证与步骤推导。\n"
                "Final Answer: 证明完成"
            )


class ExperienceExpert:
    """Expert Agent capable of deciding whether to invoke memory and adapting solutions."""

    def __init__(self, llm_client=None, vector_store=None):
        """Initialize the expert agent.

        Args:
            llm_client: Client for LLM generation. Defaults to MockLLMClient if None.
            vector_store: SimpleVectorStore instance containing indexed experiences.
        """
        self.vector_store = vector_store
        self.llm_client = llm_client or MockLLMClient(vector_store=vector_store)

    def _execute_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Execute a tool call and return formatted string output.

        Routes 'search_memory' to vector_store.retrieve.

        Args:
            tool_name: Name of tool to execute.
            kwargs: Keyword arguments for tool execution.

        Returns:
            Formatted result string.
        """
        if tool_name == "search_memory":
            if self.vector_store is None:
                return "Error: Vector store is not configured on this agent."

            query = kwargs.get("query") or kwargs.get("query_text", "")
            top_k = kwargs.get("top_k", 1)
            results = self.vector_store.retrieve(query_text=query, top_k=top_k)

            if not results:
                return "No relevant historical records found in memory."

            formatted_blocks = []
            for idx, item in enumerate(results, start=1):
                sim = item.get("similarity", 0.0)
                formatted_blocks.append(
                    f"--- Memory Record {idx} (Similarity: {sim:.4f}) ---\n"
                    f"memory_id: {item.get('memory_id')}\n"
                    f"source_problem_id: {item.get('source_problem_id')}\n"
                    f"problem_text: {item.get('problem_text')}\n"
                    f"solution_text: {item.get('solution_text')}"
                )
            return "\n\n".join(formatted_blocks)

        return f"Error: Unknown tool '{tool_name}'."

    def solve(self, new_problem: str) -> Dict[str, Any]:
        """Process a problem through multi-turn interaction with tool call handling.

        Args:
            new_problem: Input problem text.

        Returns:
            Dict containing problem, trajectory, is_memory_called, and final_answer.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
            {"role": "user", "content": new_problem},
        ]

        trajectory_parts: List[str] = [f"=== Problem ===\n{new_problem}\n"]
        is_memory_called = False
        final_answer = ""
        max_turns = 5

        for _ in range(max_turns):
            response = self.llm_client.generate(
                messages=messages,
                tools=[SEARCH_MEMORY_TOOL_DEFINITION],
            )

            if response.content:
                trajectory_parts.append(f"=== Assistant Thought ===\n{response.content}\n")

            if response.has_tool_calls:
                # 处理工具调用
                for call in response.tool_calls:
                    if call.name == "search_memory":
                        is_memory_called = True

                    trajectory_parts.append(
                        f"=== Tool Call ===\nAction: {call.name}\nArguments: {json.dumps(call.arguments, ensure_ascii=False)}\n"
                    )

                    tool_output = self._execute_tool(call.name, call.arguments)
                    trajectory_parts.append(f"=== Tool Output ===\n{tool_output}\n")

                    # 将交互历史推入对话上下文
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [call.to_dict()],
                    })
                    messages.append({
                        "role": "tool",
                        "name": call.name,
                        "content": tool_output,
                    })
            else:
                # 没有新的工具调用，本轮生成完整答复，尝试提取最终答案
                final_answer = self._extract_final_answer(response.content)
                break

        full_trajectory = "\n".join(trajectory_parts)

        return {
            "problem": new_problem,
            "trajectory": full_trajectory,
            "is_memory_called": is_memory_called,
            "final_answer": final_answer,
        }

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Extract the final answer string from model output."""
        match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
