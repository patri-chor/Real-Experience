# 任务 02：经验调用专家与复用修改模块

## 任务概述
你是一个执行子智能体。请在工作区 `p1/src/expert/` 下实现带有工具调用（Tool Calling）能力的专家 Agent。该 Agent 必须能根据当前问题决定是否调用外部记忆工具，并在得到历史案例后进行逻辑复用与修改。最后，请在 `task/p1/` 目录下生成一份名为 `report_02.md` 的阶段总结报告。

## 工作区与输出路径
- 代码根目录：`p1/`
- 需要创建的文件：
  1. `p1/src/expert/prompts.py`
  2. `p1/src/expert/agent.py`
  3. `p1/src/expert/evaluator.py`
  4. `task/p1/report_02.md`

## 详细要求

### 1. `p1/src/expert/prompts.py`
定义 `EXPERT_SYSTEM_PROMPT` 字符串，该 Prompt 需包含以下行为准则：
1. 分析问题，若足够简单（如常识题、一步计算），直接作答。
2. 若问题较复杂，可调用 `search_memory` 工具查找历史案例。
3. 若找到案例，强制要求在内部思考（CoT）中体现：“分析旧题 -> 提取可复用逻辑 -> 识别新旧差异 -> 修改旧方案”的过程。
4. 若未找到或无关，必须独立推理。

### 2. `p1/src/expert/agent.py`
实现 `ExperienceExpert` 类：
- `__init__(self, llm_client, vector_store)`: 初始化，传入 LLM 客户端和上一阶段的 `SimpleVectorStore` 实例。对于 `llm_client` 可以使用 `google.generativeai` 或者你擅长的轻量级实现（如使用 requests 访问兼容的 API，这里可以使用 gemini 或 mock 模型）。
- `_execute_tool(self, tool_name: str, kwargs: dict) -> str`: 执行工具调用，路由至 `vector_store.retrieve`。
- `solve(self, new_problem: str) -> dict`: 接收新问题，处理与 LLM 的多轮对话（包含 Tool Call 和 Response），返回字典格式结果：
  `{"problem": str, "trajectory": str, "is_memory_called": bool, "final_answer": str}`。

### 3. `p1/src/expert/evaluator.py`
实现以下评估函数：
- `evaluate_invocation(is_memory_called: bool, expected_action: str) -> bool`: 检查调用时机是否正确（例如，期望 "similar" 必须调，"simple" 必须不调）。
- `evaluate_reuse_and_modify(trajectory: str, old_problem_id: str) -> bool`: 解析模型输出的 trajectory，检查是否包含了旧题逻辑和差异修改。
- `verify_correctness(predicted_answer: str, ground_truth: str) -> bool`: 答案准确性检查。

### 4. 阶段报告
在 `task/p1/report_02.md` 中，总结你实现这三个文件的思路、遇到的挑战以及提供的接口功能。

## 提示
不需要完美连接真实的 LLM API，若没有真实环境，`agent.py` 中的 LLM 交互可以是伪代码或者基于预定义行为的模拟器，只要接口、Prompt 结构以及整体闭环逻辑严密即可。写完后请通知架构师。
