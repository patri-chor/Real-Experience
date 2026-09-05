# 阶段技术报告：任务 02 经验调用专家与复用修改模块

## 1. 任务背景与目标

根据实验计划 P1 与任务书 `task/p1/task_02_expert.md` 的要求，本阶段的核心目标是构建具备条件记忆访问能力的经验调用专家 Agent（`ExperienceExpert`）。该 Agent 面临新问题时，需具备区分题目复杂度的能力：
1. 简单常识或一步计算题直接作答，不触发外部记忆检索；
2. 复杂或同构题型主动触发 `search_memory` 工具，并在获取历史案例后，于思维链（CoT）中严密执行四步认知迁移：“分析旧题 $\rightarrow$ 提取可复用逻辑 $\rightarrow$ 识别新旧差异 $\rightarrow$ 修改旧方案并验证”；
3. 检索无结果或低相关度问题则主动退回独立第一性原理推导，避免负迁移与工具幻觉。

---

## 2. 模块架构与多轮状态机

```
               +----------------------+
               |    用户输入新问题    |
               +----------+-----------+
                          |
                          v
               +----------------------+
               |  复杂度分析与决策    |
               +----+------------+----+
                    |            |
       [简单题目]   |            | [复杂/同构题目]
                    v            v
           +-------------+  +-------------------------------+
           |  直接作答   |  | 触发 search_memory 工具调用    |
           +------+------+  +---------------+---------------+
                  |                         |
                  |                         v
                  |         +-------------------------------+
                  |         | SimpleVectorStore.retrieve    |
                  |         +---------------+---------------+
                  |                         |
                  |                         v
                  |         +-------------------------------+
                  |         | 4-Phase CoT 认知复用与修改    |
                  |         | 1. 分析旧题                   |
                  |         | 2. 提取可复用逻辑             |
                  |         | 3. 识别新旧差异               |
                  |         | 4. 修改旧方案并验证           |
                  |         +---------------+---------------+
                  |                         |
                  v                         v
               +----------------------+
               |   格式化输出最终答案 |
               +----------------------+
```

### 交互数据结构
`ExperienceExpert.solve(new_problem)` 统一返回结构化字典：
```python
{
    "problem": str,             # 原始输入题目
    "trajectory": str,          # 包含思维链、工具调用参数与返回结果的完整轨迹文本
    "is_memory_called": bool,   # 本轮是否触发了记忆检索工具
    "final_answer": str         # 提取出的最终答案结论
}
```

---

## 3. 各模块接口与核心实现细节

### 3.1 提示词设计 (`p1/src/expert/prompts.py`)
- `EXPERT_SYSTEM_PROMPT`：严格定义了五大行为准则，从规则层面约束模型必须显式打印四步迁移阶段标签（`[分析旧题]`、`[提取可复用逻辑]`、`[识别新旧差异]`、`[修改旧方案并验证]`），杜绝黑盒抄袭。
- `SEARCH_MEMORY_TOOL_DEFINITION`：提供标准 OpenAI / Gemini 兼容的 Function Calling JSON Schema，定义了 `query` 和 `top_k` 参数规范。

### 3.2 专家智能体与状态机 (`p1/src/expert/agent.py`)
- `ExperienceExpert`：
  - `__init__(self, llm_client=None, vector_store=None)`：支持外部依赖注入。默认自动装配内置轻量仿真器 `MockLLMClient`，实现零外部依赖的完全自测。
  - `_execute_tool(self, tool_name: str, kwargs: dict) -> str`：工具调度器。负责解析参数，执行向量库相似度检索，并将向量库的结构化列表格式化为清晰的纯文本案例卡片供 LLM 上下文消费。
  - `solve(self, new_problem: str) -> dict`：多轮对话状态机。维护 `messages` 列表，支持多轮交互，设置 `max_turns = 5` 防止死循环，自动抓取 `Final Answer:` 字段。
- `MockLLMClient`：
  - 针对 `data_builder.py` 构建的三类典型场景（`simple`、`similar`、`unrelated`）进行了精准的多轮模拟。
  - 第一轮输出思考与 ToolCall；第二轮接收 tool output，输出规范的四步 CoT 思考链与计算推导。

### 3.3 评估套件 (`p1/src/expert/evaluator.py`)
- `evaluate_invocation(is_memory_called: bool, expected_action: str) -> bool`：
  - 检查工具调用时机的精准度。`similar` 强制为 `True`，`simple` 强制为 `False`，避免过度检索与检索缺失。
- `evaluate_reuse_and_modify(trajectory: str, old_problem_id: str) -> bool`：
  - 通过正则与多阶段关键词组合，严格验证轨迹中是否显式具备：来源 ID 引用、旧题机制分析、逻辑骨架提取、差异参数识别、新解法调整与验算这五个关键要素。
- `verify_correctness(predicted_answer: str, ground_truth: str) -> bool`：
  - 实现了兼顾文本与数值的鲁棒比对。支持分数提取（如 `10/3`）、货币符号剥离（`$`）、物理单位过滤（`km/h`、`hours`）以及基于浮点容差（$10^{-3}$）的等价性校验。

---

## 4. 自测与验收结果

基于 `p1/tests/test_expert.py` 执行完整的单元测试套件：
1. **Prompt 规范性测试 (`test_prompt_requirements`)**：通过。所有强制思维链标签与工具定义均符合要求。
2. **工具调用通路测试 (`test_tool_execution`)**：通过。能够准确从上一阶段的 `SimpleVectorStore` 检索出 `mem_001` 等记录并完成文本格式化。
3. **简单题直接作答验证 (`test_solve_simple_cases`)**：通过。`is_memory_called` 恒为 `False`，准确率 100%。
4. **相似题经验复用验证 (`test_solve_similar_cases`)**：通过。所有相似题均触发工具调用，正确匹配目标种子，四步 CoT 轨迹评估全部判定为 `True`，最终数值计算完全正确。
5. **无关题防幻觉验证 (`test_solve_unrelated_cases`)**：通过。模型显式声明未找到相关案例，不强制迁移无关逻辑。
6. **评估器边界与反例测试 (`test_evaluator_positive_and_negative`)**：通过。对未做修改的死板抄袭轨迹能够准确判定失败。

---

## 5. 遇到的挑战与设计权衡

1. **避免简单题目的过度检索**：
   - 挑战：如果仅提示“复杂时可检索”，大模型容易对所有数学题进行防御性检索。
   - 权衡：在 Prompt 中将简单题目直接作答列为首要准则，并在评估器中设置 `evaluate_invocation(..., "simple")` 反向硬约束。
2. **多轮状态机死锁与终止保证**：
   - 挑战：如果模型反复发起相同的工具调用，Agent 可能会陷入死循环。
   - 权衡：设置 `max_turns = 5` 硬截断；同时状态机检测到没有新的 tool_calls 时即提取答案收敛退出。
3. **经验复用判定的程序化定义**：
   - 挑战：单纯检查是否包含旧题 ID 无法证明模型真正“修改”了方案；如果过于严格使用 NLP 相似度，又可能误杀合理变式。
   - 权衡：将“经验复用”严格解构为五个正交维度（旧题关联、旧题分析、逻辑提取、差异识别、方案调整），每个维度独立检查特征标记，确保评估具备确定性与可解释性。
