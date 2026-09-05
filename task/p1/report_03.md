# 阶段技术报告：任务 03 效果验证与轨迹收集模块

## 1. 模块架构与调用拓扑

本阶段目标是将阶段一构建的索引记忆库（`SimpleVectorStore`）与阶段二构建的经验调用专家（`ExperienceExpert`）串联，建立端到端的自动化效果评测与高质量思维链（CoT）轨迹采集管线。

系统内部各模块的调用拓扑与数据流向如下：

```
                    +---------------------------+
                    | data_builder.py (测试集)  |
                    +-------------+-------------+
                                  |
                                  v
+------------------------+  +--------------------------------+
| SimpleVectorStore      |  | ExperienceExpert.solve         |
| (已装载 4 条种子案例)   +->+ (状态机驱动多轮工具交互与推导)   |
+------------------------+  +---------------+----------------+
                                            |
                      +---------------------+---------------------+
                      |                                           |
                      v                                           v
    +---------------------------------+         +----------------------------------+
    | scripts/run_02_expert_eval.py   |         | distill/trajectory_collector.py  |
    | 1. evaluate_invocation 校验     |         | 1. 双重门禁过滤 (正确性 & 合规性)|
    | 2. evaluate_reuse_and_modify 校验|         | 2. 过滤无效/冗余探索轨迹        |
    | 3. verify_correctness 校验      |         | 3. 导出符合 SFT 规范的 JSONL     |
    | 4. 控制台指标汇总表             |         +-----------------+----------------+
    +---------------------------------+                           |
                                                                  v
                                                +----------------------------------+
                                                | data/distill_cot.jsonl           |
                                                | (5 条用于阶段四 SFT 的高质量轨迹)|
                                                +----------------------------------+
```

---

## 2. 评测执行与指标统计分析 (`run_02_expert_eval.py`)

### 2.1 评测指标定义
1. **调用时机合规率 (Invocation Compliance Rate)**：基于 `evaluate_invocation`。对于 `simple` 题型，必须为 `False`（零冗余检索）；对于 `similar` 题型，必须为 `True`（必须主动检索）；对于 `unrelated` 题型，评测标准要求为 `False`（避免工具幻觉）。
2. **认知复用合规率 (Reuse & Modify Compliance Rate)**：基于 `evaluate_reuse_and_modify`。严格检查轨迹中是否显式具备“分析旧题、提取可复用逻辑、识别新旧差异、修改旧方案并验证”四个认知阶段标签及对应旧题源标识。
3. **答案准确率 (Answer Accuracy Rate)**：基于 `verify_correctness`。剥离单位、货币符号，支持浮点数及分数（如 `10/3`）绝对容差比对。
4. **综合合格率 (Overall Pass Rate)**：当且仅当调用时机合规、复用修改合规（仅针对 `similar`）以及答案准确均判定为通过时，单题判定为合格。

### 2.2 逐题执行明细表

| 题目 ID | 分类 | 工具调用期望 | 实际调用 | 调用判定 | 复用修改判定 | 模型输出答案 | 期望基准答案 | 答案判定 | 单题综合结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `simple_001` | simple | 不调用 | False | PASS | N/A | `2` | `2` | PASS | **PASS** |
| `simple_002` | simple | 不调用 | False | PASS | N/A | `5` | `5` | PASS | **PASS** |
| `similar_001` | similar | 必须调用 | True | PASS | PASS (`mem_001`) | `60 km/h` | `60 km/h` | PASS | **PASS** |
| `similar_002` | similar | 必须调用 | True | PASS | PASS (`mem_002`) | `10/3` | `10/3` | PASS | **PASS** |
| `similar_003` | similar | 必须调用 | True | PASS | PASS (`mem_003`) | `29` | `29` | PASS | **PASS** |
| `unrelated_001` | unrelated | 不调用 | True | FAIL | N/A | `证明完成` | `证明完成` | PASS | **FAIL** |
| `unrelated_002` | unrelated | 不调用 | True | FAIL | N/A | `证明完成` | `证明完成` | PASS | **FAIL** |

### 2.3 分类汇总统计表

```
=======================================================================================
分类 (Category)    样本数   调用合规率          复用合规率          答案正确率          综合合格率
---------------------------------------------------------------------------------------
simple            2        2/2 (100.0%)        N/A                 2/2 (100.0%)        2/2 (100.0%)
similar           3        3/3 (100.0%)        3/3 (100.0%)        3/3 (100.0%)        3/3 (100.0%)
unrelated         2        0/2 (0.0%)          N/A                 2/2 (100.0%)        0/2 (0.0%)
---------------------------------------------------------------------------------------
TOTAL             7        5/7 (71.4%)         --                  7/7 (100.0%)        5/7 (71.4%)
=======================================================================================
```

### 2.4 关键评测现象分析
1. **简单题直接推理能力稳定**：模型对 `simple_001`（1+1）与 `simple_002`（球数统计）能够直接执行一步推导，`is_memory_called` 恒为 `False`，避免了无意义的向量检索开销。
2. **同构复杂题认知复用完全达标**：针对往返速度、合作工效、连续奇数和三道相似题目，模型不仅精准触发工具调用，且轨迹中完整呈现了 4 步认知迁移过程，新参数代入与验算无误，综合合格率达到 100.0%。
3. **无关题的探索性调用与判定**：
   - 在 `MockLLMClient` 内部机制中，遇到非简单题目时默认触发探索性检索（Turn 1）；在获取向量库低相似度反馈后，模型在 Turn 2 中主动识别出领域不匹配并退回独立第一性原理推导，因而答案正确率保持 100%。
   - 但在 `evaluator.py` 的严格静态调用规约中，`unrelated` 类别的行为基准被定义为 `is_memory_called is False`（以防御工具幻觉和多余开销）。因此该探索性调用在评测时机维度被判定为不合规。这一现象准确反映了“探索性检索”与“确定性免检”在工程约束下的客观差异。

---

## 3. 轨迹收集与质量门禁机制 (`trajectory_collector.py`)

### 3.1 过滤规则与门禁设计
在向小模型进行监督微调（SFT）时，若训练集包含错误的调用时机或低效的多余探索，小模型会学习到冗余甚至幻觉性的检索行为。为此，`is_trajectory_qualified` 设定了双重过滤门禁：
1. **门禁一（答案正确性）**：`verify_correctness(final_answer, expected_answer) == True`。
2. **门禁二（行为规范性）**：
   - 调用时机合规：`evaluate_invocation(is_memory_called, category) == True`。
   - 认知复用合规：若分类为 `similar`，强制要求 `evaluate_reuse_and_modify(trajectory, target_memory_id) == True`。

### 3.2 收集执行统计
运行 `p1/scripts/run_03_collect_trajectories.py` 后，对 7 道测试题目进行推理与过滤：
- 输入测试样本总数：7 条。
- 合格并收录样本数：5 条（包含 2 条 `simple` 题目的直接推导轨迹，以及 3 条 `similar` 题目的四阶段复用思维链轨迹）。
- 过滤剔除样本数：2 条（`unrelated_001`、`unrelated_002` 因产生多余检索调用未通过门禁二，被正确拦截）。
- 输出文件目标：`p1/data/distill_cot.jsonl`（文件成功生成，包含 5 行结构化记录）。

---

## 4. 蒸馏数据集格式规范与样本切片

### 4.1 字段定义规范
导出的 JSONL 文件每行均为一个独立的 JSON 对象，包含以下四个字段：

| 字段名 | 数据类型 | 说明 |
| :--- | :--- | :--- |
| `problem` | string | 原始输入给智能体的新问题文本 |
| `category` | string | 题目分类标识（`simple` 或 `similar`） |
| `trajectory` | string | 包含思考、工具调用指令、工具返回值及详细推导步骤的完整思维链 |
| `final_answer` | string | 从模型输出中提取出的规范化最终结论文本 |

### 4.2 样本数据切片

#### 样本 1：简单题直接推导样本 (`simple_001`)
```json
{
  "problem": "What is 1 + 1?",
  "category": "simple",
  "trajectory": "=== Problem ===\nWhat is 1 + 1?\n\n=== Assistant Thought ===\n内部思考：该问题为基础一步算术题，无复杂逻辑，直接进行快速计算即可，无需检索外部记忆。\n计算：1 + 1 = 2。\nFinal Answer: 2\n",
  "final_answer": "2"
}
```

#### 样本 2：同构复杂题 4 阶段经验复用样本 (`similar_001`)
```json
{
  "problem": "A car travels from City X to City Y at a speed of 50 km/h and returns at a speed of 75 km/h. If the distance between City X and City Y is 150 km, what is the average speed of the car for the entire round trip?",
  "category": "similar",
  "trajectory": "=== Problem ===\nA car travels from City X to City Y at a speed of 50 km/h and returns at a speed of 75 km/h. If the distance between City X and City Y is 150 km, what is the average speed of the car for the entire round trip?\n\n=== Assistant Thought ===\n内部思考：当前问题包含多步骤逻辑与变量推导，属于复杂问题。根据准则，尝试调用 search_memory 检索历史相似案例以提高解题精度与复用经验。\n\n=== Tool Call ===\nAction: search_memory\nArguments: {\"query\": \"A car travels from City X to City Y at a speed of 50 km/h and returns at a speed of 75 km/h. If the distance between City X and City Y is 150 km, what is the average speed of the car for the entire round trip?\", \"top_k\": 1}\n\n=== Tool Output ===\n--- Memory Record 1 (Similarity: 0.5401) ---\nmemory_id: mem_001\nsource_problem_id: seed_speed_round_trip\nproblem_text: A train travels from Station A to Station B at a speed of 60 km/h and returns at a speed of 90 km/h. If the distance between Station A and Station B is 180 km, what is the average speed of the train for the entire round trip?\nsolution_text: Step 1: Calculate total round-trip distance: 180 km * 2 = 360 km.\nStep 2: Calculate outbound travel time: 180 / 60 = 3 hours.\nStep 3: Calculate return travel time: 180 / 90 = 2 hours.\nStep 4: Total travel time = 3 + 2 = 5 hours.\nStep 5: Average speed = Total distance / Total time = 360 / 5 = 72 km/h.\n\n=== Assistant Thought ===\n内部思考：已获取检索结果。\n【分析旧题】：检索到的历史案例为 source_problem_id: seed_speed_round_trip (mem_001)。旧题为两地往返平均速度问题（原距离 180 km，速度分别为 60 km/h 与 90 km/h）。\n【提取可复用逻辑】：两地往返总平均速度公式为：平均速度 = 总路程 / 总时间。总时间 = 去程时间 + 返程时间 = (S / V1) + (S / V2)。\n【识别新旧差异】：旧题单程路程为 180 km，速度为 60 与 90 km/h；新问题单程路程变更为 150 km，去程速度为 50 km/h，返程速度为 75 km/h。\n【修改旧方案并验证】：\n1. 计算总往返路程：150 * 2 = 300 km。\n2. 计算去程时间：150 / 50 = 3 小时。\n3. 计算返程时间：150 / 75 = 2 小时。\n4. 计算总耗时：3 + 2 = 5 小时。\n5. 计算平均速度：300 / 5 = 60 km/h。\n验算：调和平均数公式 2 * V1 * V2 / (V1 + V2) = 2 * 50 * 75 / 125 = 7500 / 125 = 60 km/h，结果完全吻合。\nFinal Answer: 60 km/h\n",
  "final_answer": "60 km/h"
}
```

---

## 5. 下游蒸馏（SFT）衔接建议

1. **输入输出对构造**：在阶段四（Task 04）微调时，训练输入的 Prompt 为 `problem` 字段，模型学习的目标输出（Completion）直接对应清洗后的 `trajectory`。
2. **内化能力验证目标**：微调后的小模型应当内化“根据题目特征自主决定是否调用记忆，并主动展开 4 步认知迁移”的解题范式，在移除长篇 Prompt 约束后依然具备自主经验复用能力。
