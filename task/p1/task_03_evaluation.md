# 任务 03：效果验证与轨迹收集模块

## 任务概述
你是一个执行子智能体。本任务需要在工作区中将前两阶段完成的记忆模块（`memory`）与专家模型模块（`expert`）串联起来。编写独立运行的脚本来真实执行测试集，评估专家的行为表现，并实现一个轨迹收集器（`trajectory_collector.py`）将符合标准的高质量 CoT（思维链）轨迹导出为 JSONL 格式，为下一阶段的模型蒸馏（SFT）准备数据。最后，请在 `task/p1/` 下生成一份名为 `report_03.md` 的阶段总结报告。

## 工作区与输出路径
- 代码根目录：`p1/`
- 需要创建的文件：
  1. `p1/scripts/run_02_expert_eval.py`
  2. `p1/src/distill/trajectory_collector.py`
  3. `p1/scripts/run_03_collect_trajectories.py`
  4. `task/p1/report_03.md`

## 详细要求

### 1. `p1/scripts/run_02_expert_eval.py`
该脚本是阶段二效果验证的入口：
1. 实例化 `SimpleVectorStore` 并加载 `build_seed_memory()` 数据。
2. 实例化 `ExperienceExpert`。
3. 从 `build_test_cases()` 加载 `simple`、`similar`、`unrelated` 三类测试题。
4. 依次遍历每道题，调用专家进行解答。
5. 针对每题的输出，调用 `evaluate_invocation` 验证其调用时机，以及 `verify_correctness` 验证其答案。若为 `similar` 题，还需调用 `evaluate_reuse_and_modify`。
6. 在终端打印清晰的统计报告，输出各分类下的正确率和行为合规率。

### 2. `p1/src/distill/trajectory_collector.py`
实现轨迹收集功能，为大模型向小模型蒸馏做数据准备：
- `def run_expert_and_collect(expert, test_cases_dict: dict, output_jsonl: str) -> None`:
  接收专家对象和完整的字典测试集，执行推理并过滤。**过滤规则**：只有当答案准确 (`verify_correctness` 为真) 且行为符合预期（如该调的调了，该复用的复用了）的轨迹才被收集。
  导出格式为 JSONL，每行类似于：
  ```json
  {
      "problem": "...",
      "category": "similar",
      "trajectory": "...",
      "final_answer": "..."
  }
  ```

### 3. `p1/scripts/run_03_collect_trajectories.py`
该脚本是收集轨迹的入口：
- 调用 `run_expert_and_collect`，并将数据保存到 `p1/data/distill_cot.jsonl`（如果 `p1/data` 目录不存在则自动创建）。

### 4. 阶段报告
在 `task/p1/report_03.md` 中，总结上述脚本的串联逻辑、在控制台输出的评估结果，以及生成的 JSONL 数据的样本格式说明。

## 提示
请写出完整的、可直接运行的 Python 脚本（确保导入路径正确，如 `from p1.src.memory.data_builder import ...`）。完成后通知架构师。
