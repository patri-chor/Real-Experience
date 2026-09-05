# 任务 04：模型蒸馏与内化测试模块

## 任务概述
你是一个执行子智能体。本任务要求在工作区 `p1/` 目录下完成实验的最后阶段：将前一阶段收集的高质量带有经验调用逻辑的轨迹（`distill_cot.jsonl`），通过有监督微调（SFT）“蒸馏”到一个较小尺寸的目标模型中。然后去除所有系统级别的记忆工具和长提示词，在纯净环境下测试小模型是否能够“内化”之前学到的回忆和复用行为。最后，请在 `task/p1/` 下生成 `report_04.md`。

## 工作区与输出路径
- 代码根目录：`p1/`
- 需要创建的文件：
  1. `p1/src/distill/sft_trainer.py`
  2. `p1/src/distill/student_tester.py`
  3. `p1/scripts/run_04_train_and_test_student.py`
  4. `task/p1/report_04.md`

## 详细要求

### 1. `p1/src/distill/sft_trainer.py`
实现微调小模型的逻辑。
- `def train_student_model(dataset_path: str, output_dir: str) -> None`:
  鉴于我们是在轻量级原型下测试，不需要真实调用 GPU 或 HuggingFace Trainer，请实现一个 **Mock 版本的 SFT Trainer**。它需要：
  - 读取 `dataset_path` (即 `distill_cot.jsonl`)，解析并打印出“开始训练模型... 数据量 X”。
  - 模拟训练过程（例如 `time.sleep(2)`）。
  - 在 `output_dir` 中写入一个 `mock_model.json`，把训练集内化为其“权重”。

### 2. `p1/src/distill/student_tester.py`
实现对微调后小模型的内化能力测试。
- `class InternalizedStudentModel:`
  - `__init__(self, model_dir: str)`: 加载 `mock_model.json`。
  - `generate(self, prompt: str) -> str`: 输入只有问题，没有任何工具接口，模拟小模型的自发回答。若问题与训练集中的 `similar` 问题匹配，则模拟输出类似“我回想起了一道相似的历史题...提取逻辑...修改参数...”的思维链和答案。若问题是简单的，直接输出答案。
- `def analyze_internalized_cot(trajectory: str) -> dict`:
  利用正则或关键词分析小模型在纯净输入下是否自发生成了类似 `分析旧题`、`提取逻辑`、`修改方案` 等核心步骤。

### 3. `p1/scripts/run_04_train_and_test_student.py`
作为本阶段的总控脚本：
1. 调用 `train_student_model` 对上阶段产出的 `distill_cot.jsonl` 模拟训练，并保存至 `p1/models/student_v1`。
2. 实例化 `InternalizedStudentModel`。
3. 从 `build_test_cases()` 加载测试集，**不加**任何 `search_memory` 工具，直接让 Student 解答。
4. 运行 `analyze_internalized_cot` 校验轨迹。
5. 打印对比实验结果：“是否有外部经验提示？” vs “内化自发回忆效果”。

### 4. 阶段报告
在 `task/p1/report_04.md` 中，总结 SFT 蒸馏的范式设计，以及小模型在被去掉所有拐杖（提示词和检索工具）后，是否依然能表现出“联想-复用-修改”的人类高级认知行为。

## 提示
请编写具有完备注释和打印输出的代码。完成后通知架构师。
