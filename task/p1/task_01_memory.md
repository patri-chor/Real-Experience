# 任务 01：构建简易索引记忆模块

## 任务概述
你是一个执行子智能体。请在工作区 `p1/src/memory/` 下实现极简的索引记忆模块，并生成初始测试数据。无需引入复杂的依赖（如 ChromaDB/Milvus），使用基础的 numpy 余弦相似度或轻量级库即可。

## 工作区与输出路径
- 代码根目录：`p1/` (注意不是 `task/p1/`)
- 需要创建的文件：
  1. `p1/src/memory/vector_store.py`
  2. `p1/src/memory/data_builder.py`

## 详细要求

### 1. `p1/src/memory/vector_store.py`
实现 `SimpleVectorStore` 类，需包含以下方法：
- `__init__(self)`: 初始化。可使用 `sentence-transformers`（例如 `"all-MiniLM-L6-v2"`）进行文本向量化。如果没有相关依赖，可暂时用随机向量或 mock 函数占位，但接口必须完整。
- `add_record(self, memory_id: str, source_problem_id: str, problem_text: str, solution_text: str)`: 将题目文本向量化并保存。
- `retrieve(self, query_text: str, top_k: int = 1) -> list[dict]`: 接收查询题目，计算余弦相似度，返回最相似的 Top-K 个记录（返回完整的字典格式，包含解题步骤）。

### 2. `p1/src/memory/data_builder.py`
实现以下两个数据构造函数：
- `build_seed_memory() -> list[dict]`: 
  手动硬编码返回 3-5 道历史数学题或逻辑题作为“记忆种子”。包含字段：`memory_id`, `source_problem_id`, `problem_text`, `solution_text`。
- `build_test_cases() -> dict[str, list[dict]]`: 
  构造测试集，返回一个字典，包含三类题目（每类至少 1-2 题）：
  - `"simple"`: 比如 1+1=2，或者无需思考的常识题。
  - `"similar"`: 与种子记忆库中同构但参数不同的题目。
  - `"unrelated"`: 难度大但记忆库中完全没有的复杂题目。

## 验收标准 (验收官使用)
1. 类名与方法签名必须与要求完全一致。
2. 可以在根目录下独立写一个短小的 `test_memory.py` 测试脚本，证明 `retrieve` 函数能正确把 `"similar"` 问题匹配到对应的种子记忆，而不会给 `"unrelated"` 问题匹配高分。

## 提示
请保持代码轻量。完成代码后，请告知验收官（架构师），准备接受验收。
