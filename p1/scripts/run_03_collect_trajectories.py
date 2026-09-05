"""Script to run expert on test cases and collect qualified trajectories for SFT distillation."""

import sys
from pathlib import Path

# Ensure workspace and p1 root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
P1_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = P1_ROOT.parent

for path in [str(WORKSPACE_ROOT), str(P1_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from p1.src.memory.vector_store import SimpleVectorStore
from p1.src.memory.data_builder import build_seed_memory, build_test_cases
from p1.src.expert.agent import ExperienceExpert
from p1.src.distill.trajectory_collector import run_expert_and_collect


def main() -> None:
    """Execute trajectory collection pipeline and persist output."""
    print("=" * 80)
    print("                启动专家模型高质量思维链轨迹收集流水线")
    print("=" * 80)

    # 1. 实例化向量库并注入种子记忆
    print("[1/3] 初始化向量存储库并装载种子记忆库...")
    vector_store = SimpleVectorStore()
    seeds = build_seed_memory()
    for s in seeds:
        vector_store.add_record(
            memory_id=s["memory_id"],
            source_problem_id=s["source_problem_id"],
            problem_text=s["problem_text"],
            solution_text=s["solution_text"],
        )
    print(f"      成功装载 {len(seeds)} 条种子记忆。")

    # 2. 实例化经验专家与测试集
    print("[2/3] 实例化 ExperienceExpert 并加载测试用例...")
    expert = ExperienceExpert(vector_store=vector_store)
    test_cases_dict = build_test_cases()
    total_cases = sum(len(cases) for cases in test_cases_dict.values())
    print(f"      成功加载 {total_cases} 道测试题 (涵盖 simple/similar/unrelated)。")

    # 3. 确定导出路径并执行收集
    output_path = P1_ROOT / "data" / "distill_cot.jsonl"
    print(f"[3/3] 执行推理、双重质量过滤与 JSONL 导出...")
    print(f"      目标文件: {output_path}")

    run_expert_and_collect(
        expert=expert,
        test_cases_dict=test_cases_dict,
        output_jsonl=str(output_path),
    )

    # 4. 验证导出结果
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print("=" * 80)
        print(f"流水线执行完成！生成文件大小: {output_path.stat().st_size} 字节，包含样本: {len(lines)} 行。")
        print("=" * 80)
    else:
        print(f"错误: 目标文件 {output_path} 未成功生成！", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
