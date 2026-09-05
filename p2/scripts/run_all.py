"""Master execution pipeline script for p2."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Path definitions
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
P2_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = P2_ROOT.parent

# Define the sequential stages
PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "id": "data",
        "title": "数据合成与构建 (Data Generation)",
        "script_rel": Path("p2") / "src" / "data" / "generate_dataset.py",
        "desc": "生成包含同构记忆题与基础题的 post-training 训练集 (post_train.jsonl)",
    },
    {
        "id": "sft",
        "title": "SFT 冷启动微调 (SFT QLoRA)",
        "script_rel": Path("p2") / "src" / "train" / "sft_lora.py",
        "desc": "基于 4-bit Qwen2.5-0.5B-Instruct 进行格式化监督微调，产出 sft_lora_0.5B",
    },
    {
        "id": "grpo",
        "title": "GRPO 强化学习对齐 (GRPO RL)",
        "script_rel": Path("p2") / "src" / "train" / "grpo_rl.py",
        "desc": "以格式约束与解题准确度为奖励函数，强化长思维链与记忆调用能力",
    },
    {
        "id": "eval",
        "title": "模型推理与能力评测 (HF Tester)",
        "script_rel": Path("p2") / "src" / "eval" / "hf_tester.py",
        "desc": "加载 4-bit 模型与 LoRA 适配器，测试同构迁移与无关题的思维链生成",
    },
]


def print_banner(text: str) -> None:
    print("\n" + "=" * 80)
    print(f"   {text}")
    print("=" * 80)


def execute_step(step_idx: int, total_steps: int, step_info: Dict[str, Any], dry_run: bool = False) -> bool:
    """Execute a single pipeline step via subprocess.run."""
    script_path = WORKSPACE_ROOT / step_info["script_rel"]

    print_banner(f"[{step_idx}/{total_steps}] 正在执行: {step_info['title']}")
    print(f"描述: {step_info['desc']}")
    print(f"脚本路径: {script_path}")

    if not script_path.exists():
        print(f"[错误] 目标脚本不存在: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    cmd_display = f"python {step_info['script_rel'].as_posix()}"
    print(f"执行命令: {cmd_display}\n")

    if dry_run:
        print("[DRY-RUN] 空跑模式，跳过实际执行。")
        return True

    start_time = time.perf_counter()
    try:
        # 在工作区根目录下执行命令
        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            check=True,
        )
        elapsed = time.perf_counter() - start_time
        print(f"\n[成功] 阶段 {step_info['id']} 完成，耗时: {elapsed:.2f} 秒。")
        return True
    except subprocess.CalledProcessError as err:
        elapsed = time.perf_counter() - start_time
        print(f"\n[失败] 阶段 {step_info['id']} 执行异常 (返回码: {err.returncode})，耗时: {elapsed:.2f} 秒。")
        return False
    except KeyboardInterrupt:
        print("\n[中断] 用户手动中止流水线执行。")
        raise


def run_pipeline(
    skip_data: bool = False,
    skip_sft: bool = False,
    skip_grpo: bool = False,
    eval_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Run full or partial pipeline based on filter flags."""
    print_banner("P2 模块流水线总入口 (Pipeline Master Runner)")
    print(f"工作区根目录: {WORKSPACE_ROOT}")
    print(f"Python 解释器: {sys.executable}")

    # 确定待执行步骤
    steps_to_run = []
    for step in PIPELINE_STEPS:
        step_id = step["id"]
        if eval_only and step_id != "eval":
            continue
        if skip_data and step_id == "data":
            continue
        if skip_sft and step_id == "sft":
            continue
        if skip_grpo and step_id == "grpo":
            continue
        steps_to_run.append(step)

    if not steps_to_run:
        print("[提示] 所有阶段均被跳过，无任务执行。")
        return

    print(f"计划执行步骤数: {len(steps_to_run)}")
    for i, s in enumerate(steps_to_run, 1):
        print(f"  ({i}) {s['title']} -> python {s['script_rel'].as_posix()}")

    total_start = time.perf_counter()
    for idx, step in enumerate(steps_to_run, 1):
        success = execute_step(idx, len(steps_to_run), step, dry_run=dry_run)
        if not success:
            print_banner(f"流水线终止：步骤 [{step['title']}] 执行失败，退出运行。")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start
    print_banner(f"P2 流水线全部执行完成！总耗时: {total_elapsed:.2f} 秒。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete p2 training and evaluation pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Skip synthetic dataset generation")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT QLoRA training")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO RL training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (hf_tester.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    run_pipeline(
        skip_data=args.skip_data,
        skip_sft=args.skip_sft,
        skip_grpo=args.skip_grpo,
        eval_only=args.eval_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
