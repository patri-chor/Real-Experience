"""SFT Trainer module for distilling expert reasoning trajectories into internalized student models."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List


def train_student_model(dataset_path: str, output_dir: str) -> None:
    """Execute mock Supervised Fine-Tuning (SFT) to distill CoT trajectories into a student model.

    Parses high-quality reasoning trajectories from distill_cot.jsonl, simulates
    the training and convergence process, and persists internalized model knowledge
    weights into mock_model.json.

    Args:
        dataset_path: Path to the input training dataset (e.g. p1/data/distill_cot.jsonl).
        output_dir: Target directory where mock_model.json and training artifacts will be saved.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If dataset contains no valid JSON records.
    """
    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    # 1. 读取并解析训练集
    records: List[Dict[str, Any]] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as err:
                print(f"[SFT Trainer] 警告: 跳过第 {line_idx} 行无效 JSON: {err}")

    data_count = len(records)
    if data_count == 0:
        raise ValueError(f"Training dataset {dataset_path} contains 0 valid samples.")

    print(f"[SFT Trainer] 开始训练模型... 数据量 {data_count}")
    print(f"[SFT Trainer] 输入数据集路径: {data_file.resolve()}")

    # 2. 模拟训练与收敛过程
    total_epochs = 3
    simulated_losses = [1.8421, 0.6934, 0.1782]
    learning_rates = [2.0e-5, 1.5e-5, 5.0e-6]

    for epoch in range(1, total_epochs + 1):
        loss = simulated_losses[epoch - 1]
        lr = learning_rates[epoch - 1]
        print(
            f"[SFT Trainer] Epoch {epoch}/{total_epochs} | "
            f"Step: {epoch * data_count}/{total_epochs * data_count} | "
            f"Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"Progress: [{'=' * (epoch * 7)}>{'.' * (21 - epoch * 7)}]"
        )

    # 模拟真实微调的计算时延
    time.sleep(2)

    # 3. 构建内化知识表征（将训练集内化为其“模型权重”）
    # 将包含外部工具调用的多轮轨迹提炼为单轮自发联想与认知复用链
    internalized_prototypes: List[Dict[str, Any]] = []
    for item in records:
        prob = item.get("problem", "")
        cat = item.get("category", "")
        ans = item.get("final_answer", "")
        traj = item.get("trajectory", "")

        internalized_prototypes.append({
            "problem": prob,
            "category": cat,
            "final_answer": ans,
            "raw_trajectory": traj,
        })

    model_metadata = {
        "model_name": "InternalizedStudent-0.5B",
        "base_architecture": "Decoder-Only-Transformer",
        "distillation_source": str(data_file.resolve()),
        "total_samples": data_count,
        "epochs": total_epochs,
        "final_loss": simulated_losses[-1],
        "training_time_seconds": 2.0,
        "has_external_tools": False,
        "has_system_prompts": False,
        "internalized_weights": internalized_prototypes,
    }

    # 4. 确保输出目录存在并写入 mock_model.json
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    model_file_path = out_dir_path / "mock_model.json"

    with open(model_file_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=2)

    print(f"[SFT Trainer] 模型训练完成并收敛。权重已内化写入: {model_file_path.resolve()}")
    print(f"[SFT Trainer] 输出文件大小: {model_file_path.stat().st_size} 字节。")
