"""Inference and evaluation tester for Qwen2.5-0.5B with PEFT LoRA adapter.

Loads 4-bit quantized base model and attaches GRPO LoRA adapter (falling back
to SFT LoRA if GRPO adapter is unavailable). Evaluates model on isomorphic
and unrelated prompts, sampling full chain-of-thought and answers.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
P2_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = P2_ROOT.parent

MODELS_DIR = P2_ROOT / "models"
GRPO_ADAPTER_DIR = MODELS_DIR / "grpo_lora_0.5B"
SFT_ADAPTER_DIR = MODELS_DIR / "sft_lora_0.5B"


def resolve_adapter_path(
    custom_adapter: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Path], str]:
    """Resolve available adapter path with fallback priority:

    custom -> grpo -> sft -> None.
    """
    if custom_adapter:
        p = Path(custom_adapter).resolve()
        if p.exists():
            return p, "custom"
        print(f"[警告] 指定的适配器路径不存在: {p}，尝试自动查找...")

    if (GRPO_ADAPTER_DIR / "adapter_config.json").exists():
        return GRPO_ADAPTER_DIR, "grpo_lora_0.5B"
    elif GRPO_ADAPTER_DIR.is_dir() and any(GRPO_ADAPTER_DIR.iterdir()):
        return GRPO_ADAPTER_DIR, "grpo_lora_0.5B"

    if (SFT_ADAPTER_DIR / "adapter_config.json").exists():
        return SFT_ADAPTER_DIR, "sft_lora_0.5B"
    elif SFT_ADAPTER_DIR.is_dir() and any(SFT_ADAPTER_DIR.iterdir()):
        return SFT_ADAPTER_DIR, "sft_lora_0.5B"

    return None, "none"


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any, Optional[Path]]:
    """Load 4-bit quantized base model and attach PEFT LoRA adapter.

    Returns:
        Tuple of (model, tokenizer, loaded_adapter_path).
    """
    print(f"[*] 正在加载 Tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chosen_adapter, adapter_type = resolve_adapter_path(adapter_path)

    print(f"[*] 正在配置基座模型加载策略 (4-bit NF4)...")
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[*] 正在加载 4-bit 基座模型 ({model_id}) 到 GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("[!] 未检测到可用 CUDA GPU，平滑回退至 CPU float32 加载...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    if chosen_adapter is not None:
        print(f"[*] 正在挂载 LoRA 适配器 [{adapter_type}]: {chosen_adapter}")
        model = PeftModel.from_pretrained(model, str(chosen_adapter))
        print(f"[*] LoRA 适配器挂载成功。")
    else:
        print("[!] 未检测到任何训练好的 LoRA 适配器，将使用基座模型进行推理对比。")

    model.eval()
    return model, tokenizer, chosen_adapter


def run_inference(
    prompts: Optional[List[str]] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: Optional[Union[str, Path]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """Run model inference on given prompt list with sampling and parse outputs."""
    if prompts is None:
        prompts = [
            "A car travels from City A to City B at 100 km/h and returns at 60 km/h. Distance is 300 km. Average speed?",
            "What is 47 + 89?",
        ]

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_and_tokenizer(
            model_id=model_id,
            adapter_path=adapter_path,
        )

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"                       开始执行推理测试 (共 {len(prompts)} 题)")
    print("=" * 80)

    for idx, prompt_text in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] 测试输入:")
        print(f"      Q: {prompt_text}")
        print("-" * 80)

        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_token_len = inputs["input_ids"].shape[1]

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_sec = time.perf_counter() - t_start

        generated_tokens = outputs[0][input_token_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        thought_match = re.search(r"<thought>(.*?)</thought>", response_text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

        thought_content = thought_match.group(1).strip() if thought_match else None
        answer_content = answer_match.group(1).strip() if answer_match else None

        print(f"【推理耗时】: {elapsed_sec:.2f} 秒 ({len(generated_tokens)} tokens)")
        if thought_content:
            print(f"【抽取思维链 (<thought>)】:\n{thought_content}")
        else:
            print("【抽取思维链 (<thought>)】: 未能匹配到成对的 <thought> 标签")

        if answer_content:
            print(f"【抽取最终答案 (<answer>)】: {answer_content}")
        else:
            print("【抽取最终答案 (<answer>)】: 未能匹配到成对的 <answer> 标签")

        print("-" * 80)
        print(f"【完整生成文本】:\n{response_text}")
        print("=" * 80)

        results.append({
            "prompt": prompt_text,
            "raw_response": response_text,
            "thought": thought_content,
            "answer": answer_content,
            "tokens_generated": len(generated_tokens),
            "elapsed_seconds": elapsed_sec,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Tester for Qwen2.5-0.5B LoRA.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model HF repo or local path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Explicit path to LoRA adapter folder",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max generation length",
    )
    args = parser.parse_args()

    run_inference(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
