"""GRPO Reinforcement Learning for Qwen2.5-0.5B."""

import re
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("WARNING: trl version too old for GRPOTrainer. Please update trl>=0.15 or similar.")
    GRPOTrainer = None
    GRPOConfig = None

# --- Reward Functions ---

def format_reward_func(completions, **kwargs):
    """Reward if the response follows the exact <thought>...</thought><answer>...</answer> format."""
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        # Check basic XML tags
        has_thought = "<thought>" in text and "</thought>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        
        # Check cognitive stages if it's a recall
        if "我回想起了一道相似的历史题" in text:
            has_analyze = "【分析旧题】" in text
            has_extract = "【提取逻辑】" in text
            has_modify = "【修改方案】" in text
            
            score = sum([has_thought, has_answer, has_analyze, has_extract, has_modify]) / 5.0
        else:
            score = sum([has_thought, has_answer]) / 2.0
            
        rewards.append(score)
    return rewards

def accuracy_reward_func(completions, solution, **kwargs):
    """Reward if the final extracted <answer> matches the ground truth solution."""
    rewards = []
    for comp, sol in zip(completions, solution):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            pred = match.group(1).strip()
            # Try to parse as float for robust comparison
            try:
                if abs(float(pred) - float(sol)) < 1e-4:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                if pred == str(sol):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def run_grpo():
    if GRPOTrainer is None:
        return
        
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "post_train.jsonl"
    sft_adapter_path = Path(__file__).resolve().parent.parent.parent / "models" / "sft_lora_0.5B"
    output_dir = Path(__file__).resolve().parent.parent.parent / "models" / "grpo_lora_0.5B"
    
    print("Loading dataset for GRPO...")
    # For GRPO, we need a dataset with 'prompt' and we need to pass 'solution' for the reward function
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    # Extract solution from assistant response (mocking ground truth available in dataset)
    def extract_gt(example):
        assist_msg = [m["content"] for m in example["messages"] if m["role"] == "assistant"][0]
        match = re.search(r"<answer>(.*?)</answer>", assist_msg)
        sol = match.group(1).strip() if match else ""
        return {"prompt": example["prompt"], "solution": sol}
        
    dataset = dataset.map(extract_gt)

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    model.load_adapter(str(sft_adapter_path)) # Load the SFT adapter as base

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing GRPOTrainer...")
    # Note: 4GB VRAM constraint => very small batch, tiny generation
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_prompt_length=128,
        max_completion_length=256,
        num_generations=2,   # G=2 to save VRAM (default is often 8)
        bf16=True,
        logging_steps=5,
        max_steps=50,       # Short run for POC
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()
    
    print(f"Saving GRPO model to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("GRPO complete!")

if __name__ == "__main__":
    run_grpo()
