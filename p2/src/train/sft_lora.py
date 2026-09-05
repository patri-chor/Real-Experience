"""SFT Cold-Start with QLoRA for Qwen2.5-0.5B."""

import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def run_sft():
    # Model config
    # model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    # Because downloading might take long, we specify the ID but in practice
    # user will need internet connection.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "post_train.jsonl"
    output_dir = Path(__file__).resolve().parent.parent.parent / "models" / "sft_lora_0.5B"
    
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['messages'])):
            text = tokenizer.apply_chat_template(example['messages'][i], tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,     # Extremely small for 4GB VRAM
        gradient_accumulation_steps=4,     # Equivalent batch size 4
        optim="paged_adamw_8bit",
        logging_steps=5,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=2,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,       # Crucial for 4GB VRAM
        remove_unused_columns=False,
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=1024,
    )

    print("Starting SFT training...")
    trainer.train()
    
    print(f"Saving SFT model to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("SFT complete!")

if __name__ == "__main__":
    run_sft()
