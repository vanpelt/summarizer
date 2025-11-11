#!/usr/bin/env python3
"""
TRL-based training script for Gemma3-270M on DGX Spark
Alternative to Axolotl that may have better GB10 compatibility
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration matching qlora-gemma3-270m-student.yml
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = "./models/gemma3-270m-student-trl-v1"
TRAIN_DATA = "data/gpt5nano/train.jsonl"
VAL_DATA = "data/gpt5nano/val.jsonl"

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration (matching your Axolotl config)
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments (matching your Axolotl config)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=0.0003,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    bf16=True,
    logging_steps=5,
    eval_steps=25,
    save_steps=50,
    save_total_limit=3,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=2048,
    packing=True,
    report_to="wandb",
    run_name="gemma3-270m-student-trl",
)

def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    # Load model with 4-bit quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print(f"Loading training data from {TRAIN_DATA}")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    eval_dataset = load_dataset("json", data_files=VAL_DATA, split="train")

    # Format function for chat template
    def format_chat(example):
        # Apply Gemma chat template
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(format_chat, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_chat, remove_columns=eval_dataset.column_names)

    # Initialize trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")

if __name__ == "__main__":
    main()
