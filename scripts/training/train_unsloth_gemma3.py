#!/usr/bin/env python3
"""
Unsloth training script for Gemma3-270M on DGX Spark
Converted from configs/qlora-gemma3-270m-student.yml
"""

import os
import torch

# Disable torch.compile to avoid stride mismatch errors on GB10
torch._dynamo.config.disable = True
os.environ["PYTORCH_JIT"] = "0"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration matching your Axolotl config
MODEL_NAME = "unsloth/gemma-3-270m-it-bnb-4bit"  # Unsloth's pre-quantized version
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/gemma3-270m-student-unsloth-v1"

# LoRA configuration (matching your config)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0  # Set to 0 for Unsloth fast patching (0.1 causes performance hit)

def main():
    print("=" * 60)
    print("Unsloth Gemma3-270M Training on DGX Spark")
    print("=" * 60)

    # Clear any cached compiled kernels
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # Load model with Unsloth optimizations
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bfloat16 on DGX Spark)
        load_in_4bit=True,
    )

    # Add LoRA adapters with Unsloth
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    # Load datasets (Unsloth format with 'conversations' field)
    print("\nLoading datasets...")
    train_dataset = load_dataset("json", data_files="data/gpt5nano_unsloth/train.jsonl", split="train")
    eval_dataset = load_dataset("json", data_files="data/gpt5nano_unsloth/val.jsonl", split="train")

    # Format function for chat template
    def format_chat(examples):
        texts = []
        for conversations in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    train_dataset = train_dataset.map(format_chat, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_chat, batched=True, remove_columns=eval_dataset.column_names)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(eval_dataset)}")

    # Training arguments (matching your Axolotl config)
    print("\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_steps=25,
        save_steps=50,
        save_total_limit=3,
        max_steps=-1,
        seed=42,
        report_to="wandb",
        run_name="gemma3-270m-student-unsloth",
        eval_strategy="steps",  # Enable evaluation
    )

    # Initialize Unsloth's optimized trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Pack multiple samples per sequence for efficiency
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer_stats = trainer.train()

    # Save model
    print("\n" + "=" * 60)
    print(f"Saving model to {OUTPUT_DIR}")
    print("=" * 60)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save as merged model (LoRA + base)
    merged_dir = f"{OUTPUT_DIR}_merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nAdapter saved to: {OUTPUT_DIR}")
    print(f"Merged model saved to: {merged_dir}")
    print(f"\nTraining stats: {trainer_stats}")

if __name__ == "__main__":
    main()
