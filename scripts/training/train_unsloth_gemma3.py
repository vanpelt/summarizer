#!/usr/bin/env python3
"""
Unsloth training script for Gemma3-270M on DGX Spark

Usage:
    # Train with synthetic dataset (default, uses 74GB GPU memory by default)
    uv run python scripts/training/train_unsloth_gemma3.py

    # Train with custom output directory
    uv run python scripts/training/train_unsloth_gemma3.py \
        --output-dir models/gemma3-270m-v2 \
        --run-name gemma3-270m-v2

    # Train with original gpt5nano dataset
    uv run python scripts/training/train_unsloth_gemma3.py \
        --train-data data/gpt5nano_unsloth/train.jsonl \
        --eval-data data/gpt5nano_unsloth/val.jsonl \
        --output-dir models/gemma3-270m-original \
        --run-name gemma3-270m-original

    # With custom settings
    uv run python scripts/training/train_unsloth_gemma3.py \
        --max-memory-gb 100 \
        --batch-size 4 \
        --epochs 3
"""

import argparse
import os
import torch

# Disable torch.compile to avoid stride mismatch errors on GB10
torch._dynamo.config.disable = True
os.environ["PYTORCH_JIT"] = "0"

# Set memory limits early
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration matching your Axolotl config
MODEL_NAME = "unsloth/gemma-3-270m-it-bnb-4bit"  # Unsloth's pre-quantized version
MAX_SEQ_LENGTH = 2048

# LoRA configuration (matching your config)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0  # Set to 0 for Unsloth fast patching (0.1 causes performance hit)

def main():
    parser = argparse.ArgumentParser(description="Train Gemma3-270M with Unsloth")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/synthetic/train.jsonl",
        help="Path to training data (default: data/synthetic/train.jsonl)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/synthetic/test.jsonl",
        help="Path to eval data (default: data/synthetic/test.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/gemma3-270m-synthetic-v1",
        help="Output directory for model (default: ./models/gemma3-270m-synthetic-v1)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="gemma3-270m-synthetic-v1",
        help="W&B run name (default: gemma3-270m-synthetic-v1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size (default: 8)"
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=74.0,
        help="Maximum GPU memory to use in GB (default: 74GB)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Unsloth Gemma3-270M Training on DGX Spark")
    print("=" * 60)
    print(f"Train data: {args.train_data}")
    print(f"Eval data: {args.eval_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Max GPU memory: {args.max_memory_gb}GB")

    # Calculate memory limits (always set, default 74GB)
    max_memory = {0: f"{int(args.max_memory_gb)}GB"}

    # Set PyTorch CUDA memory fraction
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_fraction = (args.max_memory_gb * 1024**3) / total_memory
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    print(f"   Set memory fraction to {memory_fraction:.2%} of total GPU memory")

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
        max_memory=max_memory,
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
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")

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
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
        run_name=args.run_name,
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
    print(f"Saving model to {args.output_dir}")
    print("=" * 60)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save as merged model (LoRA + base)
    merged_dir = f"{args.output_dir}_merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nAdapter saved to: {args.output_dir}")
    print(f"Merged model saved to: {merged_dir}")
    print(f"\nTraining stats: {trainer_stats}")

if __name__ == "__main__":
    main()
