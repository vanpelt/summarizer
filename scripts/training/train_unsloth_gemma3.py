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
import re
import torch
from pathlib import Path

# Disable torch.compile to avoid stride mismatch errors on GB10
torch._dynamo.config.disable = True
os.environ["PYTORCH_JIT"] = "0"

# Set memory limits early
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb

# Configuration matching your Axolotl config
MODEL_NAME = "unsloth/gemma-3-270m-it-bnb-4bit"  # Unsloth's pre-quantized version
MAX_SEQ_LENGTH = 2048

# LoRA configuration (matching your config)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0  # Set to 0 for Unsloth fast patching (0.1 causes performance hit)

def get_next_version(base_path: str) -> tuple[str, str]:
    """
    Given a path like './models/gemma3-270m-synthetic-v1', find the next available version.

    Returns:
        tuple[str, str]: (output_dir, run_name) with the next available version number

    Examples:
        './models/gemma3-270m-synthetic-v1' -> './models/gemma3-270m-synthetic-v1' (if doesn't exist)
        './models/gemma3-270m-synthetic-v1' -> './models/gemma3-270m-synthetic-v2' (if v1 exists)
        './models/gemma3-270m-synthetic-v5' -> './models/gemma3-270m-synthetic-v6' (if v5 exists)
    """
    path = Path(base_path)

    # If the base path doesn't exist, use it as-is
    if not path.exists() and not Path(f"{base_path}_merged").exists():
        return base_path, path.name

    # Extract base name and current version
    match = re.match(r'(.+)-v(\d+)$', path.name)
    if not match:
        # No version suffix, just append -v2 if path exists
        new_path = f"{base_path}-v2"
        return new_path, Path(new_path).name

    base_name = match.group(1)
    current_version = int(match.group(2))

    # Find the next available version
    parent = path.parent
    next_version = current_version
    while True:
        candidate = parent / f"{base_name}-v{next_version}"
        candidate_merged = Path(f"{candidate}_merged")
        if not candidate.exists() and not candidate_merged.exists():
            return str(candidate), candidate.name
        next_version += 1

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
        default=2,
        help="Number of training epochs (default: 3)"
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum number of training steps. If set, overrides --epochs. -1 means use epochs (default: -1)"
    )

    args = parser.parse_args()

    # Auto-increment version if output directory exists
    original_output_dir = args.output_dir
    args.output_dir, args.run_name = get_next_version(args.output_dir)

    if args.output_dir != original_output_dir:
        print(f"[INFO] Output directory '{original_output_dir}' exists, using '{args.output_dir}' instead")

    print("=" * 60)
    print("Unsloth Gemma3-270M Training on DGX Spark")
    print("=" * 60)
    print(f"Train data: {args.train_data}")
    print(f"Eval data: {args.eval_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Max GPU memory: {args.max_memory_gb}GB")

    # Calculate and display training steps
    if args.max_steps > 0:
        print(f"Training mode: max_steps = {args.max_steps} (ignoring epochs)")
    else:
        print(f"Training mode: {args.epochs} epochs")

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
    
    wandb.setup(wandb.Settings(
        x_stats_open_metrics_endpoints={
            "sparky": "http://localhost:9400/metrics"
        }
    ))

    # Load model with Unsloth optimizations
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bfloat16 on DGX Spark)
        load_in_4bit=True,
        max_memory=max_memory,
    )

    # Set up chat template for Gemma-3
    print("Setting up Gemma-3 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
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

    # Format function for chat template (following Unsloth best practices)
    def format_chat(examples):
        texts = []
        for conversations in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Remove BOS token prefix as per Unsloth docs
            text = text.removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}

    print("Formatting datasets with Gemma-3 chat template...")
    train_dataset = train_dataset.map(format_chat, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_chat, batched=True, remove_columns=eval_dataset.column_names)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(eval_dataset)}")

    # Calculate and display estimated training steps
    effective_batch_size = args.batch_size * 4  # gradient_accumulation_steps = 4
    steps_per_epoch = len(train_dataset) // effective_batch_size
    if args.max_steps > 0:
        total_steps = args.max_steps
        estimated_epochs = total_steps / steps_per_epoch
        print(f"\nEstimated training:")
        print(f"  Steps per epoch: ~{steps_per_epoch}")
        print(f"  Total steps: {total_steps} (will train for ~{estimated_epochs:.2f} epochs)")
    else:
        total_steps = steps_per_epoch * args.epochs
        print(f"\nEstimated training:")
        print(f"  Steps per epoch: ~{steps_per_epoch}")
        print(f"  Total steps: ~{total_steps} ({args.epochs} epochs)")
        print(f"  Estimated time: {total_steps * 0.5 / 60:.1f}-{total_steps * 1.0 / 60:.1f} minutes")

    # Verify formatting with a sample
    print("\n" + "=" * 60)
    print("Sample formatted example (first training example):")
    print("=" * 60)
    print(train_dataset[0]["text"][:500])
    if len(train_dataset[0]["text"]) > 500:
        print(f"... (truncated, full length: {len(train_dataset[0]['text'])} chars)")
    print("=" * 60)

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
        warmup_steps=20,
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_steps=25,
        save_steps=50,
        save_total_limit=3,
        max_steps=args.max_steps,  # Use --max-steps if provided, else -1 (use epochs)
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

    # Train only on model responses (not user prompts)
    # This is critical for proper fine-tuning!
    print("Configuring trainer to only train on model responses...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Verify what we're training on
    print("\n" + "=" * 60)
    print("Verifying training data (with labels):")
    print("=" * 60)
    print("Input IDs (first 100 tokens):")
    print(tokenizer.decode(trainer.train_dataset[0]["input_ids"][:100]))
    print("\nFull example:")
    print(tokenizer.decode(trainer.train_dataset[0]["input_ids"])[:500])
    print("=" * 60)

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

    # Print helpful next steps
    print("\n" + "=" * 60)
    print("Next Steps: Export to GGUF and Ollama")
    print("=" * 60)

    # Extract model name from output directory for GGUF export
    model_name = Path(args.output_dir).name

    print(f"\n1. Export to GGUF format (uses LoRA adapter):")
    print(f"   just export-gguf {args.output_dir}")
    print(f"\n   Or with different quantization (Q4_K_M is default):")
    print(f"   just export-gguf {args.output_dir} \"\" Q5_K_M")
    print(f"   just export-gguf {args.output_dir} \"\" Q8_0")
    print(f"\n   Or with custom name:")
    print(f"   just export-gguf {args.output_dir} my-custom-name")

    print(f"\n2. Import into Ollama:")
    print(f"   just ollama-import {model_name}")
    print(f"\n   Or with custom Ollama name:")
    print(f"   just ollama-import {model_name} my-custom-name")

    print(f"\n3. Test your model:")
    print(f"   ollama run {model_name} 'Fix the login bug on mobile devices'")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
