#!/usr/bin/env python3
"""
Train Gemma3-270M student model using DPO (Direct Preference Optimization).

This implements Phase 2 of distillation using TRL's DPOTrainer to refine
the student model by learning to prefer teacher-like outputs.

Usage:
    uv run python scripts/distillation/train_dpo.py \
        --base-model models/gemma3-270m-student-unsloth-v1 \
        --dataset data/gpt5nano/train_dpo.jsonl \
        --val-dataset data/gpt5nano/val.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import wandb

# Set memory limits early
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"


def find_latest_version(base_name: str, models_dir: Path = Path("models")) -> int:
    """
    Find the latest version number for a given base model name.

    Args:
        base_name: Base name pattern (e.g., "gemma3-270m-student-dpo")
        models_dir: Directory containing model checkpoints

    Returns:
        Latest version number found, or 0 if none exist
    """
    if not models_dir.exists():
        return 0

    # Pattern: {base_name}-v{number}
    pattern = re.compile(rf"{re.escape(base_name)}-v(\d+)$")

    versions = []
    for path in models_dir.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

    return max(versions) if versions else 0


def get_next_version(base_name: str, models_dir: Path = Path("models")) -> str:
    """
    Get the next version name for a model.

    Args:
        base_name: Base name pattern (e.g., "gemma3-270m-student-dpo")
        models_dir: Directory containing model checkpoints

    Returns:
        Full model name with incremented version (e.g., "gemma3-270m-student-dpo-v2")
    """
    latest = find_latest_version(base_name, models_dir)
    next_version = latest + 1
    return f"{base_name}-v{next_version}"


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_dpo_dataset(data: List[Dict[str, Any]], tokenizer) -> Dataset:
    """
    Format DPO preference dataset for TRL.

    Args:
        data: List of examples with "prompt", "chosen", "rejected"
        tokenizer: Tokenizer for formatting

    Returns:
        HuggingFace Dataset ready for DPOTrainer
    """
    formatted = []

    for example in data:
        # Get prompt messages
        prompt_messages = example.get("prompt", [])

        # Format prompt using chat template
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Remove BOS token prefix as per Unsloth docs
        prompt_text = prompt_text.removeprefix('<bos>')

        formatted.append({
            "prompt": prompt_text,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        })

    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="Train DPO model with TRL")
    parser.add_argument(
        "--base-model",
        default="models/gemma3-270m-student-unsloth-v1",
        help="Base student model path to refine"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/gpt5nano/train_dpo.jsonl"),
        help="DPO preference dataset"
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=Path("data/gpt5nano/val_dpo.jsonl"),
        help="Validation dataset (DPO format with prompt/chosen/rejected)"
    )
    parser.add_argument(
        "--output-base",
        default="gemma3-270m-student-dpo",
        help="Base name for output model (version will be auto-incremented)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (DPO typically uses lower LR than SFT)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (KL penalty weight)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=74.0,
        help="Maximum GPU memory to use in GB (default: 74GB)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size (defaults to train batch size // 2 for memory efficiency)"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=100,
        help="Maximum number of samples to use for evaluation (default: 100)"
    )

    args = parser.parse_args()

    # Set eval batch size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = max(1, args.batch_size // 2)

    # Calculate memory limits
    max_memory = {0: f"{int(args.max_memory_gb * 0.9)}GB"}  # 90% of limit for GPU 0

    # Set PyTorch CUDA memory fraction
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_fraction = (args.max_memory_gb * 1024**3) / total_memory
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    print(f"   Set memory fraction to {memory_fraction:.2%} of total GPU memory")

    # Auto-increment version
    output_dir = get_next_version(args.output_base)
    output_path = Path("models") / output_dir

    print("=" * 70)
    print("DPO Training with TRL + Unsloth")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Output model: {output_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Max GPU memory: {args.max_memory_gb}GB")
    print(f"Train batch size: {args.batch_size}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print("=" * 70)

    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project="summarizer-distillation-dpo",
            settings=wandb.Settings(
                x_stats_open_metrics_endpoints={
                    "sparky": "http://localhost:9400/metrics"
                }
            ),
            name=output_dir,
            config={
                "base_model": args.base_model,
                "method": "DPO",
                "beta": args.beta,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
            }
        )

    # Load model and tokenizer with memory limits
    print("\n1. Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
        max_memory=max_memory,
    )

    # Set up chat template for Gemma-3
    print("   Setting up Gemma-3 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    # Load reference model for DPO (frozen copy of base model)
    print("\n2. Loading reference model...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
        max_memory=max_memory,
    )

    # Add LoRA adapters to the model (but not ref_model)
    print("\n3. Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load datasets
    print("\n4. Loading datasets...")
    train_data = load_jsonl(args.dataset)
    print(f"   Loaded {len(train_data)} training examples")

    train_dataset = format_dpo_dataset(train_data, tokenizer)
    print(f"   Formatted {len(train_dataset)} training examples")

    # Verify formatting with a sample
    print("\n" + "=" * 70)
    print("Sample DPO Training Example (first example):")
    print("=" * 70)
    sample = train_dataset[0]
    print("PROMPT:")
    print(sample["prompt"][:300])
    if len(sample["prompt"]) > 300:
        print(f"... (truncated, full length: {len(sample['prompt'])} chars)")
    print("\nCHOSEN:")
    print(sample["chosen"][:200])
    if len(sample["chosen"]) > 200:
        print(f"... (truncated, full length: {len(sample['chosen'])} chars)")
    print("\nREJECTED:")
    print(sample["rejected"][:200])
    if len(sample["rejected"]) > 200:
        print(f"... (truncated, full length: {len(sample['rejected'])} chars)")
    print("=" * 70)

    # Load validation dataset if provided (DPO format)
    eval_dataset = None
    if args.val_dataset.exists():
        val_data = load_jsonl(args.val_dataset)
        # Limit eval dataset size to prevent OOM
        if len(val_data) > args.max_eval_samples:
            print(f"   Limiting eval dataset from {len(val_data)} to {args.max_eval_samples} samples")
            val_data = val_data[:args.max_eval_samples]
        # Use same DPO formatting as train (val data already has prompt/chosen/rejected)
        eval_dataset = format_dpo_dataset(val_data, tokenizer)
        print(f"   Formatted {len(eval_dataset)} validation examples")

    # DPO Training Configuration
    print("\n5. Configuring DPO trainer...")
    training_args = DPOConfig(
        # Output
        output_dir=str(output_path),
        overwrite_output_dir=True,

        # Training hyperparameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,  # Use smaller eval batch
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,

        # DPO specific
        beta=args.beta,
        loss_type="sigmoid",  # Standard DPO loss

        # Optimizer
        optim="paged_adamw_8bit",
        weight_decay=0.01,

        # Scheduler
        lr_scheduler_type="cosine",
        warmup_steps=50,

        # Precision
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        # Logging
        logging_steps=5,
        logging_first_step=True,
        report_to="wandb" if not args.no_wandb else "none",

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=25 if eval_dataset else None,

        # Checkpointing
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,

        # Performance
        gradient_checkpointing=False,  # Disabled due to checkpoint error with DPO
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,

        # Misc
        seed=42,
        dataloader_num_workers=2,
    )

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train!
    print("\n6. Starting DPO training...")
    print("=" * 70)
    trainer.train()

    # Save final model
    print("\n7. Saving final model...")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Merge LoRA and save full model
    print("\n8. Merging LoRA weights...")
    merged_path = output_path.parent / f"{output_path.name}_merged"
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print("\n" + "=" * 70)
    print("DPO Training Complete!")
    print("=" * 70)
    print(f"✅ Adapter saved to: {output_path}")
    print(f"✅ Merged model saved to: {merged_path}")
    print(f"\nNext steps:")
    print(f"  1. Export to GGUF: just export-gguf {output_path}")
    print(f"  2. Import to Ollama: just ollama-import {output_dir}")
    print(f"  3. Test: ollama run {output_dir} 'Fix the login bug'")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
