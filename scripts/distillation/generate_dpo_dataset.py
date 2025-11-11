#!/usr/bin/env python3
"""
Generate DPO preference dataset for Phase 2 distillation.

This script creates preference pairs where:
- "chosen": Teacher model (Gemma3-27B) outputs (high quality)
- "rejected": Student model (Gemma3-270M) outputs (lower quality)

The DPO trainer will then train the student to prefer teacher-like outputs.

Usage:
    # Generate preference dataset
    uv run python scripts/distillation/generate_dpo_dataset.py \
        --teacher-backend ollama \
        --teacher-model gemma3:27b \
        --student-model models/gemma3-270m-student-unsloth-v1 \
        --input data/gpt5nano/train.jsonl \
        --output data/gpt5nano/train_dpo.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
from openai import OpenAI


# Production system prompt from prompt.txt
SYSTEM_PROMPT = """You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
Never include explanations or extra keys.

Turn this request for code changes into:
1) a 2–4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case)."""


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """Save to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def generate_with_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    enforce_json: bool = True,
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Generate using Ollama API with optional JSON schema enforcement.

    Args:
        model: Model name in Ollama
        messages: Chat messages
        temperature: Generation temperature
        enforce_json: If True, enforce JSON schema with summary and branch_name
        base_url: Base URL for Ollama server

    Returns:
        Generated text (JSON string if enforce_json=True)
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        }
    }

    # Add JSON schema enforcement if requested (matches prompt.txt schema)
    # This uses Ollama's structured output feature
    if enforce_json:
        payload["format"] = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2-4 words, Title Case, no punctuation"
                },
                "branch": {
                    "type": "string",
                    "description": "kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with category"
                }
            },
            "required": ["summary", "branch"]
        }

    response = requests.post(
        f"{base_url}/api/chat",
        json=payload
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def generate_with_vllm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    base_url: str = "http://localhost:8000/v1"
) -> str:
    """Generate using vLLM OpenAI-compatible API."""
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 256,
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_with_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    api_key: str = None
) -> str:
    """
    Generate using OpenAI API.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-5-mini")
        messages: Chat messages
        temperature: Generation temperature
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)

    Returns:
        Generated text
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=api_key)

    # Add response format for structured output (matches prompt.txt schema)
    # Note: No max_tokens or temperature - structured output uses default temperature=1
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "git_task_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "2-4 words, Title Case, no punctuation"
                        },
                        "branch": {
                            "type": "string",
                            "description": "kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with category like bug/, feat/, etc."
                        }
                    },
                    "required": ["summary", "branch"],
                    "additionalProperties": False
                }
            }
        }
    )

    return response.choices[0].message.content


def generate_with_unsloth(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 256
) -> str:
    """Generate using loaded Unsloth model (single example)."""
    # Format messages using chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the prompt)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_with_unsloth_batch(
    model,
    tokenizer,
    batch_messages: List[List[Dict[str, str]]],
    temperature: float = 0.7,
    max_tokens: int = 256
) -> List[str]:
    """Generate using loaded Unsloth model (batched for efficiency)."""
    # Format all messages using chat template
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated parts (skip the prompts)
    results = []
    for i, output in enumerate(outputs):
        # Find where the prompt ends
        prompt_length = inputs['input_ids'][i].shape[0]
        generated = output[prompt_length:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(decoded)

    # Free GPU memory
    del inputs, outputs
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference dataset")
    parser.add_argument(
        "--teacher-backend",
        choices=["ollama", "vllm", "openai", "auto"],
        default="auto",
        help="Backend for teacher model (auto: detect from model name)"
    )
    parser.add_argument(
        "--teacher-model",
        default="gemma3:27b",
        help="Teacher model name (e.g., gemma3:27b, gpt-4o-mini, gpt-5-mini)"
    )
    parser.add_argument(
        "--student-model",
        default="models/gemma3-270m-student-unsloth-v1",
        help="Student model path (Unsloth format)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gpt5nano/train.jsonl"),
        help="Input training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gpt5nano/train_dpo.jsonl"),
        help="Output DPO preference dataset"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server (if using vllm backend)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for student model inference (higher = faster but more VRAM). Use 1-2 for 270M model."
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save progress every N examples (default: 10)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed examples)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama server (default works with --net=host Docker)"
    )

    args = parser.parse_args()

    # Auto-detect backend from model name
    if args.teacher_backend == "auto":
        if args.teacher_model.startswith("gpt-"):
            args.teacher_backend = "openai"
            print(f"🔍 Auto-detected backend: OpenAI (model starts with 'gpt-')")
        elif "/" in args.teacher_model or args.teacher_model.startswith("models/"):
            args.teacher_backend = "vllm"
            print(f"🔍 Auto-detected backend: vLLM (model path detected)")
        else:
            args.teacher_backend = "ollama"
            print(f"🔍 Auto-detected backend: Ollama (default for model: {args.teacher_model})")

    # Validate OpenAI API key if using OpenAI
    if args.teacher_backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("   Set it with: export OPENAI_API_KEY=your-key")
            sys.exit(1)

    # Load training data
    print(f"\nLoading data from {args.input}")
    data = load_jsonl(args.input)
    if args.max_examples:
        data = data[:args.max_examples]
    print(f"Loaded {len(data)} examples")

    # Check for existing progress if resuming
    processed_indices = set()
    if args.resume and args.output.exists():
        print(f"\n📁 Resume mode: Loading existing progress from {args.output}")
        existing_data = load_jsonl(args.output)
        # Track which original indices were processed
        # Assuming we store the index in the output (we'll add this)
        processed_indices = {item.get("source_index", -1) for item in existing_data}
        processed_indices.discard(-1)  # Remove invalid indices
        print(f"   Found {len(processed_indices)} already processed examples")
        print(f"   Remaining: {len(data) - len(processed_indices)} examples")

    # Load student model
    print(f"\nLoading student model from: {args.student_model}")
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(student_model)
    print("Student model loaded!")

    # Prepare output file for incremental writes
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in append mode if resuming, write mode otherwise
    output_file = open(args.output, 'a' if args.resume else 'w')

    # Track errors separately
    errors = []
    error_file = args.output.parent / f"{args.output.stem}_errors.jsonl"

    # Statistics
    total_processed = len(processed_indices)
    total_errors = 0

    print(f"\nGenerating preference pairs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Saving every: {args.save_every} examples")

    # Process in batches for efficiency
    batch_data = []
    batch_indices = []

    try:
        for idx, example in enumerate(tqdm(data, desc="Processing examples")):
            # Skip if already processed
            if idx in processed_indices:
                continue

            messages = example["messages"]

            # Extract user content
            user_content = None
            for msg in messages:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

            if not user_content:
                print(f"\nWarning: No user message in example {idx}, skipping")
                continue

            # Build proper prompt messages with production system prompt
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            batch_data.append((idx, prompt_messages, messages))
            batch_indices.append(idx)

            # Process batch when full
            if len(batch_data) >= args.batch_size:
                process_batch(
                    batch_data,
                    args,
                    student_model,
                    student_tokenizer,
                    output_file,
                    errors
                )
                total_processed += len(batch_data)
                batch_data = []
                batch_indices = []

        # Process remaining batch
        if batch_data:
            process_batch(
                batch_data,
                args,
                student_model,
                student_tokenizer,
                output_file,
                errors
            )
            total_processed += len(batch_data)

    finally:
        # Always close the file
        output_file.close()

        # Save errors if any
        if errors:
            print(f"\n💾 Saving {len(errors)} errors to {error_file}")
            save_jsonl(errors, error_file)

    print("\n" + "=" * 60)
    print("DPO Dataset Generation Complete!")
    print("=" * 60)
    print(f"✅ Success: {total_processed}/{len(data)} examples")
    print(f"❌ Errors: {len(errors)}/{len(data)} examples")
    print(f"\nOutput: {args.output}")
    print(f"\nNext step:")
    print(f"  just train-dpo")


def process_batch(
    batch_data,
    args,
    student_model,
    student_tokenizer,
    output_file,
    errors
):
    """Process a batch of examples with teacher and student generation."""
    indices = [item[0] for item in batch_data]
    prompt_messages_list = [item[1] for item in batch_data]
    original_messages_list = [item[2] for item in batch_data]

    # Generate student outputs in batch (fast!)
    try:
        student_outputs = generate_with_unsloth_batch(
            student_model,
            student_tokenizer,
            prompt_messages_list,
            args.temperature
        )
    except Exception as e:
        print(f"\n❌ Batch student generation failed: {e}")
        # Fall back to sequential processing
        student_outputs = []
        for prompt_msgs in prompt_messages_list:
            try:
                output = generate_with_unsloth(
                    student_model,
                    student_tokenizer,
                    prompt_msgs,
                    args.temperature
                )
                student_outputs.append(output)
            except Exception as e2:
                student_outputs.append(f"ERROR: {str(e2)}")

    # Generate teacher outputs (sequential - APIs don't batch well)
    teacher_outputs = []
    for idx, prompt_msgs in zip(indices, prompt_messages_list):
        try:
            if args.teacher_backend == "ollama":
                teacher_output = generate_with_ollama(
                    args.teacher_model,
                    prompt_msgs,
                    args.temperature,
                    enforce_json=True,
                    base_url=args.ollama_base_url
                )
            elif args.teacher_backend == "openai":
                teacher_output = generate_with_openai(
                    args.teacher_model,
                    prompt_msgs,
                    args.temperature
                )
            else:  # vllm
                teacher_output = generate_with_vllm(
                    args.teacher_model,
                    prompt_msgs,
                    args.temperature,
                    args.vllm_base_url
                )
            teacher_outputs.append(teacher_output)
        except Exception as e:
            error_msg = f"Example {idx}: Teacher generation failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            errors.append({
                "index": idx,
                "error": error_msg,
                "messages": prompt_msgs
            })
            teacher_outputs.append(None)

    # Write successful examples to file immediately
    for idx, prompt_msgs, orig_msgs, teacher_out, student_out in zip(
        indices, prompt_messages_list, original_messages_list, teacher_outputs, student_outputs
    ):
        if teacher_out is None:
            continue  # Skip failed examples

        dpo_example = {
            "source_index": idx,  # Track which example this came from
            "prompt": prompt_msgs,
            "chosen": teacher_out,
            "rejected": student_out,
            "original_messages": orig_msgs,
        }

        # Write immediately (one line per example)
        output_file.write(json.dumps(dpo_example) + '\n')
        output_file.flush()  # Force write to disk


if __name__ == "__main__":
    main()
