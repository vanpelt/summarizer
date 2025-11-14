#!/usr/bin/env python3
"""
Generate teacher model outputs for knowledge distillation.

This script uses the Gemma3-27B model (teacher) to generate high-quality
outputs for the training dataset. These outputs will be used to train
the smaller Gemma3-270M student model.

Usage:
    # Using Ollama (if available)
    uv run python scripts/distillation/generate_teacher_outputs.py --backend ollama

    # Using vLLM server
    uv run python scripts/distillation/generate_teacher_outputs.py --backend vllm
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm


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
    temperature: float = 0.7
) -> str:
    """Generate using Ollama API."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 256,
            }
        }
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


def main():
    parser = argparse.ArgumentParser(description="Generate teacher model outputs")
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help="Backend to use for generation"
    )
    parser.add_argument(
        "--teacher-model",
        default="gemma3:27b",
        help="Teacher model name (for Ollama) or path (for vLLM)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/train.jsonl"),
        help="Input training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/train_teacher.jsonl"),
        help="Output file with teacher generations"
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
        help="Base URL for vLLM server"
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} examples")

    # Generate teacher outputs
    teacher_data = []

    for example in tqdm(data, desc="Generating teacher outputs"):
        messages = example["messages"]

        # Extract user message (skip system message)
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if not user_messages:
            print(f"Warning: No user message found in example, skipping")
            continue

        # Prepare messages for teacher
        teacher_messages = [
            msg for msg in messages if msg["role"] in ["system", "user"]
        ]

        # Generate with teacher model
        try:
            if args.backend == "ollama":
                teacher_output = generate_with_ollama(
                    args.teacher_model,
                    teacher_messages,
                    args.temperature
                )
            else:  # vllm
                teacher_output = generate_with_vllm(
                    args.teacher_model,
                    teacher_messages,
                    args.temperature,
                    args.vllm_base_url
                )

            # Create new example with teacher output
            teacher_example = {
                "messages": teacher_messages + [
                    {"role": "assistant", "content": teacher_output}
                ]
            }
            teacher_data.append(teacher_example)

        except Exception as e:
            print(f"\nError generating for example: {e}")
            # Fall back to original output if generation fails
            teacher_data.append(example)

    # Save teacher outputs
    print(f"\nSaving {len(teacher_data)} examples to {args.output}")
    save_jsonl(teacher_data, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
