#!/usr/bin/env python3
"""
Extend DPO dataset with synthetic prompts.

This script:
1. Loads existing training data prompts
2. Generates additional synthetic prompts using Claude
3. Creates a combined dataset file
4. Calls generate_dpo_dataset.py to generate teacher/student outputs with batching

Usage:
    # Generate 500 synthetic prompts and create DPO dataset
    uv run python scripts/distillation/extend_dpo_with_synthetic.py \
        --existing-data data/gpt5nano/train.jsonl \
        --num-synthetic 500 \
        --teacher-model gemma3:27b \
        --student-model models/gemma3-270m-student-unsloth-v1 \
        --output data/gpt5nano/train_dpo_extended.jsonl \
        --batch-size 4
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import anthropic
from dotenv import load_dotenv

load_dotenv()


# Production system prompt
SYSTEM_PROMPT = """You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
Never include explanations or extra keys.

Turn this request for code changes into:
1) a 2–4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case)."""


# Task categories for diverse prompts
TASK_CATEGORIES = [
    "new features",
    "bug fixes",
    "refactoring",
    "documentation",
    "testing",
    "performance optimization",
    "security improvements",
    "UI/UX enhancements",
    "API changes",
    "database migrations",
    "deployment",
    "configuration",
    "error handling",
    "logging and monitoring",
    "code cleanup",
]


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


def generate_synthetic_prompts(num_prompts: int, existing_prompts: List[str] = None) -> List[str]:
    """
    Generate synthetic task prompts using Claude.

    Args:
        num_prompts: Number of prompts to generate
        existing_prompts: Existing prompts to avoid duplicates

    Returns:
        List of synthetic prompts
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

    existing_set = set(existing_prompts or [])
    synthetic_prompts = []

    print(f"\nGenerating {num_prompts} synthetic prompts using Claude...")

    # Generate in batches to show progress
    batch_size = 50
    num_batches = (num_prompts + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        remaining = num_prompts - len(synthetic_prompts)
        current_batch_size = min(batch_size, remaining)

        # Pick a random category for this batch
        category = random.choice(TASK_CATEGORIES)

        prompt = f"""Generate {current_batch_size} diverse, realistic code change requests related to {category}.

Requirements:
- Each should be 5-15 words
- Focus on {category}
- Use natural, conversational language (like GitHub issue titles)
- Vary complexity from simple to detailed
- Include different technical areas (frontend, backend, API, database, etc.)

Return ONLY a JSON array of strings, no other text:
["request 1", "request 2", ...]

Examples of good requests:
- "Add dark mode toggle to settings page"
- "Fix memory leak in image processing worker"
- "Refactor authentication middleware to use JWT"
- "Update API documentation for v2 endpoints"
"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            text = response.content[0].text.strip()
            batch_prompts = json.loads(text)

            # Filter out duplicates
            for p in batch_prompts:
                if p not in existing_set and p not in synthetic_prompts:
                    synthetic_prompts.append(p)
                    if len(synthetic_prompts) >= num_prompts:
                        break

            print(f"   Generated {len(synthetic_prompts)}/{num_prompts} prompts...")

            if len(synthetic_prompts) >= num_prompts:
                break

        except Exception as e:
            print(f"   Warning: Batch {batch_idx + 1} failed: {e}")
            continue

    return synthetic_prompts[:num_prompts]


def main():
    parser = argparse.ArgumentParser(description="Extend DPO dataset with synthetic prompts")
    parser.add_argument(
        "--existing-data",
        type=Path,
        default=Path("data/gpt5nano/train.jsonl"),
        help="Existing training data (optional, for reference)"
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=500,
        help="Number of synthetic prompts to generate"
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["ollama", "vllm", "openai", "auto"],
        default="auto",
        help="Backend for teacher model (auto: detect from model name)"
    )
    parser.add_argument(
        "--teacher-model",
        default="gemma3:27b",
        help="Teacher model name (for Ollama/OpenAI) or path (for vLLM)"
    )
    parser.add_argument(
        "--student-model",
        default="models/gemma3-270m-student-unsloth-v1",
        help="Student model path (Unsloth format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gpt5nano/train_dpo_extended.jsonl"),
        help="Output extended DPO preference dataset"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--use-existing-only",
        action="store_true",
        help="Only use existing data, don't generate synthetic prompts"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for student model inference (passed to generate_dpo_dataset.py)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save progress every N examples (passed to generate_dpo_dataset.py)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server (if using vllm backend)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama server"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Extended DPO Dataset Generation")
    print("=" * 70)

    # Load existing prompts for reference
    existing_prompts = []
    all_prompts = []

    if args.existing_data.exists():
        print(f"\n1. Loading existing data from {args.existing_data}")
        existing_data = load_jsonl(args.existing_data)

        # Extract user prompts from existing data
        for example in existing_data:
            messages = example.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if user_msg:
                existing_prompts.append(user_msg)

        print(f"   Found {len(existing_prompts)} existing prompts")
        all_prompts.extend(existing_prompts)

    # Check if we can reuse existing combined dataset when resuming
    temp_input = args.output.parent / f"{args.output.stem}_combined.jsonl"
    synthetic_prompts = []

    if args.resume and temp_input.exists():
        print(f"\n2. Resume mode: checking existing combined dataset...")
        temp_data = load_jsonl(temp_input)
        temp_prompts = []
        for example in temp_data:
            messages = example.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if user_msg:
                temp_prompts.append(user_msg)

        # Count how many are synthetic (not in existing)
        existing_set = set(existing_prompts)
        temp_synthetic = [p for p in temp_prompts if p not in existing_set]

        if len(temp_synthetic) >= args.num_synthetic:
            print(f"   Found {len(temp_synthetic)} synthetic prompts in existing combined dataset")
            print(f"   Skipping synthetic generation (already have >= {args.num_synthetic})")
            synthetic_prompts = temp_synthetic[:args.num_synthetic]
            all_prompts.extend(synthetic_prompts)
        else:
            print(f"   Found only {len(temp_synthetic)} synthetic prompts, need {args.num_synthetic}")
            print(f"   Generating {args.num_synthetic - len(temp_synthetic)} more...")
            synthetic_prompts = generate_synthetic_prompts(
                args.num_synthetic - len(temp_synthetic),
                existing_prompts=existing_prompts + temp_synthetic
            )
            all_prompts.extend(temp_synthetic + synthetic_prompts)
    elif not args.use_existing_only:
        print(f"\n2. Generating {args.num_synthetic} synthetic prompts...")
        synthetic_prompts = generate_synthetic_prompts(
            args.num_synthetic,
            existing_prompts=existing_prompts
        )
        all_prompts.extend(synthetic_prompts)

    if not args.use_existing_only:
        print(f"   Total prompts: {len(all_prompts)} ({len(existing_prompts)} existing + {len(all_prompts) - len(existing_prompts)} synthetic)")
    else:
        print(f"\n2. Using only existing {len(existing_prompts)} prompts")

    # Create combined dataset with all prompts
    # Use the production system prompt that matches our task
    print(f"\n3. Creating combined dataset with {len(all_prompts)} prompts...")
    combined_data = []
    for prompt in all_prompts:
        combined_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                # No assistant message - generate_dpo_dataset.py will add responses
            ]
        })

    # Save to temporary file (temp_input already defined above)
    save_jsonl(combined_data, temp_input)
    print(f"   Saved combined dataset to: {temp_input}")

    # Call generate_dpo_dataset.py with batching support
    print(f"\n4. Generating teacher/student preference pairs with batching...")
    print(f"   This will use batch_size={args.batch_size} for efficient processing")

    cmd = [
        sys.executable,
        "scripts/distillation/generate_dpo_dataset.py",
        "--teacher-backend", args.teacher_backend,
        "--teacher-model", args.teacher_model,
        "--student-model", args.student_model,
        "--input", str(temp_input),
        "--output", str(args.output),
        "--temperature", str(args.temperature),
        "--batch-size", str(args.batch_size),
        "--save-every", str(args.save_every),
        "--vllm-base-url", args.vllm_base_url,
        "--ollama-base-url", args.ollama_base_url,
    ]

    if args.resume:
        cmd.append("--resume")

    print(f"\n   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ Extended DPO dataset generation complete!")
        print("=" * 70)
        print(f"Output: {args.output}")
        print(f"Total examples: {len(all_prompts)}")
        print(f"  - Existing: {len(existing_prompts)}")
        print(f"  - Synthetic: {len(all_prompts) - len(existing_prompts)}")

        # Clean up temp file
        if temp_input.exists():
            temp_input.unlink()
            print(f"\nCleaned up temporary file: {temp_input}")
    else:
        print(f"\n❌ Error: generate_dpo_dataset.py failed with exit code {result.returncode}")
        print(f"Temporary combined dataset saved at: {temp_input}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
