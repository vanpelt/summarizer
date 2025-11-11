#!/usr/bin/env python3
"""
Clean and prepare dataset for DPO training.

This script:
1. Removes continuation prompts (starting with "This session is being continued")
2. Updates system prompt to match production format (prompt.txt)
3. Validates JSON structure

Usage:
    python scripts/data/clean_dataset.py \
        --input data/gpt5nano/train.jsonl \
        --output data/gpt5nano/train_cleaned.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


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


def clean_dataset(data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Clean dataset by filtering and updating prompts.

    Returns:
        Tuple of (cleaned_data, stats)
    """
    cleaned = []
    stats = {
        "total": len(data),
        "continuation_filtered": 0,
        "system_prompt_updated": 0,
        "kept": 0
    }

    for item in data:
        messages = item.get("messages", [])

        # Find user message
        user_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg
                break

        if not user_msg:
            continue

        # Filter out continuation prompts
        user_content = user_msg["content"]
        if user_content.startswith("This session is being continued from a previous"):
            stats["continuation_filtered"] += 1
            continue

        # Update system prompt to production format
        updated_messages = []
        for msg in messages:
            if msg["role"] == "system":
                updated_messages.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT
                })
                stats["system_prompt_updated"] += 1
            else:
                updated_messages.append(msg)

        # Keep the item with updated messages
        cleaned_item = {
            **item,
            "messages": updated_messages
        }
        cleaned.append(cleaned_item)
        stats["kept"] += 1

    return cleaned, stats


def main():
    parser = argparse.ArgumentParser(description="Clean and prepare dataset")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output cleaned JSONL file"
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.input}")
    data = load_jsonl(args.input)

    print(f"\nCleaning dataset...")
    cleaned_data, stats = clean_dataset(data)

    print(f"\nSaving cleaned dataset to {args.output}")
    save_jsonl(cleaned_data, args.output)

    print("\n" + "=" * 60)
    print("Dataset Cleaning Complete!")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"Continuation prompts filtered: {stats['continuation_filtered']}")
    print(f"System prompts updated: {stats['system_prompt_updated']}")
    print(f"Examples kept: {stats['kept']}")
    print(f"\nReduction: {stats['continuation_filtered']}/{stats['total']} ({100*stats['continuation_filtered']/stats['total']:.1f}%)")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
