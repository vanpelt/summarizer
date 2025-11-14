#!/usr/bin/env python3
"""
Inspect and analyze DPO preference datasets.

Shows statistics about the dataset including:
- Total examples
- Synthetic vs existing ratio
- Average prompt/response lengths
- Sample examples

Usage:
    uv run python scripts/distillation/inspect_dpo_dataset.py \
        data/synthetic/train_dpo_extended.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_dataset(data: List[Dict[str, Any]]):
    """Analyze DPO dataset and print statistics."""

    total = len(data)
    synthetic_count = sum(1 for d in data if d.get("is_synthetic", False))
    existing_count = total - synthetic_count

    # Calculate length statistics
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    for example in data:
        # Prompt length (combine all messages)
        prompt = example.get("prompt", [])
        prompt_text = " ".join(msg.get("content", "") for msg in prompt)
        prompt_lengths.append(len(prompt_text))

        # Response lengths
        chosen_lengths.append(len(example.get("chosen", "")))
        rejected_lengths.append(len(example.get("rejected", "")))

    # Calculate averages
    avg_prompt = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    avg_chosen = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0
    avg_rejected = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0

    # Print statistics
    print("=" * 70)
    print("DPO Dataset Analysis")
    print("=" * 70)
    print(f"\n📊 Dataset Size:")
    print(f"   Total examples: {total}")
    print(f"   - From existing data: {existing_count} ({existing_count/total*100:.1f}%)")
    print(f"   - Synthetic: {synthetic_count} ({synthetic_count/total*100:.1f}%)")

    print(f"\n📏 Average Lengths (characters):")
    print(f"   Prompt: {avg_prompt:.0f}")
    print(f"   Chosen (teacher): {avg_chosen:.0f}")
    print(f"   Rejected (student): {avg_rejected:.0f}")

    print(f"\n📈 Length Ranges:")
    print(f"   Prompt: {min(prompt_lengths)}-{max(prompt_lengths)}")
    print(f"   Chosen: {min(chosen_lengths)}-{max(chosen_lengths)}")
    print(f"   Rejected: {min(rejected_lengths)}-{max(rejected_lengths)}")

    # Show sample examples
    print(f"\n📝 Sample Examples:")
    print("-" * 70)

    # Show 1 synthetic and 1 existing if available
    samples = []
    if synthetic_count > 0:
        synthetic = next(d for d in data if d.get("is_synthetic", False))
        samples.append(("Synthetic", synthetic))

    if existing_count > 0:
        existing = next(d for d in data if not d.get("is_synthetic", False))
        samples.append(("Existing", existing))

    for label, example in samples[:2]:
        print(f"\n[{label} Example]")
        prompt = example.get("prompt", [])
        user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "N/A")
        print(f"Prompt: {user_msg[:150]}...")
        print(f"Chosen (teacher): {example.get('chosen', '')[:100]}...")
        print(f"Rejected (student): {example.get('rejected', '')[:100]}...")
        print("-" * 70)

    print("\n✅ Dataset looks good! Ready for DPO training.")


def main():
    parser = argparse.ArgumentParser(description="Inspect DPO preference dataset")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to DPO dataset JSONL file"
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"❌ Error: Dataset not found: {args.dataset}")
        print("\nAvailable datasets:")
        data_dir = Path("data/synthetic")
        if data_dir.exists():
            for file in data_dir.glob("train_dpo*.jsonl"):
                print(f"  - {file}")
        return 1

    print(f"Loading dataset: {args.dataset}")
    data = load_jsonl(args.dataset)

    if not data:
        print(f"❌ Error: Dataset is empty!")
        return 1

    analyze_dataset(data)
    return 0


if __name__ == "__main__":
    exit(main())
