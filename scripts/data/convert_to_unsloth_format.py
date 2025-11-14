#!/usr/bin/env python3
"""
Convert dataset from 'messages' format to Unsloth's 'conversations' format
Also update the system message
"""

import json
import argparse
from pathlib import Path

# New system message from the user
NEW_SYSTEM_MESSAGE = """You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
Never include explanations or extra keys.

Turn this request for code changes into:
1) a 2–4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case)."""


def convert_example(example, merge_system=False):
    """
    Convert from messages format to conversations format

    Args:
        example: Dict with 'messages' field
        merge_system: If True, merge system message into first user message

    Returns:
        Dict with 'conversations' field
    """
    messages = example.get("messages", [])

    # Find system, user, and assistant messages
    system_msg = None
    conversations = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_msg = NEW_SYSTEM_MESSAGE  # Use updated system message
        elif role == "user":
            if merge_system and system_msg:
                # Merge system message into user message
                content = f"{system_msg}\n\nRequest:\n{content}"
                system_msg = None  # Only merge once
            conversations.append({"role": "user", "content": content})
        elif role == "assistant":
            conversations.append({"role": "assistant", "content": content})

    # If not merging, add system message at the start
    if not merge_system and system_msg:
        conversations.insert(0, {"role": "system", "content": system_msg})

    return {"conversations": conversations}


def convert_file(input_path: Path, output_path: Path, merge_system: bool = False):
    """Convert a JSONL file to Unsloth format"""

    print(f"Converting {input_path} → {output_path}")
    print(f"Merge system into user: {merge_system}")

    converted = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            example = json.loads(line)
            converted_example = convert_example(example, merge_system=merge_system)
            f_out.write(json.dumps(converted_example) + "\n")
            converted += 1

    print(f"✓ Converted {converted} examples")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to Unsloth format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Input directory with train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_unsloth"),
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--merge-system",
        action="store_true",
        help="Merge system message into first user message (recommended for Gemma 3)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each file
    for filename in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        input_path = args.input_dir / filename
        output_path = args.output_dir / filename

        if input_path.exists():
            convert_file(input_path, output_path, merge_system=args.merge_system)
        else:
            print(f"⚠ Skipping {filename} (not found)")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nConverted data saved to: {args.output_dir}")
    print("\nExample before:")
    with open(args.input_dir / "train.jsonl") as f:
        example = json.loads(f.readline())
        print(json.dumps(example, indent=2)[:300] + "...")

    print("\nExample after:")
    with open(args.output_dir / "train.jsonl") as f:
        example = json.loads(f.readline())
        print(json.dumps(example, indent=2)[:300] + "...")


if __name__ == "__main__":
    main()
