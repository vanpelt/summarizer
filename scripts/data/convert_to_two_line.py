#!/usr/bin/env python3
"""
Convert JSON format training data to two-line format.

This script converts existing data/synthetic/*.jsonl files (JSON output format)
to a new two-line format where the model outputs:
  Line 1: Summary (2-4 words, Title Case)
  Line 2: Branch name (kebab-case with prefix)

Usage:
    python scripts/data/convert_to_two_line.py

This will:
- Read from data/synthetic/{train,test}.jsonl
- Convert JSON responses to two-line format
- Replace system prompt with two-line instructions
- Write to data/synthetic_two_line/{train,test}.jsonl
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SYSTEM_PROMPT_TWO_LINE


def convert_conversation(conversation: dict) -> dict:
    """
    Convert a single conversation from JSON format to two-line format.

    Creates a separate system message instead of embedding the prompt in the user message.

    Args:
        conversation: Dict with 'conversations' field containing user/assistant messages

    Returns:
        Converted conversation dict with two-line format and separate system message
    """
    converted = {"conversations": []}

    # Add system message first
    converted["conversations"].append({
        "role": "system",
        "content": SYSTEM_PROMPT_TWO_LINE.strip()
    })

    for message in conversation["conversations"]:
        if message["role"] == "user":
            # Extract just the actual request without the system prompt or "Request:" prefix
            content = message["content"]
            if "Request:\n" in content:
                # Split and take everything after "Request:\n"
                request_part = content.split("Request:\n", 1)[1].strip()
            else:
                # Fallback: try to find where the actual request starts
                # Look for common patterns that indicate the end of system prompt
                request_part = content.strip()

            converted["conversations"].append({
                "role": "user",
                "content": request_part
            })

        elif message["role"] == "assistant":
            # Parse the JSON response and convert to two-line format
            try:
                json_response = json.loads(message["content"])
                summary = json_response.get("summary", "")
                branch = json_response.get("branch", "")

                # Create two-line format
                two_line_response = f"{summary}\n{branch}"

                converted["conversations"].append({
                    "role": "assistant",
                    "content": two_line_response
                })
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON response: {message['content']}")
                # Keep original if we can't parse
                converted["conversations"].append(message)

    return converted


def convert_file(input_path: Path, output_path: Path):
    """
    Convert an entire JSONL file from JSON format to two-line format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    print(f"Converting {input_path} -> {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    failed_count = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                conversation = json.loads(line)
                converted = convert_conversation(conversation)
                outfile.write(json.dumps(converted) + '\n')
                converted_count += 1
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                failed_count += 1

    print(f"  Converted: {converted_count}")
    if failed_count > 0:
        print(f"  Failed: {failed_count}")
    print()


def main():
    """Main conversion function."""
    print("=" * 60)
    print("Converting JSON Format to Two-Line Format")
    print("=" * 60)
    print()

    # Define input and output paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "data" / "synthetic"
    output_dir = base_dir / "data" / "synthetic_two_line"

    files_to_convert = ["train.jsonl", "test.jsonl"]

    for filename in files_to_convert:
        input_path = input_dir / filename
        output_path = output_dir / filename

        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping...")
            continue

        convert_file(input_path, output_path)

    print("=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. Train a model on the two-line format:")
    print("   just unsloth-train-two-line")
    print("\n2. Or manually:")
    print("   uv run python scripts/training/train_unsloth_gemma3.py \\")
    print("     --train-data data/synthetic_two_line/train.jsonl \\")
    print("     --eval-data data/synthetic_two_line/test.jsonl \\")
    print("     --output-dir models/gemma3-270m-synthetic-two-line-v1 \\")
    print("     --run-name gemma3-270m-synthetic-two-line-v1")
    print()


if __name__ == "__main__":
    main()
