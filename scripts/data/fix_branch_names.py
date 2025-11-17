#!/usr/bin/env python3
"""
Fix branch names in two-line dataset that are missing the '/' separator.

Converts formats like 'feat-my-branch' to 'feat/my-branch' for known prefixes.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Known valid prefixes
VALID_PREFIXES = ['bug', 'feat', 'feature', 'fix', 'chore', 'test', 'docs', 'refactor', 'style', 'perf']


def fix_branch_name(branch: str) -> tuple[str, bool]:
    """
    Fix a branch name if it's missing the '/' separator.

    Args:
        branch: The branch name to fix

    Returns:
        Tuple of (fixed_branch, was_fixed)
    """
    # If it already has a slash, it's fine
    if '/' in branch:
        return branch, False

    # Check if it starts with a known prefix followed by hyphen
    parts = branch.split('-', 1)
    if len(parts) == 2 and parts[0] in VALID_PREFIXES:
        # Fix it: prefix-name -> prefix/name
        fixed = f"{parts[0]}/{parts[1]}"
        return fixed, True

    # Can't fix it
    return branch, False


def fix_file(input_path: Path, output_path: Path):
    """
    Fix branch names in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    print(f"Fixing {input_path} -> {output_path}")

    fixed_count = 0
    total_count = 0
    skipped_count = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                total_count += 1

                # Find the assistant message
                for message in data.get('conversations', []):
                    if message.get('role') == 'assistant':
                        content = message.get('content', '')
                        lines = content.strip().split('\n')

                        if len(lines) >= 2:
                            summary = lines[0]
                            branch = lines[1]

                            # Try to fix the branch name
                            fixed_branch, was_fixed = fix_branch_name(branch)

                            if was_fixed:
                                # Update the content with fixed branch
                                message['content'] = f"{summary}\n{fixed_branch}"
                                fixed_count += 1

                # Write the (possibly fixed) entry
                outfile.write(json.dumps(data) + '\n')

            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                skipped_count += 1

    print(f"  Total: {total_count}")
    print(f"  Fixed: {fixed_count}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count}")
    print()


def main():
    """Main function."""
    print("=" * 60)
    print("Fixing Branch Names in Two-Line Dataset")
    print("=" * 60)
    print()

    # Define input and output paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "data" / "synthetic_two_line"

    # Fix in place (overwrite the files)
    files_to_fix = ["train.jsonl", "test.jsonl"]

    # Create backup first
    backup_dir = input_dir / "backup"
    backup_dir.mkdir(exist_ok=True)

    for filename in files_to_fix:
        input_path = input_dir / filename

        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping...")
            continue

        # Backup original
        backup_path = backup_dir / filename
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"Backed up {filename} to {backup_path}")

        # Fix and overwrite
        temp_path = input_dir / f"{filename}.tmp"
        fix_file(input_path, temp_path)

        # Replace original with fixed version
        temp_path.replace(input_path)

    print("=" * 60)
    print("Fixing Complete!")
    print("=" * 60)
    print(f"\nBackups saved to: {backup_dir}")
    print(f"Fixed files in: {input_dir}")
    print()


if __name__ == "__main__":
    main()
