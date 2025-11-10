#!/usr/bin/env python3
"""
Generate synthetic training data for Qwen3-8B fine-tuning.

This script uses Claude API to generate diverse software development tasks
and corresponding JSON outputs with title and git branch names.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()

# Task categories for diverse training data
TASK_CATEGORIES = [
    "features",
    "bug fixes",
    "refactoring",
    "documentation",
    "testing",
    "performance optimization",
    "security improvements",
    "UI/UX enhancements",
    "API changes",
    "database migrations",
]

# System prompt for consistent JSON generation
SYSTEM_PROMPT = """You are a helpful assistant that generates JSON output.
Always respond with valid JSON in this exact format:
{"title": "short description", "branch_name": "git-branch-name"}

Rules:
- title: concise, 3-7 words, describes the task clearly
- branch_name: lowercase, kebab-case, prefix with type (feat/, fix/, docs/, test/, refactor/, perf/, chore/)
- No markdown code blocks, just raw JSON"""


def generate_task_prompts(client: anthropic.Anthropic, num_prompts: int) -> List[str]:
    """Generate diverse task prompts using Claude."""
    console.print(f"[cyan]Generating {num_prompts} task prompts...[/cyan]")

    generation_prompt = f"""Generate {num_prompts} diverse software development task descriptions.
Each should be realistic, specific, and vary in complexity and category.

Categories to cover: {', '.join(TASK_CATEGORIES)}

Examples:
- "Add dark mode toggle to settings page"
- "Fix memory leak in image processing pipeline"
- "Update API documentation for authentication endpoints"
- "Implement rate limiting for API endpoints"
- "Refactor database connection pooling logic"

Generate a JSON array of task descriptions only, no other text:"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{"role": "user", "content": generation_prompt}]
    )

    # Extract JSON array from response
    content = response.content[0].text.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    tasks = json.loads(content)
    console.print(f"[green]Generated {len(tasks)} task prompts[/green]")
    return tasks


def generate_training_example(
    client: anthropic.Anthropic, task: str
) -> Dict[str, Any]:
    """Generate a single training example from a task description."""
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": task}]
    )

    json_output = response.content[0].text.strip()

    # Validate JSON output
    try:
        parsed = json.loads(json_output)
        assert "title" in parsed and "branch_name" in parsed
    except (json.JSONDecodeError, AssertionError) as e:
        console.print(f"[yellow]Warning: Invalid JSON for task '{task}': {e}[/yellow]")
        console.print(f"[yellow]Response: {json_output}[/yellow]")
        raise

    # Format as chat messages for Axolotl
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates JSON with a title and git branch name."
            },
            {"role": "user", "content": task},
            {"role": "assistant", "content": json_output}
        ]
    }


def generate_dataset(
    num_examples: int = 1000,
    output_dir: Path = Path("data/processed"),
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> None:
    """Generate complete synthetic dataset and split into train/val/test."""

    # Initialize Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not found in environment[/red]")
        console.print("[yellow]Create a .env file with: ANTHROPIC_API_KEY=your_key[/yellow]")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate task prompts
    tasks = generate_task_prompts(client, num_examples)

    # Generate training examples
    console.print(f"[cyan]Generating {num_examples} training examples...[/cyan]")
    training_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Generating examples...", total=len(tasks)
        )

        for i, task_desc in enumerate(tasks):
            try:
                example = generate_training_example(client, task_desc)
                training_data.append(example)
                progress.update(task, advance=1, description=f"Generated {i+1}/{len(tasks)}")
            except Exception as e:
                console.print(f"[red]Error generating example {i+1}: {e}[/red]")
                continue

    console.print(f"[green]Successfully generated {len(training_data)} examples[/green]")

    # Shuffle and split data
    random.shuffle(training_data)

    train_size = int(len(training_data) * train_split)
    val_size = int(len(training_data) * val_split)

    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]

    # Save datasets
    def save_jsonl(data: List[Dict], filepath: Path):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")

    console.print(f"\n[green]Dataset saved to {output_dir}[/green]")
    console.print(f"  Train: {len(train_data)} examples")
    console.print(f"  Val: {len(val_data)} examples")
    console.print(f"  Test: {len(test_data)} examples")

    # Save a few examples for inspection
    console.print(f"\n[cyan]Sample training examples:[/cyan]")
    for i, example in enumerate(train_data[:3], 1):
        console.print(f"\n[bold]Example {i}:[/bold]")
        console.print(f"  User: {example['messages'][1]['content']}")
        console.print(f"  Assistant: {example['messages'][2]['content']}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Qwen3-8B fine-tuning"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of training examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for generated data (default: data/processed)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set split ratio (default: 0.1)"
    )

    args = parser.parse_args()

    console.print("[bold cyan]Synthetic Data Generation for Qwen3-8B[/bold cyan]\n")

    generate_dataset(
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
