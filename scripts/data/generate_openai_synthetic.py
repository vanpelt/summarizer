#!/usr/bin/env python3
"""
Generate synthetic training data using OpenAI API for Qwen3-8B fine-tuning.

This script reads prompts from data/raw/deduplicated_prompts.json and uses
OpenAI's GPT-4o-mini to generate structured JSON outputs with summary and
branch names, formatted for Axolotl training.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

console = Console()

# System prompt for consistent JSON generation
SYSTEM_PROMPT = """You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
Never include explanations or extra keys.

Turn this request for code changes into:
1) a 2–4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case).

Examples:
- Request: "Fix memory leak in image processing"
  Response: {"summary": "Fix Memory Leak", "branch": "bug/fix-memory-leak"}

- Request: "Add dark mode toggle to settings"
  Response: {"summary": "Add Dark Mode", "branch": "feat/add-dark-mode"}

- Request: "Refactor database connection pooling"
  Response: {"summary": "Refactor DB Pooling", "branch": "refactor/db-pooling"}"""


def load_prompts(filepath: Path) -> List[str]:
    """Load prompts from deduplicated_prompts.json."""
    console.print(f"[cyan]Loading prompts from {filepath}...[/cyan]")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract just the content field from each entry
    prompts = [item['content'] for item in data if 'content' in item]

    console.print(f"[green]Loaded {len(prompts)} prompts[/green]")
    return prompts


def generate_training_example(
    client: openai.OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Generate a single training example from a prompt."""
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=f"Request:\n\n{prompt}\n\nJson:",
            reasoning={"effort": "low"},
            text={"format": {"type": "json_object"}, "verbosity": "low"}  # Enforce JSON output
        )

        json_output = response.output_text

        # Validate JSON output
        try:
            parsed = json.loads(json_output)
        except ValueError:
            console.print(f"[red]Yikes: {json_output} ({response})")
            parsed = {}

        if "summary" not in parsed or "branch" not in parsed:
            raise ValueError(f"Missing required fields: {parsed}")

        # Format as chat messages for Axolotl (ChatML format)
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates JSON with a summary and git branch name for code change requests."
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json_output}
            ]
        }

    except Exception as e:
        console.print(f"[red]Error generating example: {e}[/red]")
        console.print(f"[yellow]Prompt: {prompt[:100]}...[/yellow]")
        raise


def generate_dataset(
    input_file: Path = Path("data/raw/deduplicated_prompts.json"),
    output_dir: Path = Path("data/processed"),
    model: str = "gpt-4o-mini",
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_prompts: int = None,
    skip_prompts: int = 0,
) -> None:
    """Generate complete synthetic dataset and split into train/val/test."""

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
        console.print("[yellow]Set it in .env file: OPENAI_API_KEY=your_key[/yellow]")
        return

    console.print(f"[green]Using OpenAI API key {api_key[:5]}[/green]")
    client = openai.OpenAI(api_key=api_key)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    all_prompts = load_prompts(input_file)

    # Apply skip and max limits
    if skip_prompts > 0:
        all_prompts = all_prompts[skip_prompts:]
        console.print(f"[cyan]Skipped first {skip_prompts} prompts[/cyan]")

    if max_prompts:
        prompts = all_prompts[:max_prompts]
        console.print(f"[cyan]Processing {len(prompts)} of {len(all_prompts)} prompts[/cyan]")
    else:
        prompts = all_prompts
        console.print(f"[cyan]Processing all {len(prompts)} prompts[/cyan]")

    # Generate training examples
    console.print(f"\n[bold cyan]Generating training examples with {model}...[/bold cyan]")
    training_data = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "Generating examples...", total=len(prompts)
        )

        for i, prompt in enumerate(prompts):
            try:
                example = generate_training_example(client, prompt, model)
                training_data.append(example)
                progress.update(
                    task,
                    advance=1,
                    description=f"Generated {i+1}/{len(prompts)} examples"
                )
            except Exception as e:
                errors.append((i, prompt[:100], str(e)))
                progress.update(task, advance=1)
                continue

    console.print(f"\n[green]Successfully generated {len(training_data)} examples[/green]")
    if errors:
        console.print(f"[yellow]Failed to generate {len(errors)} examples[/yellow]")

    # Split data (no shuffling to maintain determinism)
    train_size = int(len(training_data) * train_split)
    val_size = int(len(training_data) * val_split)

    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]

    # Save datasets in JSONL format (Axolotl standard)
    def save_jsonl(data: List[Dict], filepath: Path):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")

    console.print(f"\n[bold green]Dataset saved to {output_dir}[/bold green]")
    console.print(f"  Train: {len(train_data)} examples ({train_split*100:.0f}%)")
    console.print(f"  Val: {len(val_data)} examples ({val_split*100:.0f}%)")
    console.print(f"  Test: {len(test_data)} examples ({(1-train_split-val_split)*100:.0f}%)")

    # Save error log if any
    if errors:
        error_file = output_dir / "generation_errors.jsonl"
        with open(error_file, 'w') as f:
            for idx, prompt, error in errors:
                f.write(json.dumps({
                    "index": idx,
                    "prompt_preview": prompt,
                    "error": error
                }) + '\n')
        console.print(f"[yellow]Error log saved to {error_file}[/yellow]")

    # Show sample examples
    console.print(f"\n[bold cyan]Sample training examples:[/bold cyan]")
    for i, example in enumerate(train_data[:3], 1):
        console.print(f"\n[bold]Example {i}:[/bold]")
        user_msg = example['messages'][1]['content']
        assistant_msg = example['messages'][2]['content']
        console.print(f"  [dim]User:[/dim] {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")
        console.print(f"  [dim]Assistant:[/dim] {assistant_msg}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using OpenAI API for Qwen3-8B fine-tuning"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/raw/deduplicated_prompts.json"),
        help="Input file with prompts (default: data/raw/deduplicated_prompts.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for generated data (default: data/processed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (default: all)"
    )
    parser.add_argument(
        "--skip-prompts",
        type=int,
        default=0,
        help="Number of prompts to skip from the start (default: 0)"
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

    console.print("[bold cyan]OpenAI Synthetic Data Generation for Qwen3-8B[/bold cyan]\n")

    generate_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model=args.model,
        train_split=args.train_split,
        val_split=args.val_split,
        max_prompts=args.max_prompts,
        skip_prompts=args.skip_prompts,
    )


if __name__ == "__main__":
    main()
