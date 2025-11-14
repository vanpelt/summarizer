#!/usr/bin/env python3
"""
Generate enhanced dataset for Gemma3-270M fine-tuning.

This script uses OpenAI's gpt-5-mini or (OPENAI_MODEL) with low reasoning effort and structured outputs for reliable JSON generation.
API calls are made in parallel (default: 10 workers) for 10x faster generation.

Process:
1. Loads existing training data from gpt5nano
2. Generates 2000+ new synthetic user prompts using gpt-5-mini
3. Generates JSON responses with structured outputs (strict schema enforcement, low reasoning effort) in parallel
4. Converts to unsloth format
5. Splits into train/test datasets
6. Saves to data/synthetic/

Usage:
    # Preview mode: generate 5 synthetic examples and show them
    uv run python scripts/data/generate_enhanced_dataset.py --preview

    # Full generation: 2000 new prompts (10 parallel workers by default)
    uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 2000

    # Full generation with custom parallelism
    uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 2000 --max-workers 5

    # Resume: add 1000 MORE prompts to existing dataset
    uv run python scripts/data/generate_enhanced_dataset.py --resume --num-synthetic 1000
    # This will:
    # 1. Load existing synthetic prompts and labeled examples
    # 2. Generate 1000 NEW prompts (avoiding duplicates)
    # 3. Append to synthetic_prompts.jsonl
    # 4. Label only the NEW prompts
    # 5. Merge with existing train/test and re-split
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SYSTEM_PROMPT

load_dotenv()
console = Console()

# Task categories for diverse prompts (from extend_dpo_with_synthetic.py)
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
    "accessibility improvements",
    "mobile responsiveness",
    "internationalization",
    "caching strategies",
    "authentication and authorization",
]

# Programming languages
LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Ruby",
    "C",
    "C++",
    "Java",
    "Zig",
]

# Application types
APP_TYPES = [
    "Web App",
    "Mobile App",
    "Desktop App",
    "CLI Tool",
    "SDK",
    "Library",
    "API Service",
    "Microservice",
]

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Pydantic model for structured outputs
class CodeChangeResponse(BaseModel):
    """Response format for code change requests"""
    summary: str  # 2-4 words, Title Case, no punctuation
    branch: str   # kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with category


def load_existing_prompts(data_dir: Path) -> List[str]:
    """Load existing user prompts from gpt5nano/train.jsonl"""
    train_file = data_dir / "gpt5nano" / "train.jsonl"

    if not train_file.exists():
        console.print(f"[yellow]Warning: {train_file} not found[/yellow]")
        return []

    prompts = []
    filtered_count = 0
    with open(train_file) as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if user_msg:
                # Filter out "continued session" prompts
                if user_msg.startswith("This session is being continued from"):
                    filtered_count += 1
                    continue
                prompts.append(user_msg)

    console.print(f"[green]Loaded {len(prompts)} existing prompts[/green]")
    if filtered_count > 0:
        console.print(f"[yellow]Filtered out {filtered_count} 'continued session' prompts[/yellow]")
    return prompts


def clean_prompt(prompt: str) -> str:
    """
    Clean up generated prompts by removing length prefixes and extra whitespace.

    Removes prefixes like "Short:", "MEDIUM:", "long:", etc. that leak from generation examples.
    """
    # Remove length prefixes (case-insensitive, with optional colon and whitespace)
    import re
    cleaned = re.sub(r'^\s*(short|medium|long)\s*:?\s*', '', prompt, flags=re.IGNORECASE)
    # Strip extra whitespace
    cleaned = cleaned.strip()
    return cleaned


def generate_synthetic_prompts(
    client: openai.OpenAI,
    num_prompts: int,
    existing_prompts: List[str] = None,
    max_workers: int = 10,
    focus_long: bool = False
) -> List[str]:
    """
    Generate synthetic user prompts using OpenAI with parallel API calls.

    Pre-calculates a grid of (category, language, app_type) combinations to ensure diversity
    without duplicates. Uses parallel API calls for faster generation.

    Args:
        client: OpenAI client
        num_prompts: Number of prompts to generate
        existing_prompts: Existing prompts to avoid duplicates
        max_workers: Number of parallel workers
        focus_long: If True, generate mostly long prompts (80% long, 15% medium, 5% short)
    """
    console.print(f"\n[cyan]Generating {num_prompts} synthetic user prompts with {max_workers} parallel workers...[/cyan]")
    if focus_long:
        console.print(f"[yellow]Focus mode: Generating mostly LONG prompts (80% long, 15% medium, 5% short)[/yellow]")

    existing_set = set(existing_prompts or [])

    # Generate grid of combinations
    batch_size = 50  # Each API call generates this many prompts
    num_batches = (num_prompts + batch_size - 1) // batch_size

    console.print(f"[dim]Creating {num_batches} batches with diverse language/app combinations[/dim]")

    # Distribution based on focus_long flag
    if focus_long:
        short_pct, medium_pct, long_pct = 5, 15, 80
    else:
        short_pct, medium_pct, long_pct = 10, 30, 60

    # Create combinations grid (avoiding duplicates by cycling through systematically)
    combinations = []
    for i in range(num_batches):
        category = TASK_CATEGORIES[i % len(TASK_CATEGORIES)]
        language = LANGUAGES[i % len(LANGUAGES)]
        app_type = APP_TYPES[i % len(APP_TYPES)]
        combinations.append((category, language, app_type, min(batch_size, num_prompts - i * batch_size)))

    def generate_batch(combo_idx: int, category: str, language: str, app_type: str, batch_size: int) -> List[str]:
        """Generate a single batch of prompts"""
        prompt = f"""Generate {batch_size} diverse, realistic code change requests for a {language} {app_type} related to {category}.

Requirements:
- Context: {language} {app_type}
- Task category: {category}
- Vary length from short (10-50 words) to very long (100-1000+ words)
- Use natural, conversational language (like GitHub issues or Slack messages)
- {short_pct}% should be concise (10-50 words, just the request)
- {medium_pct}% should have context or explanation (50-150 words)
- {long_pct}% should include {language} code snippets, error logs, stack traces, or detailed reproduction steps (150-1000+ words)
- Include different technical areas (frontend, backend, API, database, etc.)
- For longer requests, include realistic {language} code examples, error messages, logs, or stack traces
- Make them specific to {language} {app_type} conventions and best practices
- DO NOT prefix requests with length labels like "Short:", "MEDIUM:", or "Long:"

Return a JSON object with a "requests" key containing an array of strings:
{{"requests": ["request 1", "request 2", ...]}}

Examples (note: NO length prefixes):

Add dark mode toggle to settings page
---
The user authentication flow is breaking on Safari. Users can log in but the session cookie isn't being set properly, causing them to be logged out on page refresh. Need to investigate cookie settings and SameSite attributes.
---
Getting a crash in the image upload handler. Here's the stack trace:

Traceback (most recent call last):
  File "/home/vanpelt/Development/lab/summary-finetune/scripts/data/generate_enhanced_dataset.py", line 271, in generate_synthetic_prompts
    for future in as_completed(future_to_combo):
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 243, in as_completed
    waiter.event.wait(wait_timeout)
  File "/usr/lib/python3.12/threading.py", line 655, in wait
    signaled = self._cond.wait(timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/threading.py", line 355, in wait
    waiter.acquire()
KeyboardInterrupt

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/vanpelt/Development/lab/summary-finetune/scripts/data/generate_enhanced_dataset.py", line 690, in <module>
    exit(main())
         ^^^^^^
  File "/home/vanpelt/Development/lab/summary-finetune/scripts/data/generate_enhanced_dataset.py", line 585, in main
    new_synthetic_prompts = generate_synthetic_prompts(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vanpelt/Development/lab/summary-finetune/scripts/data/generate_enhanced_dataset.py", line 263, in generate_synthetic_prompts
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 647, in __exit__
    self.shutdown(wait=True)
  File "/usr/lib/python3.12/concurrent/futures/thread.py", line 238, in shutdown
    t.join()
  File "/usr/lib/python3.12/threading.py", line 1147, in join
    self._wait_for_tstate_lock()
  File "/usr/lib/python3.12/threading.py", line 1167, in _wait_for_tstate_lock
    if lock.acquire(block, timeout):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
---
"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=25000,
                reasoning_effort="low",
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            text = response.choices[0].message.content.strip()
            parsed_response = json.loads(text)

            # Handle different response formats
            if isinstance(parsed_response, list):
                batch_prompts = parsed_response
            elif "requests" in parsed_response:
                batch_prompts = parsed_response["requests"]
            else:
                # Find the first list value
                for value in parsed_response.values():
                    if isinstance(value, list):
                        batch_prompts = value
                        break
                else:
                    batch_prompts = []

            return batch_prompts

        except Exception as e:
            console.print(f"[yellow]Batch {combo_idx + 1} ({category}/{language}/{app_type}) warning: {e}[/yellow]")
            return []

    # Generate batches in parallel
    synthetic_prompts = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Generating prompts...", total=num_batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch generation tasks
            future_to_combo = {
                executor.submit(generate_batch, i, cat, lang, app, size): (i, cat, lang, app)
                for i, (cat, lang, app, size) in enumerate(combinations)
            }

            # Collect results as they complete
            for future in as_completed(future_to_combo):
                combo_idx, category, language, app_type = future_to_combo[future]
                try:
                    batch_prompts = future.result()

                    # Clean and filter out duplicates
                    for p in batch_prompts:
                        # Clean the prompt first
                        cleaned_p = clean_prompt(p)
                        # Skip empty prompts after cleaning
                        if not cleaned_p:
                            continue
                        # Filter duplicates
                        if cleaned_p not in existing_set and cleaned_p not in synthetic_prompts:
                            synthetic_prompts.append(cleaned_p)
                            # Stop if we've reached our target
                            if len(synthetic_prompts) >= num_prompts:
                                break

                    progress.update(task, advance=1)

                except Exception as e:
                    console.print(f"[red]Error in batch {combo_idx + 1}: {e}[/red]")
                    progress.update(task, advance=1)

    # Trim to exact number requested
    actual_generated = len(synthetic_prompts)
    synthetic_prompts = synthetic_prompts[:num_prompts]

    console.print(f"[green]Generated {actual_generated} synthetic prompts (using {len(synthetic_prompts)})[/green]")
    return synthetic_prompts


def generate_json_response(
    client: openai.OpenAI,
    user_prompt: str
) -> str:
    """Generate JSON response for a user prompt using OpenAI with structured outputs"""

    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Request:\n{user_prompt}"}
            ],
            max_completion_tokens=712,
            reasoning_effort="low",
            response_format=CodeChangeResponse,
        )

        # Check for refusal
        if completion.choices[0].message.refusal:
            console.print(f"[yellow]Warning: Model refused to respond: {completion.choices[0].message.refusal}[/yellow]")
            raise ValueError(f"Model refusal: {completion.choices[0].message.refusal}")

        # Parse using SDK's native parsing
        parsed = completion.choices[0].message.parsed

        if not parsed:
            raise ValueError("No parsed content returned")

        # Convert to JSON string
        return json.dumps({
            "summary": parsed.summary,
            "branch": parsed.branch
        })

    except Exception as e:
        console.print(f"[red]Error generating response for prompt: {user_prompt[:100]}...[/red]")
        console.print(f"[red]Exception: {type(e).__name__}: {str(e)}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        # Fallback
        return json.dumps({
            "summary": "Update Code",
            "branch": "feat/update-code"
        })


def generate_all_responses(
    client: openai.OpenAI,
    prompts: List[str],
    preview: bool = False,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate JSON responses for all prompts using parallel API calls.

    Args:
        client: OpenAI client
        prompts: List of user prompts
        preview: If True, only generate first 5 examples
        max_workers: Number of parallel API requests (default: 10)
    """

    console.print(f"\n[cyan]Generating JSON responses for {len(prompts)} prompts with {max_workers} parallel workers...[/cyan]")

    examples = []
    preview_limit = 5 if preview else len(prompts)
    prompts_to_process = prompts[:preview_limit]

    def process_prompt(user_prompt: str) -> Dict[str, Any]:
        """Process a single prompt and return the example"""
        json_response = generate_json_response(client, user_prompt)

        # Create example in unsloth format (conversations with merged system prompt)
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nRequest:\n{user_prompt}"
                },
                {
                    "role": "assistant",
                    "content": json_response
                }
            ]
        }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Generating responses...", total=len(prompts_to_process))

        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(process_prompt, prompt): prompt
                for prompt in prompts_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_prompt):
                try:
                    example = future.result()
                    examples.append(example)
                    progress.update(task, advance=1)
                except Exception as e:
                    prompt = future_to_prompt[future]
                    console.print(f"[red]Error processing prompt: {prompt[:50]}...[/red]")
                    console.print(f"[red]Exception: {e}[/red]")
                    # Add fallback example
                    fallback_example = {
                        "conversations": [
                            {
                                "role": "user",
                                "content": f"{SYSTEM_PROMPT}\n\nRequest:\n{prompt}"
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps({
                                    "summary": "Update Code",
                                    "branch": "feat/update-code"
                                })
                            }
                        ]
                    }
                    examples.append(fallback_example)
                    progress.update(task, advance=1)

    console.print(f"[green]Generated {len(examples)} complete examples[/green]")
    return examples


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def preview_examples(examples: List[Dict[str, Any]], num_show: int = 10):
    """Display preview of examples"""
    console.print(f"\n[bold cyan]Preview of {min(num_show, len(examples))} Examples:[/bold cyan]\n")

    for i, example in enumerate(examples[:num_show], 1):
        conversations = example["conversations"]
        user_msg = conversations[0]["content"]
        assistant_msg = conversations[1]["content"]

        # Extract just the request part (after "Request:\n")
        request_part = user_msg.split("Request:\n", 1)[-1] if "Request:\n" in user_msg else user_msg
        original_length = len(request_part)

        # Truncate if too long
        if len(request_part) > 200:
            request_part = request_part[:200] + f"... ({original_length} chars)"
        else:
            request_part = request_part + f" ({original_length} chars)"

        console.print(f"[bold]Example {i}:[/bold]")
        console.print(f"[dim]Request:[/dim] {request_part}")
        console.print(f"[dim]Response:[/dim] {assistant_msg}")
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced dataset for Gemma3-270M fine-tuning"
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=2000,
        help="Number of synthetic prompts to generate (default: 2000)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of examples for test set (default: 200)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: only generate 5 synthetic examples and show them"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing prompts file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel API requests (default: 10)"
    )
    parser.add_argument(
        "--focus-long",
        action="store_true",
        help="Generate mostly long prompts (80%% long, 15%% medium, 5%% short)"
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
        console.print("[yellow]Add it to your .env file[/yellow]")
        return 1

    client = openai.OpenAI(api_key=api_key)

    console.print("[bold cyan]Enhanced Dataset Generation for Gemma3-270M[/bold cyan]\n")
    console.print(f"System Prompt: {SYSTEM_PROMPT[:100]}...\n")

    # Preview mode
    if args.preview:
        console.print("[yellow]PREVIEW MODE: Generating 5 synthetic examples only[/yellow]\n")
        args.num_synthetic = 5

    # Step 1: Load existing prompts
    console.print("[bold]Step 1: Load existing prompts[/bold]")
    existing_prompts = load_existing_prompts(args.data_dir)

    # Step 2: Generate or load synthetic prompts
    console.print(f"\n[bold]Step 2: Generate {args.num_synthetic} synthetic prompts[/bold]")

    prompts_file = args.output_dir / "synthetic_prompts.jsonl"
    train_file = args.output_dir / "train.jsonl"
    test_file = args.output_dir / "test.jsonl"

    # Load existing synthetic prompts if resuming
    existing_synthetic_prompts = []
    if args.resume and prompts_file.exists():
        console.print(f"[yellow]Resuming: loading existing synthetic prompts from {prompts_file}[/yellow]")
        with open(prompts_file) as f:
            existing_synthetic_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
        console.print(f"[green]Found {len(existing_synthetic_prompts)} existing synthetic prompts[/green]")

    # Load already-labeled examples to avoid re-labeling
    already_labeled_prompts = set()
    existing_train_data = []
    existing_test_data = []

    if args.resume:
        if train_file.exists():
            console.print(f"[yellow]Loading and cleaning existing train data from {train_file}[/yellow]")
            with open(train_file) as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # Extract and clean the request part from user message
                        user_content = example["conversations"][0]["content"]
                        if "Request:\n" in user_content:
                            request = user_content.split("Request:\n", 1)[1]
                            cleaned_request = clean_prompt(request)

                            # Update the example with cleaned request
                            example["conversations"][0]["content"] = f"{SYSTEM_PROMPT}\n\nRequest:\n{cleaned_request}"

                            existing_train_data.append(example)
                            already_labeled_prompts.add(cleaned_request)
            console.print(f"[green]Loaded and cleaned {len(existing_train_data)} existing train examples[/green]")

        if test_file.exists():
            console.print(f"[yellow]Loading and cleaning existing test data from {test_file}[/yellow]")
            with open(test_file) as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # Extract and clean the request part from user message
                        user_content = example["conversations"][0]["content"]
                        if "Request:\n" in user_content:
                            request = user_content.split("Request:\n", 1)[1]
                            cleaned_request = clean_prompt(request)

                            # Update the example with cleaned request
                            example["conversations"][0]["content"] = f"{SYSTEM_PROMPT}\n\nRequest:\n{cleaned_request}"

                            existing_test_data.append(example)
                            already_labeled_prompts.add(cleaned_request)
            console.print(f"[green]Loaded and cleaned {len(existing_test_data)} existing test examples[/green]")

        if already_labeled_prompts:
            console.print(f"[cyan]Total already-labeled prompts: {len(already_labeled_prompts)}[/cyan]")

    # Generate NEW synthetic prompts (not in existing synthetic or already labeled)
    all_existing = set(existing_prompts + existing_synthetic_prompts) | already_labeled_prompts

    console.print(f"[cyan]Generating {args.num_synthetic} NEW synthetic prompts...[/cyan]")
    new_synthetic_prompts = generate_synthetic_prompts(
        client,
        args.num_synthetic,
        existing_prompts=list(all_existing),
        max_workers=args.max_workers,
        focus_long=args.focus_long
    )
    console.print(f"[green]Generated {len(new_synthetic_prompts)} new synthetic prompts[/green]")

    # Append new prompts to synthetic_prompts.jsonl
    if new_synthetic_prompts:
        mode = 'a' if (args.resume and prompts_file.exists()) else 'w'
        with open(prompts_file, mode) as f:
            for prompt in new_synthetic_prompts:
                f.write(json.dumps({"prompt": prompt}) + '\n')
        console.print(f"[green]{'Appended' if mode == 'a' else 'Saved'} {len(new_synthetic_prompts)} prompts to {prompts_file}[/green]")

    # Step 3: Generate responses ONLY for new synthetic prompts
    console.print(f"\n[bold]Step 3: Generate JSON responses for NEW prompts[/bold]")

    if args.preview:
        # In preview mode, only generate responses for new synthetic prompts
        console.print(f"Preview mode: generating responses for {min(5, len(new_synthetic_prompts))} new synthetic prompts only")
        preview_examples_data = generate_all_responses(
            client,
            new_synthetic_prompts[:5],
            preview=True,
            max_workers=args.max_workers
        )
        preview_examples(preview_examples_data, num_show=len(preview_examples_data))
        console.print("\n[yellow]Preview complete! Run without --preview to generate full dataset.[/yellow]")
        return 0

    # Full mode: label ONLY new prompts (skip existing ones if resuming)
    if args.resume:
        # Only label the NEW synthetic prompts
        console.print(f"[cyan]Resume mode: labeling {len(new_synthetic_prompts)} new synthetic prompts only[/cyan]")
        prompts_to_label = new_synthetic_prompts
    else:
        # Fresh run: label existing + new synthetic
        console.print(f"[cyan]Fresh run: labeling all prompts (existing + synthetic)[/cyan]")
        prompts_to_label = existing_prompts + new_synthetic_prompts

    new_examples = generate_all_responses(
        client,
        prompts_to_label,
        preview=False,
        max_workers=args.max_workers
    )

    # Step 4: Merge with existing data and split
    console.print(f"\n[bold]Step 4: Merge and split into train/test[/bold]")

    if args.resume:
        # Merge new examples with existing train/test data
        all_examples = existing_train_data + existing_test_data + new_examples
        console.print(f"[cyan]Merged: {len(existing_train_data)} old train + {len(existing_test_data)} old test + {len(new_examples)} new = {len(all_examples)} total[/cyan]")
    else:
        all_examples = new_examples

    random.shuffle(all_examples)

    test_data = all_examples[:args.test_size]
    train_data = all_examples[args.test_size:]

    console.print(f"Train: {len(train_data)} examples")
    console.print(f"Test: {len(test_data)} examples")

    # Step 5: Save datasets (overwrite with merged data)
    console.print(f"\n[bold]Step 5: Save datasets[/bold]")

    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)

    console.print(f"[green]Saved train data to: {train_file}[/green]")
    console.print(f"[green]Saved test data to: {test_file}[/green]")

    # Show summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]Dataset Generation Complete![/bold green]")
    console.print("=" * 70)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Dataset", style="cyan")
    table.add_column("Examples", justify="right", style="green")
    table.add_column("File", style="dim")

    table.add_row("Train", str(len(train_data)), str(train_file))
    table.add_row("Test", str(len(test_data)), str(test_file))
    table.add_row("Total", str(len(all_examples)), "")

    console.print(table)

    # Show preview
    console.print(f"\n[bold cyan]Sample Training Examples:[/bold cyan]")
    preview_examples(train_data[:5], num_show=5)

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"1. Review the generated data in {args.output_dir}")
    console.print(f"2. Train with: uv run python scripts/training/train_unsloth_gemma3.py")
    console.print(f"   (Update the script to use {train_file} and {test_file})")

    return 0


if __name__ == "__main__":
    exit(main())
