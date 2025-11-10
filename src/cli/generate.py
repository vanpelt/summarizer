#!/usr/bin/env python3
"""
CLI tool for generating title and branch names.

Usage:
    finetune-generate "Add user authentication"
    finetune-generate "Fix memory leak" --model model-v2
"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..inference.client import InferenceClient

console = Console()


@click.command()
@click.argument('prompt', type=str)
@click.option(
    '--model',
    default='default',
    help='Model adapter to use (default: default)'
)
@click.option(
    '--server',
    default='http://localhost:8000',
    help='vLLM server URL (default: http://localhost:8000)'
)
@click.option(
    '--temperature',
    default=0.7,
    type=float,
    help='Sampling temperature (default: 0.7)'
)
@click.option(
    '--raw',
    is_flag=True,
    help='Output raw JSON without formatting'
)
@click.option(
    '--no-constrained',
    is_flag=True,
    help='Disable constrained decoding'
)
def cli(
    prompt: str,
    model: str,
    server: str,
    temperature: float,
    raw: bool,
    no_constrained: bool
):
    """
    Generate title and branch name from a task description.

    \b
    Examples:
        finetune-generate "Add dark mode toggle"
        finetune-generate "Fix login bug" --model model-v2
        finetune-generate "Update API docs" --temperature 0.3
    """
    try:
        with InferenceClient(base_url=server) as client:
            # Check server health
            if not client.health_check():
                console.print("[red]Error: vLLM server is not available[/red]")
                console.print(f"[yellow]Make sure the server is running at {server}[/yellow]")
                sys.exit(1)

            # Generate
            console.print(f"[cyan]Generating for:[/cyan] {prompt}")

            result = client.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                use_constrained_decoding=not no_constrained,
            )

            # Output
            if raw:
                console.print(f'{{"title": "{result.title}", "branch_name": "{result.branch_name}"}}')
            else:
                # Pretty output
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Title", result.title)
                table.add_row("Branch Name", result.branch_name)
                table.add_row("Model", model)

                panel = Panel(
                    table,
                    title="[bold]Generated Output[/bold]",
                    border_style="green"
                )
                console.print(panel)

                # Show git command
                console.print(f"\n[dim]Git command:[/dim]")
                console.print(f"  git checkout -b {result.branch_name}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
