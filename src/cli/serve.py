#!/usr/bin/env python3
"""
CLI tool for starting vLLM server with trained adapters.

Usage:
    finetune-serve --discover
    finetune-serve --adapters model-v1 model-v2
    finetune-serve --api
"""

import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from ..inference.vllm_server import VLLMServer, discover_trained_models

console = Console()


@click.command()
@click.option(
    '--discover',
    is_flag=True,
    help='Auto-discover and load all trained adapters from models/ directory'
)
@click.option(
    '--adapters',
    multiple=True,
    help='Specific adapter names to load (can be used multiple times)'
)
@click.option(
    '--models-dir',
    type=click.Path(exists=True, path_type=Path),
    default=Path('models'),
    help='Directory containing trained models (default: models/)'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Port to run vLLM server on (default: 8000)'
)
@click.option(
    '--api',
    is_flag=True,
    help='Also start FastAPI wrapper on port 8080'
)
@click.option(
    '--api-port',
    default=8080,
    type=int,
    help='Port for FastAPI server (default: 8080)'
)
@click.option(
    '--base-model',
    default='Qwen/Qwen3-8B',
    help='Base model to use (default: Qwen/Qwen3-8B)'
)
@click.option(
    '--gpu-memory',
    default=0.9,
    type=float,
    help='GPU memory utilization (default: 0.9)'
)
def cli(
    discover: bool,
    adapters: tuple,
    models_dir: Path,
    port: int,
    api: bool,
    api_port: int,
    base_model: str,
    gpu_memory: float,
):
    """
    Start vLLM inference server with trained LoRA adapters.

    \b
    Examples:
        # Auto-discover and load all models
        finetune-serve --discover

        # Load specific models
        finetune-serve --adapters model-v1 --adapters model-v2

        # Start with FastAPI wrapper
        finetune-serve --discover --api
    """
    console.print("[bold cyan]Qwen3-8B Inference Server[/bold cyan]\n")

    # Determine which adapters to load
    adapter_paths = {}

    if discover:
        console.print(f"[cyan]Discovering models in {models_dir}...[/cyan]")
        adapter_paths = discover_trained_models(models_dir)

        if not adapter_paths:
            console.print(f"[yellow]No trained models found in {models_dir}[/yellow]")
            console.print("[yellow]Train a model first or specify --adapters manually[/yellow]")
            sys.exit(1)

    elif adapters:
        for adapter_name in adapters:
            adapter_path = models_dir / adapter_name
            if not adapter_path.exists():
                console.print(f"[red]Error: Adapter not found: {adapter_path}[/red]")
                sys.exit(1)
            adapter_paths[adapter_name] = adapter_path

    # Show loaded adapters
    if adapter_paths:
        table = Table(title="Loaded Adapters", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="blue")

        for name, path in adapter_paths.items():
            table.add_row(name, str(path))

        console.print(table)
        console.print()

    # Create and start server
    try:
        server = VLLMServer(
            base_model=base_model,
            port=port,
            gpu_memory_utilization=gpu_memory,
        )

        # Add adapters
        for name, path in adapter_paths.items():
            server.add_lora_adapter(name, path)

        # Start vLLM server
        server.start(background=False if not api else True)

        if api:
            # Wait for vLLM to be ready
            if not server.wait_for_ready():
                console.print("[red]vLLM server failed to start[/red]")
                sys.exit(1)

            # Start FastAPI server
            console.print(f"\n[cyan]Starting FastAPI server on port {api_port}...[/cyan]")
            import uvicorn
            uvicorn.run(
                "src.api.main:app",
                host="0.0.0.0",
                port=api_port,
                log_level="info",
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if 'server' in locals():
            server.stop()
        sys.exit(0)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
