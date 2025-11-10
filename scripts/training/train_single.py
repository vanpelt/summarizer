#!/usr/bin/env python3
"""
Single model training script using Axolotl.

This is a simple wrapper around Axolotl CLI for training a single model.
For multiple concurrent training jobs, use train_multi.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def train_model(config_path: Path, output_dir: Path = None) -> int:
    """Train a single model using Axolotl CLI."""

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        return 1

    console.print(f"[cyan]Starting training with config: {config_path}[/cyan]")

    # Build Axolotl command
    cmd = ["axolotl", "train", str(config_path)]

    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])

    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        console.print("\n[green]Training completed successfully![/green]")
        return result.returncode
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Training failed with exit code {e.returncode}[/red]")
        return e.returncode
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Train a single Qwen3-8B model using Axolotl"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qlora-8b.yml"),
        help="Path to Axolotl config file (default: configs/qlora-8b.yml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config"
    )

    args = parser.parse_args()

    console.print("[bold cyan]Qwen3-8B Single Model Training[/bold cyan]\n")

    exit_code = train_model(args.config, args.output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
