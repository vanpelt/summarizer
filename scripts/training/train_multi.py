#!/usr/bin/env python3
"""
Multi-model training orchestration script.

Launches multiple Axolotl training jobs concurrently on the same GPU.
Monitors GPU memory usage and manages job lifecycle.
"""

import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import signal
import sys

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()

# Global process list for cleanup
processes: List[subprocess.Popen] = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    console.print("\n[yellow]Received interrupt signal. Stopping all training jobs...[/yellow]")
    for proc in processes:
        if proc.poll() is None:  # Still running
            proc.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_gpu_memory_usage() -> Optional[Dict[str, str]]:
    """Get current GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        used, total = result.stdout.strip().split(',')
        return {
            "used": f"{int(used)/1024:.1f}",
            "total": f"{int(total)/1024:.1f}",
            "percent": f"{(int(used)/int(total))*100:.1f}"
        }
    except Exception:
        return None


def create_status_table(jobs: List[Dict]) -> Table:
    """Create a status table for all training jobs."""
    table = Table(title="Multi-Model Training Status", show_header=True)

    table.add_column("Job", style="cyan")
    table.add_column("Config", style="blue")
    table.add_column("Output Dir", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("PID", style="yellow")

    for job in jobs:
        status = "Running" if job["process"].poll() is None else "Completed"
        status_color = "green" if status == "Running" else "dim"

        table.add_row(
            job["name"],
            job["config"].name,
            job["output_dir"].name if job["output_dir"] else "default",
            f"[{status_color}]{status}[/{status_color}]",
            str(job["process"].pid) if job["process"].poll() is None else "-"
        )

    return table


def launch_training_job(
    name: str,
    config_path: Path,
    output_dir: Optional[Path] = None,
    log_file: Optional[Path] = None
) -> subprocess.Popen:
    """Launch a single training job as a subprocess."""

    cmd = ["axolotl", "train", str(config_path)]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_dir", str(output_dir)])

    # Set up logging
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, 'w')
    else:
        log_handle = subprocess.DEVNULL

    console.print(f"[cyan]Launching job '{name}' with config {config_path.name}[/cyan]")

    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env={"CUDA_VISIBLE_DEVICES": "0"}
    )

    return process


def train_multiple_models(job_configs: List[Dict], monitor_interval: int = 30):
    """Launch and monitor multiple training jobs."""

    console.print(f"[bold cyan]Starting {len(job_configs)} training jobs[/bold cyan]\n")

    # Launch all jobs
    jobs = []
    for i, config in enumerate(job_configs, 1):
        name = config.get("name", f"job-{i}")
        config_path = config["config"]
        output_dir = config.get("output_dir")
        log_file = Path(f"logs/{name}.log")

        process = launch_training_job(name, config_path, output_dir, log_file)
        processes.append(process)

        jobs.append({
            "name": name,
            "config": config_path,
            "output_dir": output_dir,
            "process": process,
            "log_file": log_file
        })

        time.sleep(2)  # Small delay between launches

    console.print(f"\n[green]All {len(jobs)} jobs launched successfully[/green]\n")

    # Monitor jobs
    try:
        with Live(console=console, refresh_per_second=0.5) as live:
            while any(job["process"].poll() is None for job in jobs):
                # Update status table
                status_table = create_status_table(jobs)

                # Get GPU memory usage
                gpu_mem = get_gpu_memory_usage()
                if gpu_mem:
                    gpu_info = f"GPU Memory: {gpu_mem['used']}GB / {gpu_mem['total']}GB ({gpu_mem['percent']}%)"
                else:
                    gpu_info = "GPU Memory: unavailable"

                # Combine info
                display = Panel(
                    status_table,
                    subtitle=f"[dim]{gpu_info} | Refresh: {monitor_interval}s[/dim]",
                    border_style="cyan"
                )

                live.update(display)
                time.sleep(monitor_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping all jobs...[/yellow]")
        for job in jobs:
            if job["process"].poll() is None:
                job["process"].terminate()

    # Final status
    console.print("\n[bold]Final Status:[/bold]")
    completed = sum(1 for job in jobs if job["process"].poll() == 0)
    failed = sum(1 for job in jobs if job["process"].poll() not in [None, 0])

    console.print(f"  Completed: {completed}/{len(jobs)}")
    console.print(f"  Failed: {failed}/{len(jobs)}")

    # Print log file locations
    console.print("\n[cyan]Log files:[/cyan]")
    for job in jobs:
        console.print(f"  {job['name']}: {job['log_file']}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple Qwen3-8B models concurrently"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        help="List of config files to train"
    )
    parser.add_argument(
        "--preset",
        choices=["default", "max-concurrency"],
        help="Use preset configuration"
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=30,
        help="GPU monitoring interval in seconds (default: 30)"
    )

    args = parser.parse_args()

    # Set up job configurations
    job_configs = []

    if args.preset == "default":
        # Train 4 models with different configs
        configs_dir = Path("configs")
        job_configs = [
            {
                "name": "model-v1",
                "config": configs_dir / "qlora-8b.yml",
                "output_dir": Path("models/qwen3-8b-json-v1")
            },
            {
                "name": "model-v2",
                "config": configs_dir / "qlora-8b.yml",
                "output_dir": Path("models/qwen3-8b-json-v2")
            },
            {
                "name": "model-v3",
                "config": configs_dir / "qlora-8b-small.yml",
                "output_dir": Path("models/qwen3-8b-json-v3")
            },
            {
                "name": "model-v4",
                "config": configs_dir / "qlora-8b-small.yml",
                "output_dir": Path("models/qwen3-8b-json-v4")
            },
        ]
    elif args.preset == "max-concurrency":
        # Train 8 models with small config for maximum concurrency
        configs_dir = Path("configs")
        job_configs = [
            {
                "name": f"model-v{i}",
                "config": configs_dir / "qlora-8b-small.yml",
                "output_dir": Path(f"models/qwen3-8b-json-v{i}")
            }
            for i in range(1, 9)
        ]
    elif args.configs:
        # Use provided configs
        for i, config_path in enumerate(args.configs, 1):
            job_configs.append({
                "name": f"model-{i}",
                "config": config_path,
                "output_dir": None  # Use config default
            })
    else:
        console.print("[red]Error: Provide --configs or --preset[/red]")
        parser.print_help()
        sys.exit(1)

    # Validate configs exist
    for job in job_configs:
        if not job["config"].exists():
            console.print(f"[red]Error: Config file not found: {job['config']}[/red]")
            sys.exit(1)

    train_multiple_models(job_configs, args.monitor_interval)


if __name__ == "__main__":
    main()
