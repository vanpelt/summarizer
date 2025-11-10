"""
vLLM server wrapper for serving multiple LoRA adapters.

Provides utilities for starting and managing vLLM server with multi-LoRA support.
"""

import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict
import requests

from rich.console import Console

console = Console()


class VLLMServer:
    """Wrapper for vLLM server with multi-LoRA support."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-8B",
        host: str = "0.0.0.0",
        port: int = 8000,
        gpu_memory_utilization: float = 0.9,
        max_loras: int = 8,
        max_lora_rank: int = 32,
    ):
        self.base_model = base_model
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.process: Optional[subprocess.Popen] = None
        self.lora_modules: Dict[str, Path] = {}

    def add_lora_adapter(self, name: str, adapter_path: Path):
        """Add a LoRA adapter to be loaded at startup."""
        if not adapter_path.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_path}")
        self.lora_modules[name] = adapter_path

    def start(self, background: bool = True) -> subprocess.Popen:
        """Start the vLLM server."""

        cmd = [
            "vllm", "serve", self.base_model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        # Add LoRA configuration if adapters are specified
        if self.lora_modules:
            cmd.extend([
                "--enable-lora",
                "--max-loras", str(self.max_loras),
                "--max-lora-rank", str(self.max_lora_rank),
            ])

            # Add lora modules
            for name, path in self.lora_modules.items():
                cmd.extend(["--lora-modules", f"{name}={path}"])

        console.print(f"[cyan]Starting vLLM server on {self.host}:{self.port}[/cyan]")
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

        if self.lora_modules:
            console.print(f"[cyan]Loading {len(self.lora_modules)} LoRA adapters:[/cyan]")
            for name, path in self.lora_modules.items():
                console.print(f"  - {name}: {path}")

        if background:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            console.print(f"[green]Server started in background (PID: {self.process.pid})[/green]")
        else:
            self.process = subprocess.Popen(cmd)
            self.process.wait()

        return self.process

    def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for the server to be ready."""
        console.print("[cyan]Waiting for server to be ready...[/cyan]")

        url = f"http://{self.host}:{self.port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    console.print("[green]Server is ready![/green]")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(5)

        console.print("[red]Server failed to start within timeout[/red]")
        return False

    def stop(self):
        """Stop the vLLM server."""
        if self.process and self.process.poll() is None:
            console.print("[yellow]Stopping vLLM server...[/yellow]")
            self.process.terminate()
            self.process.wait(timeout=10)
            console.print("[green]Server stopped[/green]")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.process is not None and self.process.poll() is None

    def get_url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"


def start_vllm_with_adapters(
    adapter_paths: Dict[str, Path],
    base_model: str = "Qwen/Qwen3-8B",
    port: int = 8000,
    background: bool = True
) -> VLLMServer:
    """
    Convenience function to start vLLM server with adapters.

    Args:
        adapter_paths: Dictionary mapping adapter names to paths
        base_model: Base model to use
        port: Port to run server on
        background: Run in background or foreground

    Returns:
        VLLMServer instance
    """
    server = VLLMServer(base_model=base_model, port=port)

    for name, path in adapter_paths.items():
        server.add_lora_adapter(name, path)

    server.start(background=background)

    if background:
        server.wait_for_ready()

    return server


def discover_trained_models(models_dir: Path = Path("models")) -> Dict[str, Path]:
    """
    Discover trained LoRA adapters in the models directory.

    Returns:
        Dictionary mapping model names to adapter paths
    """
    adapters = {}

    if not models_dir.exists():
        return adapters

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Check for adapter files
            adapter_files = list(model_dir.glob("adapter_model.*"))
            if adapter_files:
                adapters[model_dir.name] = model_dir

    return adapters
