#!/usr/bin/env python3
"""
Upload a GGUF model to HuggingFace Hub

Usage:
    uv run python scripts/export/upload_to_hf.py \
        --model-dir models/gguf/gemma3-270m-synthetic-v11 \
        --repo-id vanpelt/summarizer \
        --private \
        --wandb-run https://wandb.ai/wandb/summarizer/runs/abc123 \
        --exclude-safetensors
"""

import argparse
import os
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Upload GGUF model to HuggingFace Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to GGUF model directory (e.g., models/gguf/gemma3-270m-synthetic-v11)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., vanpelt/summarizer)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload GGUF model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default="",
        help="W&B run URL or ID to link in model card (e.g., https://wandb.ai/entity/project/runs/run_id)",
    )
    parser.add_argument(
        "--exclude-safetensors",
        action="store_true",
        help="Exclude model.safetensors file from upload (saves bandwidth, only upload GGUF)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Fine-tuned Gemma-3-270M for task summarization and branch naming",
        help="Model description for the model card",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="vanpelt",
        help="W&B entity/username (default: vanpelt)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="summarizer",
        help="W&B project name (default: summarizer)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1

    print(f"Uploading model from: {model_dir}")
    print(f"Repository: {args.repo_id}")
    print(f"Private: {args.private}")
    if args.exclude_safetensors:
        print(f"Excluding: model.safetensors")

    # Auto-detect W&B run if not provided
    wandb_run_url = args.wandb_run
    if not wandb_run_url:
        # Try to find W&B run by model name
        model_name = model_dir.name
        print(f"\nNo W&B run specified, searching for run with name '{model_name}'...")
        wandb_run_url = find_wandb_run(args.wandb_entity, args.wandb_project, model_name)
        if wandb_run_url:
            print(f"✓ Found W&B run: {wandb_run_url}")
        else:
            print(f"  No matching W&B run found")
    else:
        print(f"Using specified W&B run: {wandb_run_url}")

    # Initialize HF API
    api = HfApi()

    # Create repository if it doesn't exist
    print(f"\nCreating repository (if it doesn't exist)...")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return 1

    # Create model card with metadata
    print(f"\nGenerating model card...")
    model_card = generate_model_card(
        repo_id=args.repo_id,
        model_dir=model_dir,
        wandb_run=wandb_run_url,
        description=args.description,
    )

    # Write model card to repo
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card with metadata",
        )
        print(f"✓ Model card uploaded")
    except Exception as e:
        print(f"⚠ Warning: Could not upload model card: {e}")

    # Prepare ignore patterns
    ignore_patterns = []
    if args.exclude_safetensors:
        ignore_patterns.append("*.safetensors")

    # Upload all files from the model directory
    print(f"\nUploading files from {model_dir}...")
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )
        print(f"\n✓ Upload complete!")
        print(f"   View at: https://huggingface.co/{args.repo_id}")
        print(f"\n   Use with Ollama:")
        print(f"   ollama pull hf.co/{args.repo_id}")
    except Exception as e:
        print(f"✗ Error uploading: {e}")
        return 1

    return 0


def find_wandb_run(entity: str, project: str, run_name: str) -> str | None:
    """
    Search for a W&B run by displayName.

    Returns:
        str: W&B run URL if found, None otherwise
    """
    if not WANDB_AVAILABLE:
        return None

    try:
        api = wandb.Api()

        # Check for API key
        if api.api_key is None:
            return None

        # Search runs with displayName filter
        runs = api.runs(f"{entity}/{project}", filters={"displayName": run_name})

        # Convert to list to check if any runs were found
        runs_list = list(runs)

        if len(runs_list) > 0:
            return runs_list[0].url

        return None
    except Exception as e:
        print(f"  Warning: Could not query W&B API: {e}")
        return None


def generate_model_card(repo_id: str, model_dir: Path, wandb_run: str, description: str) -> str:
    """Generate a model card with metadata."""

    # Extract model name from repo_id
    model_name = repo_id.split("/")[-1]

    # Find GGUF file
    gguf_files = list(model_dir.glob("*.gguf"))
    gguf_file = gguf_files[0].name if gguf_files else "model.gguf"

    # Get file sizes
    files_info = []
    for f in model_dir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            size_mb = f.stat().st_size / (1024 * 1024)
            files_info.append(f"- `{f.name}` ({size_mb:.1f} MB)")

    # Build metadata
    metadata = {
        "language": ["en"],
        "license": "apache-2.0",
        "tags": ["text-generation", "gguf", "gemma", "summarization"],
        "base_model": "google/gemma-3-270m-it",
        "model_type": "gemma",
    }

    # Format YAML frontmatter
    yaml_lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            yaml_lines.append(f"{key}:")
            for item in value:
                yaml_lines.append(f"  - {item}")
        else:
            yaml_lines.append(f"{key}: {value}")
    yaml_lines.append("---")
    yaml_frontmatter = "\n".join(yaml_lines)

    # Build model card content
    card_parts = [yaml_frontmatter, ""]
    card_parts.append(f"# {model_name}")
    card_parts.append("")

    card_parts.append(description)
    card_parts.append("")

    card_parts.append("## Model Details")
    card_parts.append("")
    card_parts.append("- **Base Model**: google/gemma-3-270m-it")
    card_parts.append("- **Format**: GGUF (quantized for efficient inference)")
    card_parts.append("- **Quantization**: Q4_K_M")
    card_parts.append("- **Use Case**: Generating concise task titles and git branch names")
    card_parts.append("")

    if wandb_run:
        card_parts.append("## Training")
        card_parts.append("")
        card_parts.append(f"- **Training Run**: [{wandb_run}]({wandb_run})")
        card_parts.append("")

    card_parts.append("## Usage")
    card_parts.append("")
    card_parts.append("### With Ollama")
    card_parts.append("")
    card_parts.append("```bash")
    card_parts.append(f"ollama pull hf.co/{repo_id}")
    card_parts.append(f"ollama run hf.co/{repo_id}")
    card_parts.append("```")
    card_parts.append("")

    card_parts.append("### With llama.cpp")
    card_parts.append("")
    card_parts.append("```bash")
    card_parts.append(f"# Download the GGUF file")
    card_parts.append(f"huggingface-cli download {repo_id} {gguf_file}")
    card_parts.append("")
    card_parts.append("# Run with llama.cpp")
    card_parts.append(f"./main -m {gguf_file} -p 'Your prompt here'")
    card_parts.append("```")
    card_parts.append("")

    card_parts.append("## Files")
    card_parts.append("")
    card_parts.extend(files_info)
    card_parts.append("")

    return "\n".join(card_parts)


if __name__ == "__main__":
    exit(main())
