# CLAUDE.md - Project Documentation

## Project Overview

This is a **Qwen3-8B fine-tuning project** for generating JSON outputs (task titles and git branch names) from arbitrary user prompts. The project uses QLoRA for memory-efficient fine-tuning and vLLM for high-performance inference.

### Key Technologies
- **Base Model**: Qwen3-8B
- **Training**: QLoRA (4-bit quantization + LoRA adapters) via Axolotl
- **Inference**: vLLM with multi-adapter support
- **Package Manager**: uv (fast Python package manager)
- **Task Runner**: just (modern make alternative)
- **Hardware**: NVIDIA DGX Spark (ARM64 Grace Hopper architecture)

## Hardware Environment: DGX Spark

This project is designed to run on the **NVIDIA DGX Spark**, which features:
- **Architecture**: ARM64 (aarch64) with NVIDIA Grace Hopper (GH200)
- **GPU**: 128GB HBM3 memory
- **CUDA Compute Capability**: 12.0f (Spark architecture)
- **Platform**: Linux on ARM64

### Why This Matters

The ARM64 + Grace Hopper architecture requires special build considerations:

1. **PyTorch**: Must use nightly builds (PyTorch 2.10+) for ARM64 + CUDA support
2. **vLLM**: Must be built from source with specific build flags
3. **CUDA**: Requires CUDA 12.x with ARM64 support
4. **Build Tools**: Need to specify CUDA architecture list and PTXAS path

## vLLM Setup on DGX Spark

### The Challenge

vLLM does not provide pre-compiled wheels for ARM64 + CUDA. The package must be built from source with specific environment variables to target the Grace Hopper architecture correctly.

### The Solution: fix.sh

We created a `fix.sh` wrapper script that sets all required environment variables:

```bash
#!/bin/bash
export TORCH_CUDA_ARCH_LIST=12.0f        # Grace Hopper compute capability
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  # CUDA assembly compiler
export VLLM_TARGET_DEVICE=cuda           # Build for CUDA (not CPU/ROCm)
export VLLM_USE_PRECOMPILED=0            # Force source build
export MAX_JOBS=16                       # Parallel build jobs

exec "$@"  # Run the command with these env vars
```

### Initial Setup

To set up the environment initially:

```bash
# Clone vLLM source code adjacent to this project
cd /home/vanpelt/Development/lab
git clone https://github.com/vllm-project/vllm.git

# Build and install vLLM with proper environment
cd summary-finetune
./fix.sh uv sync
```

This will:
1. Install PyTorch nightly for ARM64 + CUDA (via uv.toml config)
2. Build vLLM from source with Grace Hopper support
3. Install all other project dependencies

**Note**: The initial build takes 15-30 minutes. Be patient!

### Why uv?

We use **uv** (https://github.com/astral-sh/uv) instead of pip because:
- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Modern**: Built in Rust, designed for modern Python workflows
- **ARM64 Support**: Handles architecture-specific packages elegantly

The `pyproject.toml` configures uv to:
- Use PyTorch nightly for ARM64 (`platform_machine == 'aarch64'`)
- Build vLLM from local source (`path = "../vllm"`)
- Override PyTorch version constraints
- Disable build isolation for vLLM

#### UV and Virtual Environments

If uv warns about refusing `VIRTUAL_ENV` and suggests using `--active`, add this to your `.env`:

```bash
# Make uv respect active virtual environment by default
UV_PROJECT_ENVIRONMENT=.venv
```

This tells uv to use the `.venv` directory as the project environment, eliminating the need for `--active` flag on every command.

## Project Structure

```
summary-finetune/
├── fix.sh                    # Environment setup script for DGX Spark
├── justfile                  # Task runner commands
├── pyproject.toml            # Python project + uv configuration
├── configs/                  # Training configurations
│   ├── qlora-8b.yml         # Standard QLoRA config
│   └── qlora-8b-small.yml   # Memory-efficient config
├── data/                     # Training data
│   ├── processed/           # Formatted JSONL datasets
│   └── synthetic/           # Generated synthetic data
├── models/                   # Trained LoRA adapters
├── scripts/
│   ├── data/                # Data generation scripts
│   └── training/            # Training orchestration
├── src/
│   ├── api/                 # FastAPI application
│   ├── cli/                 # Command-line tools
│   └── inference/           # vLLM client/server wrappers
└── tests/                   # Unit tests
```

## Common Tasks

### Using the justfile

The `justfile` provides convenient commands for common tasks:

```bash
# See all available commands
just

# Download the base model
just download-model

# Check if model is already downloaded
just check-model

# Start vLLM server
just serve

# Start with custom port
just serve "Qwen/Qwen3-8B" 8080

# Start with LoRA adapters
just serve-lora "Qwen/Qwen3-8B" 8000 "model-v1,model-v2"

# Test the server
just test-server

# Generate synthetic training data
just generate-data 1000

# Train a model
just train

# Format code
just format

# Run tests
just test

# Monitor GPU
just gpu
```

### Running vLLM with fix.sh

Any vLLM-related command should be wrapped with `./fix.sh`:

```bash
# Good - uses fix.sh
./fix.sh uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B

# Bad - missing environment variables, will fail or use CPU
uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B
```

The `justfile` automatically wraps vLLM commands with `./fix.sh`.

## Environment Variables

### Required for vLLM (set by fix.sh)
- `TORCH_CUDA_ARCH_LIST=12.0f` - Target Grace Hopper architecture
- `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` - CUDA assembly path
- `VLLM_TARGET_DEVICE=cuda` - Build for CUDA backend
- `VLLM_USE_PRECOMPILED=0` - Force source build
- `MAX_JOBS=16` - Parallel compilation jobs

### Optional (for application)
- `ANTHROPIC_API_KEY` - For synthetic data generation with Claude
- `OPENAI_API_KEY` - For OpenAI-based data generation
- `WANDB_API_KEY` - For training metrics logging
- `HF_TOKEN` - For private HuggingFace models

Create a `.env` file from `.env.example` and add these keys.

## Training

### Memory Usage

On DGX Spark (128GB GPU), you can train multiple models concurrently:

| Config | Memory/Model | Max Concurrent |
|--------|--------------|----------------|
| `qlora-8b.yml` | 8-10 GB | 11-12 models |
| `qlora-8b-small.yml` | 6-8 GB | 14-16 models |

### Training Workflow

1. **Generate synthetic data**:
   ```bash
   just generate-data 1000
   ```

2. **Train a single model**:
   ```bash
   just train configs/qlora-8b.yml
   ```

3. **Train multiple models** (for experimentation):
   ```bash
   uv run python scripts/training/train_multi.py --preset max-concurrency
   ```

4. **Monitor GPU usage**:
   ```bash
   just gpu
   # or
   watch -n 1 nvidia-smi
   ```

## Inference

### Start the Server

```bash
# Simple: just the base model
just serve

# With trained LoRA adapters
just serve-lora "Qwen/Qwen3-8B" 8000 "model-v1,model-v2"
```

### Test the Server

```bash
# Quick test
just test-server

# Manual test with curl
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "Generate a title and branch name for: Add user authentication",
    "max_tokens": 100
  }'
```

### Using the CLI

```bash
# Generate with CLI
finetune-generate "Add dark mode toggle"

# Use specific adapter
finetune-generate "Fix memory leak" --model model-v2

# Raw JSON output
finetune-generate "Update API docs" --raw
```

## Development Workflow

### Initial Setup

```bash
# 1. Clone and setup
cd /home/vanpelt/Development/lab/summary-finetune
just setup

# 2. Download base model (if not cached)
just download-model

# 3. Configure environment
cp .env.example .env
# Edit .env and add API keys
```

### Making Changes

```bash
# 1. Make code changes
vim src/...

# 2. Format and lint
just format
just lint

# 3. Run tests
just test

# 4. Test inference
just serve
# In another terminal:
just test-server
```

### Updating vLLM

If vLLM needs to be updated:

```bash
cd ../vllm
git pull
cd ../summary-finetune
./fix.sh uv sync  # Rebuilds vLLM
```

## Troubleshooting

### vLLM Build Fails

**Symptoms**: Build errors during `uv sync`, CUDA errors, or architecture mismatches

**Solutions**:
1. Verify CUDA toolkit: `nvcc --version` (should be 12.x)
2. Check environment: `./fix.sh env | grep -E 'TORCH|VLLM|TRITON'`
3. Clean build: `cd ../vllm && git clean -fdx && cd ../summary-finetune && ./fix.sh uv sync`
4. Check PyTorch: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### vLLM Server Won't Start

**Symptoms**: Server crashes, can't find CUDA, or uses CPU

**Solutions**:
1. Always use `./fix.sh` or `just serve`
2. Check GPU: `nvidia-smi`
3. Verify model downloaded: `just check-model`
4. Check logs for detailed errors
5. Try reducing GPU memory: `just serve "Qwen/Qwen3-8B" 8000` (default is 90% utilization)

### Import Errors

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. Ensure virtual env activated: `source .venv/bin/activate`
2. Or use uv: `uv run python your_script.py`
3. Reinstall dependencies: `./fix.sh uv sync`

### Out of Memory

**Symptoms**: CUDA OOM errors during training or inference

**Solutions**:
- **Training**: Use `qlora-8b-small.yml`, reduce `micro_batch_size`, or train fewer models concurrently
- **Inference**: Reduce `--gpu-memory-utilization`, use smaller `--max-model-len`, or serve fewer adapters

### Slow Performance

**Symptoms**: Training or inference is slower than expected

**Check**:
1. GPU utilization: `nvidia-smi dmon -s u`
2. vLLM is using GPU: Look for CUDA messages in server logs
3. Not using precompiled wrong arch: `./fix.sh` should be used
4. Batch sizes and concurrent requests are optimized

## Performance Notes

### Training Performance (DGX Spark)
- **Single model**: ~2-3 hours for 3 epochs (1000 examples), 8-10GB VRAM
- **Concurrent training**: 8-12 models simultaneously on 128GB GPU
- **Throughput**: ~100-150 examples/sec per job

### Inference Performance (vLLM + DGX Spark)
- **Base model + 8 adapters**: ~18 GB VRAM
- **Remaining KV cache**: ~100 GB
- **Throughput**: 1000-2000 tokens/sec
- **Concurrent requests**: 50-100+ (with vLLM's paged attention)

## Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [uv Documentation](https://github.com/astral-sh/uv)
- [just Documentation](https://just.systems/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-8B)

### Related Projects
- vLLM: https://github.com/vllm-project/vllm
- Axolotl: https://github.com/OpenAccess-AI-Collective/axolotl
- uv: https://github.com/astral-sh/uv
- just: https://github.com/casey/just

## Notes for Claude

When working with this project:
1. **Always use `./fix.sh`** when running vLLM-related commands
2. **Use `just` commands** for common tasks (see `justfile`)
3. **Remember the architecture**: ARM64 + Grace Hopper requires special handling
4. **uv is preferred** over pip for all Python package operations
5. **Model path**: `Qwen/Qwen3-8B`
6. **Training data format**: JSONL with prompt/completion pairs
7. **LoRA adapters**: Stored in `models/` directory
8. **vLLM source**: Must be in `../vllm` directory (sibling to this project)

## Future Work

- [ ] Add evaluation metrics and benchmarking
- [ ] Support for other base models (Llama, Mistral)
- [ ] Docker containerization (with ARM64 support)
- [ ] CI/CD pipeline for testing
- [ ] Web UI for generation
- [ ] Fine-tuning on custom datasets
- [ ] Automated hyperparameter tuning
- [ ] Multi-GPU training support
