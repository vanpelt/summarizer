# Gemma3-270M Distillation

Knowledge distillation from larger models (Gemma3-27B, GPT-4o-mini) to Gemma3-270M for efficient JSON generation. Uses supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) for task titles and git branch names.

## Features

- **Two-Phase Distillation**: SFT + DPO training using Unsloth
- **Synthetic Data**: Generate training data from larger teacher models
- **Efficient**: 270M parameter student model runs on modest hardware
- **GGUF Export**: Deploy to Ollama for CPU/GPU inference
- **Docker-based**: Reproducible training environment

## Quick Start

See [CLAUDE.md](CLAUDE.md) for detailed documentation.

### Phase 1: Supervised Fine-tuning (SFT)

```bash
# Train student model on teacher outputs
just unsloth-train
```

### Phase 2 (optional): Direct Preference Optimization (DPO)

```bash
# Generate preference dataset (teacher vs student outputs)
just generate-dpo-extended

# Split dataset for training and evaluation
DATASET=data/synthetic/train_dpo_extended_split.jsonl \
VAL_DATASET=data/synthetic/val_dpo_extended.jsonl \
just train-dpo
```

### Export and Evaluate

```bash
# Export GGUF and import to Ollama
just export-gguf models/gemma3-270m-synthetic-v10
just ollama-import gemma3-270m-synthetic-v10 gemma3-summary:v10

# Smoke Test
ollama smoke-test gemma3-summary:v10

# Weave Evaluation
uv run scripts/evaluate_summarizer.py --models gemma3-summary:v10 -t lora_r:64 -t epochs:2 -t lr:2e4
```

## Project Structure

```
summary-finetune/
├── data/
│   └── synthetic/         # Training datasets (SFT and DPO)
├── models/                # Trained checkpoints
│   └── gemma3-270m-*/     # Student model versions
├── scripts/
│   ├── distillation/      # DPO dataset generation and training
│   ├── training/          # Unsloth SFT training
│   └── export/            # GGUF export
|   evaluate_summarizer.py # Weave evaluation
├── justfile               # Task runner commands
├── CLAUDE.md              # Detailed documentation
└── Dockerfile.unsloth     # Training environment
```

## Key Commands

```bash
# Build Docker image
just build-unsloth

# Train student model (Phase 1: SFT)
just unsloth-train

# Generate DPO dataset with synthetic prompts (Phase 2)
just generate-dpo-extended

# Train with DPO (Phase 2)
just train-dpo

# Export to GGUF
just export-gguf <model-path> <output-name>

# Import to Ollama
just ollama-import <model-name>

# Interactive shell
just unsloth-shell
```

## Environment Variables

```bash
# Required for synthetic data generation
ANTHROPIC_API_KEY=<your-key>    # For Claude-based generation
OPENAI_API_KEY=<your-key>       # For GPT-based generation

# Optional
WANDB_API_KEY=<your-key>        # For training metrics
HF_TOKEN=<your-key>             # For private HuggingFace models
CUDA_VISIBLE_DEVICES=0          # GPU selection
```

## Model Versions

| Model | Training | Description |
|-------|----------|-------------|
| `gemma3-270m-student-unsloth-v1` | SFT | Initial distillation from teacher |
| `gemma3-270m-student-dpo-v*` | DPO | Refined with preference learning |

## Training Configuration

**DPO Training** (default):
- Batch size: 4
- Gradient accumulation: 4
- Learning rate: 5e-5
- Epochs: 3
- Beta (KL penalty): 0.1
- LoRA rank: 64

Customize via environment variables:
```bash
BATCH_SIZE=8 LR=1e-4 EPOCHS=5 just train-dpo
```

## Hardware Requirements

- **Training**: NVIDIA GPU with 40GB+ VRAM (tested on DGX Spark GH200)
- **Inference**: CPU or GPU (via Ollama)
- **Export**: GPU for GGUF conversion

## Performance

- **Model size**: 270M parameters
- **Training time**: ~2-3 hours per phase on GH200
- **Inference**: Fast on CPU with GGUF quantization

## Documentation

- [CLAUDE.md](CLAUDE.md) - Complete documentation for development
- [justfile](justfile) - All available commands

## License

MIT
