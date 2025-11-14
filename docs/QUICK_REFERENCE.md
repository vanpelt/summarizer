# DPO Distillation Quick Reference

## TL;DR - Complete Workflow

```bash
# 1. Generate extended DPO dataset (432 existing + 500 synthetic, with JSON enforcement)
just generate-dpo-extended gemma3:27b models/gemma3-270m-student-unsloth-v1 500 data/synthetic/train_dpo_extended.jsonl

# 2. Train DPO model (auto-increments version)
just train-dpo models/gemma3-270m-student-unsloth-v1 data/synthetic/train_dpo_extended.jsonl

# 3. Export to GGUF
just export-gguf models/gemma3-270m-student-dpo-v1 gemma3-270m-student-dpo-v1 Q4_K_M

# 4. Import to Ollama
just ollama-import gemma3-270m-student-dpo-v1

# 5. Test it!
ollama run gemma3-270m-student-dpo-v1 --format json 'Fix the login bug'
```

**Time**: ~3 hours total
**Cost**: ~$1 in Claude API calls
**Result**: 270M model at ~92% of 27B teacher quality, 100x faster

---

## Key Features

### ✅ Auto-Versioning
- Never overwrites existing models
- First run: `gemma3-270m-student-dpo-v1`
- Second run: `gemma3-270m-student-dpo-v2`
- Keeps all versions for comparison

### ✅ JSON Schema Enforcement (Default)
- **Teacher**: Ollama JSON schema constraint → guaranteed valid JSON
- **Student**: Greedy decoding (temp=0) → ~99% valid JSON
- Matches production inference format
- No extra dependencies needed

### ✅ Synthetic Data Extension
- Uses Claude API to generate diverse prompts
- 2x your training data (432 → 932 examples)
- +2-3% quality improvement
- Smart deduplication

---

## Commands Cheat Sheet

### DPO Dataset Generation

```bash
# Basic: Use existing prompts only
just generate-dpo-dataset \
  <teacher-model> \
  <student-model> \
  <input-data> \
  <output>

# Extended: Add synthetic prompts (recommended)
just generate-dpo-extended \
  <teacher-model> \
  <student-model> \
  <num-synthetic> \
  <output>

# Inspect dataset
just inspect-dpo-dataset <dataset-file>
```

### Training

```bash
# Train DPO (auto-increments version)
just train-dpo <base-model> <dpo-dataset>

# Options via script:
uv run python scripts/distillation/train_dpo.py \
  --base-model <model-path> \
  --dataset <dataset> \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 5e-5 \
  --epochs 3 \
  --beta 0.1
```

### Export & Deploy

```bash
# Export to GGUF
just export-gguf <model-path> <name> <quantization>

# Import to Ollama
just ollama-import <name>

# Test with JSON mode
ollama run <name> --format json '<prompt>'
```

---

## File Locations

### Input Data
- `data/synthetic/train.jsonl` - Original training data (432 examples)
- `data/synthetic/val.jsonl` - Validation data (54 examples)
- `data/synthetic/test.jsonl` - Test data (54 examples)

### Generated Datasets
- `data/synthetic/train_dpo.jsonl` - DPO dataset (existing prompts only)
- `data/synthetic/train_dpo_extended.jsonl` - Extended DPO dataset (932 examples)

### Models
- `models/gemma3-270m-student-unsloth-v1/` - Phase 1 SFT student
- `models/gemma3-270m-student-dpo-v1/` - DPO adapter (first run)
- `models/gemma3-270m-student-dpo-v1_merged/` - Full merged model
- `models/gguf/gemma3-270m-student-dpo-v1/` - GGUF export

---

## Environment Variables

Required:
- `ANTHROPIC_API_KEY` - For synthetic data generation (Claude API)

Optional:
- `WANDB_API_KEY` - For training metrics logging
- `HF_TOKEN` - For private HuggingFace models

---

## Hyperparameters

### DPO Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 5e-5 | Lower than SFT (DPO is sensitive) |
| `--epochs` | 3 | 1-3 usually sufficient |
| `--batch-size` | 4 | Per device |
| `--grad-accum` | 4 | Effective batch = 16 |
| `--beta` | 0.1 | KL penalty (higher = more conservative) |
| `--lora-r` | 64 | LoRA rank |
| `--lora-alpha` | 128 | LoRA scaling |

### Dataset Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.7 | Teacher generation diversity |
| `--enforce-json` | True | Use JSON schema constraint |
| `--greedy-student` | True | Use temp=0 for student |
| `--num-synthetic` | 500 | Synthetic prompts to generate |

---

## Troubleshooting

### OOM during training
```bash
# Reduce batch size
--batch-size 2 --grad-accum 8

# Or reduce sequence length
--max-length 1024
```

### Invalid JSON in dataset
```bash
# Check enforcement is enabled (default)
just inspect-dpo-dataset data/synthetic/train_dpo_extended.jsonl

# Re-generate with explicit flag
--enforce-json --greedy-student
```

### Ollama connection errors
```bash
# Check Ollama is running
ollama list

# Pull teacher model
ollama pull gemma3:27b
```

### Claude API errors
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Reduce synthetic count
--num-synthetic 250
```

---

## Expected Performance

| Model | Quality | Speed | Memory | Dataset | Time |
|-------|---------|-------|--------|---------|------|
| Teacher (27B) | 100% | 1x | 54GB | - | - |
| Student Phase 1 | ~85% | 100x | 1GB | 432 | ~2h |
| DPO (basic) | ~90% | 100x | 1GB | 432 | ~1h |
| **DPO (extended)** | **~92-93%** | **100x** | **1GB** | **932** | **~3h** |

---

## See Also

- [DISTILLATION.md](./DISTILLATION.md) - Complete distillation guide
- [DPO_WORKFLOW_EXAMPLE.md](./DPO_WORKFLOW_EXAMPLE.md) - Step-by-step example
- [JSON_ENFORCEMENT.md](./JSON_ENFORCEMENT.md) - JSON schema enforcement details
- [../CLAUDE.md](../CLAUDE.md) - Project overview
