# Complete DPO Distillation Workflow

This guide shows the complete workflow for Phase 2 DPO distillation with synthetic data extension.

## Prerequisites

1. **Phase 1 student model** trained: `models/gemma3-270m-student-unsloth-v1`
2. **Ollama** running with teacher model: `ollama pull gemma3:27b`
3. **ANTHROPIC_API_KEY** in your `.env` file (for synthetic data generation)

## Workflow

### Step 1: Generate Extended DPO Dataset (Recommended)

Generate ~932 examples (432 existing + 500 synthetic):

```bash
just generate-dpo-extended \
  gemma3:27b \
  models/gemma3-270m-student-unsloth-v1 \
  500 \
  data/gpt5nano/train_dpo_extended.jsonl
```

**Time**: ~30-40 minutes
**Output**: `data/gpt5nano/train_dpo_extended.jsonl`

This will:
1. Load your 432 existing training prompts
2. Generate 500 new diverse synthetic prompts using Claude
3. Run both teacher (27B) and student (270M) on all 932 prompts
4. Create preference pairs (teacher=chosen, student=rejected)

**Alternative (faster, less data):**
If you want to skip synthetic generation, use only existing data:

```bash
just generate-dpo-dataset \
  gemma3:27b \
  models/gemma3-270m-student-unsloth-v1 \
  data/gpt5nano/train.jsonl \
  data/gpt5nano/train_dpo.jsonl
```

**Time**: ~10-15 minutes
**Output**: `data/gpt5nano/train_dpo.jsonl` (432 examples)

### Step 2: Train DPO Model

Train the student model to prefer teacher-like outputs:

```bash
just train-dpo \
  models/gemma3-270m-student-unsloth-v1 \
  data/gpt5nano/train_dpo_extended.jsonl
```

**Time**: ~45-90 minutes (3 epochs)
**Output**:
- `models/gemma3-270m-student-dpo-v1/` (LoRA adapter)
- `models/gemma3-270m-student-dpo-v1_merged/` (full merged model)

**Auto-versioning**: If you run this again, it will create `v2`, `v3`, etc.

### Step 3: Export to GGUF

Export for Ollama deployment:

```bash
just export-gguf \
  models/gemma3-270m-student-dpo-v1 \
  gemma3-270m-student-dpo-v1 \
  Q4_K_M
```

**Time**: ~5-10 minutes
**Output**: `models/gguf/gemma3-270m-student-dpo-v1/`
  - `gemma3-270m-student-dpo-v1-Q4_K_M-unsloth.gguf`
  - `Modelfile`

### Step 4: Import to Ollama

```bash
just ollama-import gemma3-270m-student-dpo-v1
```

**Time**: ~1 minute

### Step 5: Test Your Model!

```bash
ollama run gemma3-270m-student-dpo-v1 'Fix the login bug'
```

## Expected Results

| Model | Quality | Speed | Memory | Dataset Size |
|-------|---------|-------|--------|--------------|
| Teacher (27B) | 100% | 1x | 54GB | - |
| Student Phase 1 | ~85% | 100x | 1GB | 432 examples |
| Student DPO (432) | ~90% | 100x | 1GB | 432 examples |
| Student DPO (932) | ~92-93% | 100x | 1GB | 932 examples |

**Key insight**: The extended synthetic dataset typically gives you an extra 2-3% quality improvement!

## Iterative Improvement

For even better results, run multiple DPO rounds:

```bash
# Round 1
just generate-dpo-extended gemma3:27b models/gemma3-270m-student-unsloth-v1 500 data/gpt5nano/train_dpo_r1.jsonl
just train-dpo models/gemma3-270m-student-unsloth-v1 data/gpt5nano/train_dpo_r1.jsonl
# Creates: gemma3-270m-student-dpo-v1

# Round 2 (use v1 as new student baseline)
just generate-dpo-extended gemma3:27b models/gemma3-270m-student-dpo-v1 500 data/gpt5nano/train_dpo_r2.jsonl
just train-dpo models/gemma3-270m-student-dpo-v1 data/gpt5nano/train_dpo_r2.jsonl
# Creates: gemma3-270m-student-dpo-v2

# Round 3 (diminishing returns after this)
just generate-dpo-extended gemma3:27b models/gemma3-270m-student-dpo-v2 500 data/gpt5nano/train_dpo_r3.jsonl
just train-dpo models/gemma3-270m-student-dpo-v2 data/gpt5nano/train_dpo_r3.jsonl
# Creates: gemma3-270m-student-dpo-v3
```

Each round typically improves quality by 1-3% until convergence (usually 2-3 rounds).

## Troubleshooting

### Claude API errors during synthetic generation
- Check `ANTHROPIC_API_KEY` is set in `.env`
- Check you have API credits
- Try reducing `--num-synthetic` to 250 or 100

### Ollama connection errors
- Make sure Ollama is running: `ollama list`
- Check teacher model is pulled: `ollama pull gemma3:27b`
- Verify Ollama is listening on port 11434

### Out of memory during DPO training
- Reduce batch size: Edit the `train-dpo` command to use `--batch-size 2`
- Reduce sequence length: Add `--max-length 1024`

### DPO model quality is worse than Phase 1
- Your learning rate might be too high, try `--learning-rate 2e-5`
- Beta might be too low, try `--beta 0.2` or `0.3`
- You might be overfitting, reduce epochs to 1-2

## Cost Estimates

### API Costs (Claude for synthetic generation)
- 500 synthetic prompts: ~$0.50-$1.00 (using Claude 3.5 Sonnet)
- 1000 synthetic prompts: ~$1.00-$2.00

### Compute Time (on DGX Spark / H100)
- Synthetic generation: ~20-30 min
- Teacher/student inference (932 examples): ~30-40 min
- DPO training (3 epochs, 932 examples): ~60-90 min
- **Total**: ~2-3 hours for the complete workflow

## Next Steps

1. Compare models using the comparison script:
```bash
uv run python scripts/distillation/compare_models.py \
  --teacher gemma3:27b \
  --student-baseline models/gemma3-270m-student-unsloth-v1 \
  --student-dpo models/gemma3-270m-student-dpo-v1 \
  --test-data data/gpt5nano/test.jsonl
```

2. Deploy the best model to production via Ollama

3. Consider iterative DPO rounds if quality isn't sufficient
