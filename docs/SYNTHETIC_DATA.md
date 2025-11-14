# Enhanced Synthetic Dataset Generation

This guide explains how to generate an enhanced dataset for Gemma3-270M fine-tuning.

## Overview

The `generate_enhanced_dataset.py` script:
1. Loads existing 432 prompts from `data/gpt5nano/train.jsonl`
2. Generates 2000+ new diverse synthetic prompts using OpenAI (gpt-5-mini with low reasoning effort)
3. Generates JSON responses for all prompts using structured outputs
4. Converts to Unsloth format (with `conversations` field)
5. Splits into train (~2232) and test (200) datasets
6. Saves to `data/synthetic/`

## Data Format

### Unsloth Format (Required for training)
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<system prompt>\n\nRequest:\n<user request>"
    },
    {
      "role": "assistant",
      "content": "{\"summary\": \"Fix Login Bug\", \"branch\": \"bug/fix-login\"}"
    }
  ]
}
```

### System Prompt
The system prompt from `prompt.txt` is used:
```
You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
```

## Usage

### 1. Preview Mode (Recommended First)

Generate just 20 examples and preview them:

```bash
uv run python scripts/data/generate_enhanced_dataset.py --preview
```

This will show you:
- Sample user requests
- Generated JSON responses
- Data format

Review the output to ensure quality before generating the full dataset.

### 2. Generate Full Dataset

Once satisfied with preview:

```bash
uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 2000
```

This generates:
- **Train**: ~2232 examples (432 existing + 1800 synthetic)
- **Test**: 200 examples
- **Total**: ~2432 examples

Output files:
- `data/synthetic/train.jsonl` (training data)
- `data/synthetic/test.jsonl` (test data)
- `data/synthetic/synthetic_prompts.jsonl` (generated prompts for resume)

### 3. Resume Generation

If generation is interrupted or you want to add MORE data, use `--resume`:

```bash
# Add 1000 more synthetic prompts to existing dataset
uv run python scripts/data/generate_enhanced_dataset.py --resume --num-synthetic 1000
```

**How resume works:**
1. Loads existing `synthetic_prompts.jsonl` (previously generated prompts)
2. Loads existing `train.jsonl` and `test.jsonl` (already-labeled examples)
3. Generates `--num-synthetic` NEW prompts (avoiding duplicates with existing/labeled)
4. **Appends** new prompts to `synthetic_prompts.jsonl`
5. Labels ONLY the new prompts (skips already-labeled ones)
6. Merges new examples with existing train/test data
7. Re-shuffles and re-splits the combined dataset
8. **Overwrites** train/test files with merged data

**Example workflow:**
```bash
# Initial run: 2000 prompts
uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 2000
# Result: 2000 synthetic prompts + 432 existing = 2432 total

# Later: add 1000 more
uv run python scripts/data/generate_enhanced_dataset.py --resume --num-synthetic 1000
# Result: 3000 synthetic prompts + 432 existing = 3432 total

# Later: add 500 more
uv run python scripts/data/generate_enhanced_dataset.py --resume --num-synthetic 500
# Result: 3500 synthetic prompts + 432 existing = 3932 total
```

**Benefits:**
- ✅ Never re-labels existing examples (saves time and API costs)
- ✅ Appends to `synthetic_prompts.jsonl` (preserves history)
- ✅ Avoids duplicate prompts across all sources
- ✅ Re-shuffles to ensure good train/test distribution

### 4. Custom Options

```bash
# Generate more synthetic prompts
uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 3000

# Larger test set
uv run python scripts/data/generate_enhanced_dataset.py --test-size 300

# Custom output directory
uv run python scripts/data/generate_enhanced_dataset.py --output-dir data/my_dataset

# Adjust parallel workers (default: 10)
uv run python scripts/data/generate_enhanced_dataset.py --max-workers 5
```

### Parallel Processing

The script uses parallel API calls to speed up response generation:

- **Default**: 10 parallel workers
- **Performance**: ~10x faster than serial processing
- **Adjustable**: Use `--max-workers` to control parallelism

Example timing:
- Serial: ~16-17 minutes for 2000 examples
- Parallel (10 workers): ~1.7-2 minutes for 2000 examples

Note: Higher parallelism may trigger rate limits. Start with default and adjust if needed.

## Prompt Diversity

The script generates diverse prompts across multiple dimensions:

### Task Categories (20+)
- New features, Bug fixes, Refactoring
- Documentation, Testing, Performance optimization
- Security improvements, UI/UX enhancements
- API changes, Database migrations
- And more...

### Programming Languages (10)
- Python, JavaScript, TypeScript
- Go, Rust, Ruby
- C, C++, Java, Zig

### Application Types (8)
- Web App, Mobile App, Desktop App
- CLI Tool, SDK, Library
- API Service, Microservice

### Combination Grid
The script pre-calculates combinations of (category, language, app_type) to ensure:
- **Maximum diversity**: Each batch gets a unique combination
- **No duplicates**: Systematic cycling through options
- **Language-specific content**: Code examples and errors match the language
- **1600+ possible combinations**: 20 categories × 10 languages × 8 app types

Prompts vary in length:
- **30% SHORT** (10-50 words): "Add dark mode toggle"
- **40% MEDIUM** (50-100 words): Context + explanation
- **30% LONG** (100-1000+ words): Code snippets, stack traces, detailed steps

## Training

After generating the dataset, train with:

```bash
uv run python scripts/training/train_unsloth_gemma3.py \
    --train-data data/synthetic/train.jsonl \
    --eval-data data/synthetic/test.jsonl \
    --output-dir models/gemma3-270m-enhanced-v1 \
    --run-name gemma3-270m-enhanced-v1
```

Available options:
- `--max-memory-gb` - GPU memory limit in GB (default: 74)
- `--batch-size` - Training batch size (default: 8)
- `--epochs` - Number of epochs (default: 5)

## Troubleshooting

### Missing OPENAI_API_KEY
```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Generation fails mid-way
Use `--resume` to continue from saved prompts file

### Want different system prompt
Edit the `SYSTEM_PROMPT` variable in the script to match your needs

## Quality Checks

Before training, review:
1. Preview examples look realistic
2. JSON responses are valid
3. Branch names follow kebab-case pattern
4. Summaries are 2-4 words in Title Case
5. Prompt diversity is good (check different categories)

## Next Steps

After generating the dataset:
1. Review `data/synthetic/train.jsonl` and `data/synthetic/test.jsonl`
2. Run training (see command above)
3. Monitor training metrics in Weights & Biases
4. Test the fine-tuned model on real examples
