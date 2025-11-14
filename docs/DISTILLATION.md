# Teacher-Student Distillation Guide

This guide covers knowledge distillation from **Gemma3-27B (teacher)** to **Gemma3-270M (student)** for the summarization task.

## Overview

**Knowledge Distillation** is a technique where a smaller "student" model learns to mimic a larger "teacher" model. The student can achieve near-teacher performance while being much faster and more memory-efficient.

### Models

- **Teacher**: `gemma3:27b` (27B parameters)
  - High quality but slow and memory-intensive
  - Used to generate reference outputs

- **Student**: `gemma3:270m` (270M parameters)
  - 100x smaller than teacher
  - Much faster inference
  - Goal: Match ~90% of teacher quality

## Approach

We'll implement distillation in two phases:

### Phase 1: Simple Knowledge Distillation (Start Here)

Train the student model on teacher-generated outputs instead of original labels.

**Steps:**
1. Generate teacher outputs for training data
2. Train student on teacher outputs
3. Evaluate against original validation set

**Pros:**
- Simple to implement
- No special RL infrastructure needed
- Works well for many tasks

**Cons:**
- Student learns from static teacher outputs
- No iterative improvement

### Phase 2: RL-based Distillation (Advanced)

Use reinforcement learning where teacher provides reward signal.

**Options:**
- **OpenPipe**: Managed distillation service
- **TRL (Transformer Reinforcement Learning)**: Open-source RL for LLMs
- **Custom PPO/DPO**: Implement your own RL loop

We'll focus on **Phase 1** first since it's simpler and often achieves 85-95% of Phase 2 performance.

## Phase 1: Implementation

### Step 1: Setup Teacher Model

You have two options for running the teacher:

#### Option A: Using Ollama (Recommended)

```bash
# Install Ollama if not already installed
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma3-27B
ollama pull gemma3:27b

# Test it
ollama run gemma3:27b "Generate JSON: Add dark mode"
```

#### Option B: Using vLLM

```bash
# Start vLLM server with Gemma3-27B
./fix.sh uv run python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-27b-it \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096
```

### Step 2: Generate Teacher Outputs

```bash
# Using Ollama (default)
uv run python scripts/distillation/generate_teacher_outputs.py \
  --backend ollama \
  --teacher-model gemma3:27b \
  --input data/synthetic/train.jsonl \
  --output data/synthetic/train_teacher.jsonl

# Or using vLLM
uv run python scripts/distillation/generate_teacher_outputs.py \
  --backend vllm \
  --teacher-model google/gemma-3-27b-it \
  --input data/synthetic/train.jsonl \
  --output data/synthetic/train_teacher.jsonl
```

This will:
- Load your 432 training examples
- Generate teacher outputs for each
- Save to `data/synthetic/train_teacher.jsonl`
- Take ~5-10 minutes with Ollama

### Step 3: Train Student Model (Baseline)

First, train the student on original data for comparison:

```bash
# Baseline: Student trained on original data
just train configs/qlora-gemma3-270m-student.yml
```

This creates: `models/gemma3-270m-student-v1/`

### Step 4: Train Student Model (Distilled)

Now train on teacher outputs:

```bash
# Distilled: Student trained on teacher outputs
just train configs/qlora-gemma3-270m-distilled.yml
```

This creates: `models/gemma3-270m-distilled-v1/`

### Step 5: Compare Models

```bash
# Compare all three models on test set
uv run python scripts/distillation/compare_models.py \
  --teacher gemma3:27b \
  --student-baseline models/gemma3-270m-student-v1 \
  --student-distilled models/gemma3-270m-distilled-v1 \
  --test-data data/synthetic/test.jsonl
```

## Expected Results

Typical performance hierarchy:

| Model | Quality | Speed | Memory |
|-------|---------|-------|--------|
| Teacher (27B) | 100% | 1x | 54GB |
| Student-Distilled (270M) | ~85-90% | 100x | 1GB |
| Student-Baseline (270M) | ~75-80% | 100x | 1GB |

## Phase 2: RL-based Distillation (DPO Implementation)

Once Phase 1 is working, we can use DPO to further refine the student model. This implementation uses TRL (Transformers Reinforcement Learning) with Direct Preference Optimization.

### What is DPO?

DPO (Direct Preference Optimization) is a simpler alternative to RLHF that:
- Trains the model to prefer high-quality outputs (teacher) over lower-quality ones (student baseline)
- Doesn't require a separate reward model
- More stable and easier to tune than PPO
- Typically gives 5-10% additional improvement over Phase 1

### Implementation Steps

#### Step 1: Generate DPO Preference Dataset

You have two options:

**Option A: Use existing prompts only (faster)**

```bash
# Make sure Ollama is running with the teacher model
ollama pull gemma3:27b

# Generate preference dataset from existing training data
just generate-dpo-dataset \
  gemma3:27b \
  models/gemma3-270m-student-unsloth-v1 \
  data/synthetic/train.jsonl \
  data/synthetic/train_dpo.jsonl
```

This will:
- Use prompts from your existing training data (~432 examples)
- Generate outputs from both teacher (via Ollama) and student
- Create preference pairs in TRL format
- Save to `data/synthetic/train_dpo.jsonl`
- Take ~10-15 minutes

**Option B: Extend with synthetic prompts (recommended for better performance)**

```bash
# Generate extended dataset with 500 additional synthetic prompts
just generate-dpo-extended \
  gemma3:27b \
  models/gemma3-270m-student-unsloth-v1 \
  500 \
  data/synthetic/train_dpo_extended.jsonl
```

This will:
1. Load existing prompts from `data/synthetic/train.jsonl` (~432 examples)
2. Generate 500 new synthetic prompts using Claude API (requires `ANTHROPIC_API_KEY`)
3. Generate teacher and student outputs for ALL prompts (existing + synthetic)
4. Create extended preference dataset (~932 total examples)
5. Save to `data/synthetic/train_dpo_extended.jsonl`
6. Take ~30-40 minutes for the full dataset

**Why use synthetic data?**
- **More training examples**: ~2x the data (432 → 932 examples)
- **Better generalization**: Diverse synthetic prompts cover more edge cases
- **Improved performance**: Typically gives 2-3% additional quality improvement
- **Cost-effective**: Reuses existing teacher/student models

**JSON Schema Enforcement:**

Both scripts enforce valid JSON output by default:
- **Teacher (Ollama)**: Uses JSON schema constraint for guaranteed valid output
- **Student (Unsloth)**: Uses greedy decoding (temperature=0) for reliable JSON

This ensures your training data matches production inference where you'll use:
```bash
ollama run your-model --format json 'Fix the login bug'
```

See [JSON_ENFORCEMENT.md](./JSON_ENFORCEMENT.md) for details and customization.

**Dataset Format:**
```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": "Teacher's high-quality output",
  "rejected": "Student's lower-quality output"
}
```

#### Step 2: Train with DPO

```bash
# Train DPO model (auto-increments version)
# Use standard dataset:
just train-dpo \
  models/gemma3-270m-student-unsloth-v1 \
  data/synthetic/train_dpo.jsonl

# OR use extended dataset (recommended):
just train-dpo \
  models/gemma3-270m-student-unsloth-v1 \
  data/synthetic/train_dpo_extended.jsonl
```

This will:
- Load the base student model
- Create a reference model (frozen copy)
- Add LoRA adapters for training
- Train using DPO loss to prefer teacher outputs
- Auto-increment version: `gemma3-270m-student-dpo-v1`, `v2`, etc.
- Save adapter and merged model
- Take ~30-60 minutes for 3 epochs

**Auto-versioning:**
The script automatically finds existing DPO models and increments:
- First run: `gemma3-270m-student-dpo-v1`
- Second run: `gemma3-270m-student-dpo-v2`
- And so on...

#### Step 3: Export and Test

```bash
# Export to GGUF
just export-gguf \
  models/gemma3-270m-student-dpo-v1 \
  gemma3-270m-student-dpo-v1 \
  Q4_K_M

# Import to Ollama
just ollama-import gemma3-270m-student-dpo-v1

# Test it
ollama run gemma3-270m-student-dpo-v1 'Fix the login bug'
```

### DPO Hyperparameters

Key hyperparameters in `train_dpo.py`:

#### Training Settings
- `--learning-rate`: 5e-5 (lower than SFT, DPO is sensitive to LR)
- `--epochs`: 3 (usually 1-3 epochs sufficient)
- `--batch-size`: 4 (per device)
- `--grad-accum`: 4 (effective batch size = 16)

#### DPO Specific
- `--beta`: 0.1 (KL divergence penalty weight)
  - Higher beta (0.2-0.5): More conservative, stays closer to base model
  - Lower beta (0.05-0.1): More aggressive learning from preferences
  - Start with 0.1 and adjust based on results

#### LoRA Settings
- `--lora-r`: 64 (rank)
- `--lora-alpha`: 128 (scaling factor)

### Expected Results

Typical performance improvement:

| Model | Quality | Speed | Memory | Improvement |
|-------|---------|-------|--------|-------------|
| Teacher (27B) | 100% | 1x | 54GB | Baseline |
| Student-Phase1 (270M) | ~85% | 100x | 1GB | - |
| Student-DPO (270M) | ~90-92% | 100x | 1GB | +5-7% over Phase1 |

### Troubleshooting

#### DPO training is unstable
- **Lower learning rate**: Try 2e-5 or 1e-5
- **Increase beta**: Try 0.2 or 0.3 for more stability
- **Reduce epochs**: Try 1-2 epochs instead of 3

#### Model quality degrades
- **Beta too low**: Increase to 0.2-0.3
- **Learning rate too high**: Lower to 2e-5
- **Overfitting**: Train for fewer epochs or add more dropout

#### Out of memory
- **Reduce batch size**: Try `--batch-size 2`
- **Increase grad accumulation**: Keep effective batch size at 8-16
- **Reduce max length**: Try `--max-length 1024`

### Monitoring with Weights & Biases

Training automatically logs to W&B:
- DPO loss (should decrease)
- Reward accuracy (chosen > rejected)
- KL divergence (should stay < 1.0)
- Reward margins (difference between chosen/rejected scores)

Disable with `--no-wandb` flag.

### Advanced: Iterative DPO

For even better results, iterate DPO training:

```bash
# Round 1: DPO from Phase 1 student
just train-dpo \
  models/gemma3-270m-student-unsloth-v1 \
  data/synthetic/train_dpo.jsonl
# Creates: gemma3-270m-student-dpo-v1

# Round 2: Generate new preferences with DPO-v1 as student
just generate-dpo-dataset \
  gemma3:27b \
  models/gemma3-270m-student-dpo-v1 \
  data/synthetic/train.jsonl \
  data/synthetic/train_dpo_round2.jsonl

# Round 2: Train DPO on new preferences
just train-dpo \
  models/gemma3-270m-student-dpo-v1 \
  data/synthetic/train_dpo_round2.jsonl
# Creates: gemma3-270m-student-dpo-v2
```

Each iteration typically gives 1-3% improvement until convergence (usually 2-3 rounds).

### Alternative Options

#### Option A: OpenPipe

OpenPipe is a managed service for model distillation with RL.

**Pros:**
- Fully managed, no infrastructure
- Automatic hyperparameter tuning
- Built-in evaluation

**Cons:**
- Costs money (but probably cheaper than managing infra)
- Less control over the process

**Setup:**
```bash
pip install openpipe
openpipe init
openpipe distill \
  --teacher gemma3:27b \
  --student gemma3:270m \
  --dataset data/synthetic/train.jsonl \
  --eval data/synthetic/val.jsonl
```

#### Option C: Custom PPO Implementation

Implement custom PPO RL loop:

1. **Student generates** output for prompt
2. **Teacher scores** student output (reward)
3. **Student updates** to maximize reward via PPO
4. Repeat

This gives maximum control but requires more engineering and is more complex than DPO.

## Next Steps

1. **Start with Phase 1**: Run the teacher generation and train distilled student
2. **Evaluate quality**: Compare distilled vs baseline student
3. **Iterate if needed**: Adjust hyperparameters, try different temperatures
4. **Consider Phase 2**: If quality isn't good enough, explore RL methods

## Hyperparameter Tuning

Key hyperparameters for distillation:

### Teacher Generation
- `temperature`: 0.7-1.0 (higher = more diverse outputs)
- `top_p`: 0.9-0.95 (nucleus sampling)
- `repetition_penalty`: 1.0-1.1

### Student Training
- `learning_rate`: 0.0002-0.0005 (student often needs higher LR)
- `lora_r`: 64-128 (larger rank = more capacity)
- `num_epochs`: 3-10 (depends on dataset size)
- `dropout`: 0.05-0.15 (prevents overfitting to teacher)

## Troubleshooting

### Student quality is poor
- Increase LoRA rank (lora_r: 128)
- Train for more epochs
- Lower learning rate
- Generate teacher outputs with temperature=0.8-0.9

### Student overfits to teacher
- Add more dropout (0.1-0.15)
- Use weight decay
- Train for fewer epochs
- Use label smoothing

### Teacher generation fails
- Check Ollama is running: `ollama list`
- Try lower temperature
- Reduce max_tokens if hitting limits

## Resources

- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [OpenPipe Docs](https://openpipe.ai/docs)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
