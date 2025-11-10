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
  --input data/gpt5nano/train.jsonl \
  --output data/gpt5nano/train_teacher.jsonl

# Or using vLLM
uv run python scripts/distillation/generate_teacher_outputs.py \
  --backend vllm \
  --teacher-model google/gemma-3-27b-it \
  --input data/gpt5nano/train.jsonl \
  --output data/gpt5nano/train_teacher.jsonl
```

This will:
- Load your 432 training examples
- Generate teacher outputs for each
- Save to `data/gpt5nano/train_teacher.jsonl`
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
  --test-data data/gpt5nano/test.jsonl
```

## Expected Results

Typical performance hierarchy:

| Model | Quality | Speed | Memory |
|-------|---------|-------|--------|
| Teacher (27B) | 100% | 1x | 54GB |
| Student-Distilled (270M) | ~85-90% | 100x | 1GB |
| Student-Baseline (270M) | ~75-80% | 100x | 1GB |

## Phase 2: RL-based Distillation

Once Phase 1 is working, we can explore RL methods:

### Option A: OpenPipe

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
  --dataset data/gpt5nano/train.jsonl \
  --eval data/gpt5nano/val.jsonl
```

### Option B: TRL (Transformers Reinforcement Learning)

Open-source library from HuggingFace for RL fine-tuning.

**Methods:**
- **DPO (Direct Preference Optimization)**: Simpler, works well
- **PPO (Proximal Policy Optimization)**: More powerful, more complex

**Example:**
```python
from trl import DPOTrainer

# Create preference dataset: teacher output (chosen) vs student output (rejected)
# Train student to prefer teacher-like outputs
trainer = DPOTrainer(
    model=student_model,
    ref_model=ref_student_model,
    train_dataset=preference_dataset,
    beta=0.1,  # KL penalty
)
trainer.train()
```

### Option C: Custom Implementation

Implement custom RL loop:

1. **Student generates** output for prompt
2. **Teacher scores** student output (reward)
3. **Student updates** to maximize reward
4. Repeat

This gives maximum control but requires more engineering.

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
