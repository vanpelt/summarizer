#!/bin/bash
# Quick-start script for teacher-student distillation
# This script automates the entire distillation pipeline

set -e  # Exit on error

echo "=========================================="
echo "Teacher-Student Distillation Pipeline"
echo "Teacher: Gemma3-27B"
echo "Student: Gemma3-270M"
echo "=========================================="
echo ""

# Configuration
TEACHER_MODEL="${TEACHER_MODEL:-gemma3:27b}"
BACKEND="${BACKEND:-ollama}"
TRAIN_DATA="data/gpt5nano/train.jsonl"
VAL_DATA="data/gpt5nano/val.jsonl"
TEST_DATA="data/gpt5nano/test.jsonl"
TEACHER_OUTPUT="data/gpt5nano/train_teacher.jsonl"

# Step 0: Check if teacher model is available
echo "Step 0: Checking teacher model availability..."
if [ "$BACKEND" == "ollama" ]; then
    if ! command -v ollama &> /dev/null; then
        echo "ERROR: Ollama not found. Please install Ollama first:"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi

    if ! ollama list | grep -q "$TEACHER_MODEL"; then
        echo "Teacher model not found. Pulling $TEACHER_MODEL..."
        ollama pull "$TEACHER_MODEL"
    else
        echo "✓ Teacher model $TEACHER_MODEL is available"
    fi
fi

# Step 1: Generate teacher outputs
echo ""
echo "=========================================="
echo "Step 1: Generating teacher outputs"
echo "=========================================="
if [ -f "$TEACHER_OUTPUT" ]; then
    echo "Teacher outputs already exist at $TEACHER_OUTPUT"
    read -p "Regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping teacher generation..."
    else
        echo "Generating teacher outputs..."
        uv run python scripts/distillation/generate_teacher_outputs.py \
            --backend "$BACKEND" \
            --teacher-model "$TEACHER_MODEL" \
            --input "$TRAIN_DATA" \
            --output "$TEACHER_OUTPUT"
    fi
else
    echo "Generating teacher outputs..."
    uv run python scripts/distillation/generate_teacher_outputs.py \
        --backend "$BACKEND" \
        --teacher-model "$TEACHER_MODEL" \
        --input "$TRAIN_DATA" \
        --output "$TEACHER_OUTPUT"
fi

# Step 2: Train baseline student
echo ""
echo "=========================================="
echo "Step 2: Training baseline student (optional)"
echo "=========================================="
if [ -d "models/gemma3-270m-student-v1" ]; then
    echo "Baseline student already exists"
    read -p "Retrain? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Training baseline student..."
        just train configs/qlora-gemma3-270m-student.yml
    fi
else
    read -p "Train baseline student for comparison? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Training baseline student..."
        just train configs/qlora-gemma3-270m-student.yml
    else
        echo "Skipping baseline training..."
    fi
fi

# Step 3: Train distilled student
echo ""
echo "=========================================="
echo "Step 3: Training distilled student"
echo "=========================================="
if [ -d "models/gemma3-270m-distilled-v1" ]; then
    echo "Distilled student already exists"
    read -p "Retrain? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping distilled training..."
    else
        echo "Training distilled student..."
        just train configs/qlora-gemma3-270m-distilled.yml
    fi
else
    echo "Training distilled student on teacher outputs..."
    just train configs/qlora-gemma3-270m-distilled.yml
fi

# Step 4: Compare models
echo ""
echo "=========================================="
echo "Step 4: Comparing models"
echo "=========================================="
read -p "Run comparison on test set? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Comparing models..."
    uv run python scripts/distillation/compare_models.py \
        --teacher "$TEACHER_MODEL" \
        --student-baseline models/gemma3-270m-student-v1 \
        --student-distilled models/gemma3-270m-distilled-v1 \
        --test-data "$TEST_DATA"
fi

echo ""
echo "=========================================="
echo "Distillation pipeline complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the comparison results"
echo "  2. If quality is good, deploy the distilled model"
echo "  3. If quality needs improvement, try:"
echo "     - Adjusting teacher temperature"
echo "     - Increasing LoRA rank"
echo "     - Training for more epochs"
echo "  4. For further improvement, explore RL-based distillation (see DISTILLATION.md)"
echo ""
