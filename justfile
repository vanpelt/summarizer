# Gemma3-270M Distillation Project Justfile
# Manages training, DPO, evaluation, and export tasks

# Default recipe - show available commands
default:
    @just --list

# Build Unsloth Docker image
unsloth-build:
    docker build -t spark-unsloth ./docker --load

# Train a single model (Axolotl/QLoRA)
train CONFIG="configs/qlora-gemma3-270m-student.yml":
    uv run python scripts/training/train_single.py --config {{CONFIG}}

# Train multiple models concurrently (Axolotl/QLoRA)
train-multi PRESET="default":
    uv run python scripts/training/train_multi.py --preset {{PRESET}}

# Generate extended DPO dataset with synthetic prompts (Phase 2 distillation)
# Configure via environment variables:
#   NUM_SYNTHETIC=1000 BATCH=8 just generate-dpo-extended
#   STUDENT_BACKEND=ollama STUDENT=gemma3-270m-student just generate-dpo-extended
#   RESUME=1 just generate-dpo-extended  # Resume from checkpoint
generate-dpo-extended:
    #!/usr/bin/env bash
    TEACHER="{{ env('TEACHER', 'gemma3:27b') }}"
    STUDENT_BACKEND="{{ env('STUDENT_BACKEND', 'unsloth') }}"
    STUDENT="{{ env('STUDENT', 'models/gemma3-270m-student-unsloth-v1') }}"
    NUM_SYNTHETIC="{{ env('NUM_SYNTHETIC', '500') }}"
    EXISTING_DATA="{{ env('EXISTING_DATA', 'data/synthetic/train.jsonl') }}"
    OUTPUT="{{ env('OUTPUT', 'data/synthetic/train_dpo_extended.jsonl') }}"
    TEMPERATURE="{{ env('TEMPERATURE', '0.7') }}"
    BATCH="{{ env('BATCH', '2') }}"
    RESUME="{{ env('RESUME', '') }}"

    echo "Generating extended DPO preference dataset with synthetic prompts..."
    echo "Teacher: $TEACHER"
    echo "Student backend: $STUDENT_BACKEND"
    echo "Student: $STUDENT"
    echo "Existing data: $EXISTING_DATA"
    echo "Synthetic prompts: $NUM_SYNTHETIC"
    echo "Output: $OUTPUT"
    echo "Temperature: $TEMPERATURE"
    echo "Batch size: $BATCH (for DPO generation)"

    RESUME_FLAG=""
    if [ -n "$RESUME" ]; then
        RESUME_FLAG="--resume"
        echo "Resume mode: ON"
    fi

    uv run --no-project python scripts/distillation/extend_dpo_with_synthetic.py \
        --existing-data "$EXISTING_DATA" \
        --num-synthetic "$NUM_SYNTHETIC" \
        --teacher-backend auto \
        --teacher-model "$TEACHER" \
        --student-backend "$STUDENT_BACKEND" \
        --student-model "$STUDENT" \
        --output "$OUTPUT" \
        --temperature "$TEMPERATURE" \
        --batch-size "$BATCH" \
        --save-every 10 \
        --ollama-base-url http://localhost:11434 \
        $RESUME_FLAG

# Inspect DPO preference dataset
inspect-dpo-dataset DATASET="data/synthetic/train_dpo_extended.jsonl":
    uv run --no-project python scripts/distillation/inspect_dpo_dataset.py {{DATASET}}

# Train with DPO (Phase 2 distillation - RL-based refinement)
# Configure via environment variables:
#   BASE=models/my-model DATASET=data/my_dpo.jsonl just train-dpo
#   VAL_DATASET=data/synthetic/val_dpo.jsonl just train-dpo  # Use DPO-formatted val set
train-dpo:
    #!/usr/bin/env bash
    BASE="{{ env('BASE', 'models/gemma3-270m-student-unsloth-v1_merged') }}"
    DATASET="{{ env('DATASET', 'data/synthetic/train_dpo.jsonl') }}"
    VAL_DATASET="{{ env('VAL_DATASET', 'data/synthetic/val_dpo.jsonl') }}"
    IMAGE="{{ env('IMAGE', 'spark-unsloth') }}"
    BATCH_SIZE="{{ env('BATCH_SIZE', '4') }}"
    GRAD_ACCUM="{{ env('GRAD_ACCUM', '4') }}"
    LR="{{ env('LR', '5e-5') }}"
    EPOCHS="{{ env('EPOCHS', '3') }}"
    BETA="{{ env('BETA', '0.1') }}"
    LORA_R="{{ env('LORA_R', '64') }}"
    LORA_ALPHA="{{ env('LORA_ALPHA', '128') }}"

    echo "Training with DPO (Direct Preference Optimization) in Docker..."
    echo "Base model: $BASE"
    echo "Train dataset: $DATASET"
    echo "Val dataset: $VAL_DATASET"
    echo "Batch size: $BATCH_SIZE, Grad accum: $GRAD_ACCUM"
    echo "LR: $LR, Epochs: $EPOCHS, Beta: $BETA"
    echo "LoRA r: $LORA_R, alpha: $LORA_ALPHA"

    docker run --rm \
        --gpus=all \
        --net=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace \
        -v ~/.netrc:/root/.netrc:ro \
        -v ~/.cache/uv:/root/.cache/uv \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/wandb:/root/.cache/wandb \
        -w /workspace \
        -e WANDB_PROJECT=summarizer \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e UV_NO_PROJECT=1 \
        $IMAGE \
        python scripts/distillation/train_dpo.py \
            --base-model "$BASE" \
            --dataset "$DATASET" \
            --val-dataset "$VAL_DATASET" \
            --batch-size "$BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --learning-rate "$LR" \
            --epochs "$EPOCHS" \
            --beta "$BETA" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA"

# Run Unsloth Docker container (interactive shell)
unsloth-shell IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    echo "Starting Unsloth Docker container (interactive)..."
    docker run -it --rm \
        --gpus=all \
        --net=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace \
        -v ~/.netrc:/root/.netrc:ro \
        -v ~/.cache/uv:/root/.cache/uv \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/wandb:/root/.cache/wandb \
        -w /workspace \
        -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-} \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-} \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e UV_NO_PROJECT=1 \
        {{IMAGE}} \
        bash

# Train with Unsloth in Docker (recommended for GB10)
# Configure via environment variables to override script defaults:
#   MODEL_VARIANT=1b just unsloth-train  # Train 1B model
#   MODEL_VARIANT=4b just unsloth-train  # Train 4B model
#   MODEL_VARIANT=12b just unsloth-train  # Train 12B model
#   MODEL_VARIANT=27b just unsloth-train  # Train 27B model
#   TRAIN_DATA=data/custom/train.jsonl just unsloth-train
#   EVAL_DATA=data/custom/test.jsonl just unsloth-train
#   OUTPUT_DIR=models/gemma3-270m-v2 RUN_NAME=gemma3-270m-v2 just unsloth-train
#   EPOCHS=5 BATCH_SIZE=4 MAX_MEMORY_GB=100 just unsloth-train
#   MAX_STEPS=100 just unsloth-train  # Override epochs with max_steps
#   LR=5e-5 just unsloth-train  # Override learning rate
#   NUM_EVAL_EXAMPLES=20 just unsloth-train  # Log 20 eval examples
unsloth-train IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    # Read environment variables with defaults from script
    MODEL_VARIANT="{{ env('MODEL_VARIANT', '') }}"
    TRAIN_DATA="{{ env('TRAIN_DATA', '') }}"
    EVAL_DATA="{{ env('EVAL_DATA', '') }}"
    OUTPUT_DIR="{{ env('OUTPUT_DIR', '') }}"
    RUN_NAME="{{ env('RUN_NAME', '') }}"
    EPOCHS="{{ env('EPOCHS', '') }}"
    BATCH_SIZE="{{ env('BATCH_SIZE', '') }}"
    MAX_MEMORY_GB="{{ env('MAX_MEMORY_GB', '') }}"
    MAX_STEPS="{{ env('MAX_STEPS', '') }}"
    LR="{{ env('LR', '') }}"
    NUM_EVAL_EXAMPLES="{{ env('NUM_EVAL_EXAMPLES', '') }}"

    echo "Starting Unsloth training in Docker..."

    # Build command arguments from environment variables
    CMD="python scripts/training/train_unsloth_gemma3.py"
    [ -n "$MODEL_VARIANT" ] && CMD="$CMD --model-variant $MODEL_VARIANT" && echo "  Model variant: $MODEL_VARIANT"
    [ -n "$TRAIN_DATA" ] && CMD="$CMD --train-data $TRAIN_DATA" && echo "  Train data: $TRAIN_DATA"
    [ -n "$EVAL_DATA" ] && CMD="$CMD --eval-data $EVAL_DATA" && echo "  Eval data: $EVAL_DATA"
    [ -n "$OUTPUT_DIR" ] && CMD="$CMD --output-dir $OUTPUT_DIR" && echo "  Output dir: $OUTPUT_DIR"
    [ -n "$RUN_NAME" ] && CMD="$CMD --run-name $RUN_NAME" && echo "  Run name: $RUN_NAME"
    [ -n "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS" && echo "  Epochs: $EPOCHS"
    [ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE" && echo "  Batch size: $BATCH_SIZE"
    [ -n "$MAX_MEMORY_GB" ] && CMD="$CMD --max-memory-gb $MAX_MEMORY_GB" && echo "  Max memory: ${MAX_MEMORY_GB}GB"
    [ -n "$MAX_STEPS" ] && CMD="$CMD --max-steps $MAX_STEPS" && echo "  Max steps: $MAX_STEPS"
    [ -n "$LR" ] && CMD="$CMD --learning-rate $LR" && echo "  Learning rate: $LR"
    [ -n "$NUM_EVAL_EXAMPLES" ] && CMD="$CMD --num-eval-examples $NUM_EVAL_EXAMPLES" && echo "  Eval examples: $NUM_EVAL_EXAMPLES"

    docker run --rm \
        --gpus=all \
        --net=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace \
        -v ~/.netrc:/root/.netrc:ro \
        -v ~/.cache/uv:/root/.cache/uv \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/wandb:/root/.cache/wandb \
        -w /workspace \
        -e WANDB_PROJECT=summarizer \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e UV_NO_PROJECT=1 \
        {{IMAGE}} \
        bash -c "rm -rf /workspace/unsloth_compiled_cache /tmp/torchinductor_* && $CMD"

# Shortcut recipes for training specific model variants
unsloth-train-270m:
    MODEL_VARIANT=270m just unsloth-train

unsloth-train-1b:
    MODEL_VARIANT=1b just unsloth-train

unsloth-train-4b:
    MODEL_VARIANT=4b just unsloth-train

unsloth-train-12b:
    MODEL_VARIANT=12b just unsloth-train

unsloth-train-27b:
    MODEL_VARIANT=27b just unsloth-train

# Convert JSON format datasets to two-line format
convert-two-line:
    uv run python scripts/data/convert_to_two_line.py

# Train with two-line format datasets
# This uses the same unsloth training but with two-line format data
unsloth-train-two-line IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    # Override defaults for two-line format training
    TRAIN_DATA="{{ env('TRAIN_DATA', 'data/synthetic_two_line/train.jsonl') }}"
    EVAL_DATA="{{ env('EVAL_DATA', 'data/synthetic_two_line/test.jsonl') }}"
    OUTPUT_DIR="{{ env('OUTPUT_DIR', 'models/gemma3-270m-synthetic-two-line-v1') }}"
    RUN_NAME="{{ env('RUN_NAME', 'gemma3-270m-synthetic-two-line-v1') }}"
    MODEL_VARIANT="{{ env('MODEL_VARIANT', '') }}"
    EPOCHS="{{ env('EPOCHS', '') }}"
    BATCH_SIZE="{{ env('BATCH_SIZE', '') }}"
    MAX_MEMORY_GB="{{ env('MAX_MEMORY_GB', '') }}"
    MAX_STEPS="{{ env('MAX_STEPS', '') }}"
    LR="{{ env('LR', '') }}"
    NUM_EVAL_EXAMPLES="{{ env('NUM_EVAL_EXAMPLES', '') }}"

    echo "Starting Unsloth training with TWO-LINE format data in Docker..."

    # Build command arguments
    CMD="python scripts/training/train_unsloth_gemma3.py"
    CMD="$CMD --train-data $TRAIN_DATA"
    CMD="$CMD --eval-data $EVAL_DATA"
    CMD="$CMD --output-dir $OUTPUT_DIR"
    CMD="$CMD --run-name $RUN_NAME"
    echo "  Train data: $TRAIN_DATA"
    echo "  Eval data: $EVAL_DATA"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  Run name: $RUN_NAME"

    [ -n "$MODEL_VARIANT" ] && CMD="$CMD --model-variant $MODEL_VARIANT" && echo "  Model variant: $MODEL_VARIANT"
    [ -n "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS" && echo "  Epochs: $EPOCHS"
    [ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE" && echo "  Batch size: $BATCH_SIZE"
    [ -n "$MAX_MEMORY_GB" ] && CMD="$CMD --max-memory-gb $MAX_MEMORY_GB" && echo "  Max memory: ${MAX_MEMORY_GB}GB"
    [ -n "$MAX_STEPS" ] && CMD="$CMD --max-steps $MAX_STEPS" && echo "  Max steps: $MAX_STEPS"
    [ -n "$LR" ] && CMD="$CMD --learning-rate $LR" && echo "  Learning rate: $LR"
    [ -n "$NUM_EVAL_EXAMPLES" ] && CMD="$CMD --num-eval-examples $NUM_EVAL_EXAMPLES" && echo "  Eval examples: $NUM_EVAL_EXAMPLES"

    docker run --rm \
        --gpus=all \
        --net=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace \
        -v ~/.netrc:/root/.netrc:ro \
        -v ~/.cache/uv:/root/.cache/uv \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/wandb:/root/.cache/wandb \
        -w /workspace \
        -e WANDB_PROJECT=summarizer \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e UV_NO_PROJECT=1 \
        {{IMAGE}} \
        bash -c "rm -rf /workspace/unsloth_compiled_cache /tmp/torchinductor_* && $CMD"

# Export trained model to GGUF for Ollama
export-gguf MODEL="./models/gemma3-270m-student-unsloth-v1" NAME="" QUANT="Q4_K_M" IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    # If NAME is empty, use basename of MODEL directory
    if [ -z "{{NAME}}" ]; then
        NAME=$(basename "{{MODEL}}")
    else
        NAME="{{NAME}}"
    fi

    echo "Exporting model to GGUF..."
    echo "Model: {{MODEL}}"
    echo "Output name: $NAME"
    echo "Quantization: {{QUANT}}"

    docker run --rm \
        --gpus=all \
        -v $(pwd):/workspace \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -w /workspace \
        {{IMAGE}} \
        python scripts/export/export_to_gguf.py --model-path {{MODEL}} --output-name "$NAME" --quantization {{QUANT}}

# Upload GGUF model to HuggingFace Hub
# Usage: just upload-to-hf gemma3-270m-synthetic-v11 vanpelt/catnip-summarizer
# Configure via environment variables:
#   WANDB_RUN=https://wandb.ai/... just upload-to-hf ...
#   INCLUDE_SAFETENSORS=1 just upload-to-hf ...  (includes safetensors, adds ~512MB)
#   DESCRIPTION="My model description" just upload-to-hf ...
upload-to-hf MODEL_DIR REPO_ID="vanpelt/catnip-summarizer" PRIVATE="--private":
    #!/usr/bin/env bash
    WANDB_RUN="{{ env('WANDB_RUN', '') }}"
    INCLUDE_SAFETENSORS="{{ env('INCLUDE_SAFETENSORS', '') }}"
    DESCRIPTION="{{ env('DESCRIPTION', '') }}"

    echo "Uploading GGUF model to HuggingFace Hub..."
    echo "Model: models/gguf/{{MODEL_DIR}}"
    echo "Repo: {{REPO_ID}}"

    PRIVACY_FLAG=""
    if [ "{{PRIVATE}}" = "--private" ]; then
        PRIVACY_FLAG="--private"
        echo "Visibility: Private"
    else
        echo "Visibility: Public"
    fi

    INCLUDE_FLAG=""
    if [ -n "$INCLUDE_SAFETENSORS" ]; then
        INCLUDE_FLAG="--include-safetensors"
        echo "Including: model.safetensors (~512MB)"
    else
        echo "Excluding: model.safetensors (use INCLUDE_SAFETENSORS=1 to upload)"
    fi

    WANDB_FLAG=""
    if [ -n "$WANDB_RUN" ]; then
        WANDB_FLAG="--wandb-run $WANDB_RUN"
        echo "W&B Run: $WANDB_RUN"
    fi

    DESC_FLAG=""
    if [ -n "$DESCRIPTION" ]; then
        DESC_FLAG="--description \"$DESCRIPTION\""
        echo "Description: $DESCRIPTION"
    fi

    uv run --no-project python scripts/export/upload_to_hf.py \
        --model-dir models/gguf/{{MODEL_DIR}} \
        --repo-id {{REPO_ID}} \
        $PRIVACY_FLAG \
        $INCLUDE_FLAG \
        $WANDB_FLAG \
        $DESC_FLAG

# Import GGUF model into Ollama
# Usage: just ollama-import <directory-name> [model-name]
# If model-name is not provided, uses directory-name
ollama-import DIR_NAME MODEL_NAME="":
    #!/usr/bin/env bash
    # If MODEL_NAME is empty, use DIR_NAME
    if [ -z "{{MODEL_NAME}}" ]; then
        MODEL_NAME="{{DIR_NAME}}"
    else
        MODEL_NAME="{{MODEL_NAME}}"
    fi

    echo "Importing from directory: {{DIR_NAME}}"
    echo "Creating Ollama model: $MODEL_NAME"

    # Check if the model directory exists
    MODEL_DIR="models/gguf/{{DIR_NAME}}"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model directory not found: $MODEL_DIR"
        echo ""
        echo "Available models:"
        ls -d models/gguf/*/ 2>/dev/null || echo "  (none)"
        echo ""
        echo "Run 'just export-gguf' to create one"
        exit 1
    fi

    # Check if Modelfile exists
    if [ ! -f "$MODEL_DIR/Modelfile" ]; then
        echo "Error: Modelfile not found in $MODEL_DIR"
        exit 1
    fi

    echo "Using model directory: $MODEL_DIR"

    # Import from the model directory
    cd "$MODEL_DIR"
    ollama create "$MODEL_NAME" -f Modelfile
    cd - > /dev/null

    echo ""
    echo "✅ Model imported as: $MODEL_NAME"
    echo "Test with: ollama run $MODEL_NAME 'Fix the login bug'"

# Smoke test
# Usage: just smoke-test MODEL
# Usage with custom temp: HEAT=1.2 just smoke-test MODEL
smoke-test MODEL="gemma3-summary-v4" HEAT="0.8":
    #!/usr/bin/env bash
    FORMAT_FLAG=""
    if [[ "{{MODEL}}" != *"2l"* ]]; then FORMAT_FLAG="--format json"; fi
    TEST_MODEL="{{MODEL}}"
    if [ "{{HEAT}}" != "0.8" ]; then \
        TEMP_MODEL="{{MODEL}}-heat{{HEAT}}"; \
        echo "FROM {{MODEL}}" > /tmp/smoke-modelfile; \
        echo "PARAMETER temperature {{HEAT}}" >> /tmp/smoke-modelfile; \
        ollama create "$TEMP_MODEL" -f /tmp/smoke-modelfile || true; \
        TEST_MODEL="$TEMP_MODEL"; \
        echo "Using temp model: $TEMP_MODEL (temperature={{HEAT}})"; \
    fi
    echo ""; echo "-------------------"; echo "Prompt: Fix the bug with the button not re-activating"; echo "-------------------"; ollama run "$TEST_MODEL" --verbose $FORMAT_FLAG "Fix the bug with the button not re-activating"
    echo ""; echo "-------------------"; echo "Prompt: Fix the login bug"; echo "-------------------"; ollama run "$TEST_MODEL" --verbose $FORMAT_FLAG "Fix the login bug"
    echo ""; echo "-------------------"; echo "Prompt: Add a new feature to the login page"; echo "-------------------"; ollama run "$TEST_MODEL" --verbose $FORMAT_FLAG "Add a new feature to the login page"
    echo ""; echo "-------------------"; echo "Prompt: My XML isn't parsing we need to fix it"; echo "-------------------"; ollama run "$TEST_MODEL" --verbose $FORMAT_FLAG "My XML isn't parsing we need to fix it"
    echo ""; echo "-------------------"; echo "Prompt: That button isn't working, make it animate"; echo "-------------------"; ollama run "$TEST_MODEL" --verbose $FORMAT_FLAG "That button isn't working, make it animate"
    if [ "{{HEAT}}" != "0.8" ]; then ollama rm "$TEMP_MODEL" || true; fi

# Run tests
test:
    uv run pytest tests/ -v

# Format code
format:
    uv run black src/ scripts/ tests/
    uv run ruff check --fix src/ scripts/ tests/

# Check code quality
lint:
    uv run ruff check src/ scripts/ tests/
    uv run mypy src/

# Clean up Python cache and build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    rm -rf .pytest_cache .mypy_cache .coverage htmlcov

# Benchmark GGUF model with llama-bench
# Usage: just bench <model-dir-name>
# Example: just bench gemma3-270m-synthetic-v11
bench MODEL_DIR PROMPTS="64,128" TOKENS="32,64":
    #!/usr/bin/env bash
    GGUF_PATH="models/gguf/{{MODEL_DIR}}/gemma3-270m-summarizer-Q4_K_M.gguf"

    if [ ! -f "$GGUF_PATH" ]; then
        echo "Error: Model not found: $GGUF_PATH"
        echo ""
        echo "Available models:"
        find models/gguf -name "*.gguf" -type f | grep -E "gemma3.*Q4_K_M" | head -10
        exit 1
    fi

    echo "Benchmarking: {{MODEL_DIR}}"
    echo "Model: $GGUF_PATH"
    echo "Prompt sizes: {{PROMPTS}}"
    echo "Generation sizes: {{TOKENS}}"
    echo ""
    echo "Running: ../llama.cpp/build/bin/llama-bench -m \"$GGUF_PATH\" -p {{PROMPTS}} -n {{TOKENS}} -b 512 -ub 128 -t 8 -ngl 99 -r 3 --output md"
    echo ""

    ../llama.cpp/build/bin/llama-bench \
        -m "$GGUF_PATH" \
        -p {{PROMPTS}} \
        -n {{TOKENS}} \
        -b 512 \
        -ub 128 \
        -t 8 \
        -ngl 99 \
        -r 3 \
        --output md

# Compare two models side by side
# Usage: just bench-compare model1 model2
bench-compare MODEL1 MODEL2:
    @echo "=== Benchmarking {{MODEL1}} ==="
    @just bench {{MODEL1}}
    @echo ""
    @echo "=== Benchmarking {{MODEL2}} ==="
    @just bench {{MODEL2}}

# Start development server for browser inference demo
# Usage: just dev-server [port]
# Then open: http://localhost:8000/?model=gemma3-270m-synthetic-two-line-v1
dev-server PORT="8000":
    uv run python docs/dev_server.py --port {{PORT}}

# Monitor GPU usage
gpu:
    watch -n 1 nvidia-smi
