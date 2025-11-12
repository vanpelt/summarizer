# Qwen3-8B Fine-tuning Project Justfile
# Manages model downloads, vLLM server, and training tasks

# Default recipe - show available commands
default:
    @just --list

# Start vLLM server with proper environment variables
serve MODEL="Qwen/Qwen3-8B" PORT="8000" GPU_MEM="0.5":
    #!/usr/bin/env bash
    echo "Starting vLLM server..."
    echo "Model: {{MODEL}}"
    echo "Port: {{PORT}}"
    echo "GPU Memory Utilization: {{GPU_MEM}}"
    ./fix.sh uv run python -m vllm.entrypoints.openai.api_server \
        --model {{MODEL}} \
        --port {{PORT}} \
        --gpu-memory-utilization {{GPU_MEM}} \
        --max-model-len 1024 \
        --max-num-batched-tokens 1024 \
        --no-enable-chunked-prefill \
        --trust-remote-code

# Start vLLM server with LoRA adapters
serve-lora MODEL="Qwen/Qwen3-8B" PORT="8000" ADAPTERS="" GPU_MEM="0.5":
    #!/usr/bin/env bash
    echo "Starting vLLM server with LoRA adapters..."
    echo "Model: {{MODEL}}"
    echo "Port: {{PORT}}"
    echo "Adapters: {{ADAPTERS}}"
    echo "GPU Memory Utilization: {{GPU_MEM}}"

    LORA_ARGS=""
    if [ -n "{{ADAPTERS}}" ]; then
        # Split adapters by comma and create --lora-modules arguments
        IFS=',' read -ra ADAPTER_ARRAY <<< "{{ADAPTERS}}"
        for adapter in "${ADAPTER_ARRAY[@]}"; do
            adapter=$(echo "$adapter" | xargs) # trim whitespace
            LORA_ARGS="$LORA_ARGS --lora-modules $adapter=models/$adapter"
        done
    fi

    ./fix.sh uv run python -m vllm.entrypoints.openai.api_server \
        --model {{MODEL}} \
        --port {{PORT}} \
        --gpu-memory-utilization {{GPU_MEM}} \
        --max-model-len 1024 \
        --max-num-batched-tokens 1024 \
        --no-enable-chunked-prefill \
        --trust-remote-code \
        $LORA_ARGS

# Test vLLM server with a simple completion request
test-server PORT="8000":
    #!/usr/bin/env bash
    echo "Testing vLLM server at http://localhost:{{PORT}}/v1/completions..."
    curl -X POST "http://localhost:{{PORT}}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-8B",
            "prompt": "Hello, my name is",
            "max_tokens": 50,
            "temperature": 0.7
        }'

# Send a custom prompt to vLLM server using chat completions (accepts stdin if PROMPT is empty string)
prompt PROMPT="" PORT="8000":
    #!/usr/bin/env bash
    # If PROMPT is empty, read from stdin
    if [ -z "{{PROMPT}}" ]; then
        PROMPT_TEXT=$(cat)
    else
        PROMPT_TEXT="{{PROMPT}}"
    fi

    echo "Sending prompt to vLLM server at http://localhost:{{PORT}}/v1/chat/completions..."

    # Use jq to properly escape the prompt text for JSON
    # Parameters match Ollama's Qwen3 defaults: temp=0.6, top_k=20, top_p=0.95, repeat_penalty=1
    # Using chat completions endpoint for proper formatting
    # /no_think tag disables Qwen3's chain-of-thought reasoning
    JSON_PAYLOAD=$(jq -n \
        --arg model "Qwen/Qwen3-8B" \
        --arg content "$PROMPT_TEXT" \
        '{
            model: $model,
            messages: [
                {
                    role: "user",
                    content: ("/no_think\n" + $content)
                }
            ],
            max_tokens: 256,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            repetition_penalty: 1.0
        }')

    time curl -X POST "http://localhost:{{PORT}}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$JSON_PAYLOAD"

# Start vLLM server using official NVIDIA Docker image
serve-docker MODEL="Qwen/Qwen3-8B" PORT="8000" GPU_MEM="0.5":
    #!/usr/bin/env bash
    echo "Starting vLLM server in Docker..."
    echo "Model: {{MODEL}}"
    echo "Port: {{PORT}}"
    echo "GPU Memory Utilization: {{GPU_MEM}}"
    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -p {{PORT}}:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        nvcr.io/nvidia/vllm:25.10-py3 \
        vllm serve {{MODEL}} \
        --port 8000 \
        --gpu-memory-utilization {{GPU_MEM}} \
        --max-model-len 1024 \
        --max-num-batched-tokens 1024 \
        --no-enable-chunked-prefill \
        --trust-remote-code

build-unsloth:
    docker build -f Dockerfile.unsloth -t unsloth-dgx-spark .

# Clean dataset: remove continuation prompts and update system prompt
clean-dataset INPUT="data/gpt5nano/train.jsonl" OUTPUT="data/gpt5nano/train_cleaned.jsonl":
    uv run --no-project python scripts/data/clean_dataset.py --input {{INPUT}} --output {{OUTPUT}}

# Convert dataset to Unsloth format (conversations instead of messages)
convert-data-unsloth MERGE="--merge-system":
    uv run python scripts/data/convert_to_unsloth_format.py {{MERGE}}

# Generate synthetic training data
generate-data NUM="1000":
    uv run python scripts/data/generate_synthetic.py --num-examples {{NUM}}

# Train a single model
train CONFIG="configs/qlora-8b.yml":
    uv run python scripts/training/train_single.py --config {{CONFIG}}

# Train multiple models concurrently
train-multi PRESET="default":
    uv run python scripts/training/train_multi.py --preset {{PRESET}}

# Train using TRL directly (alternative to Axolotl, may work better on GB10)
train-trl:
    ./fix.sh uv run python scripts/training/train_trl_gemma3.py

# Generate DPO preference dataset (Phase 2 distillation)
# Configure via environment variables:
#   TEACHER=gpt-4o-mini BATCH=4 just generate-dpo-dataset
#   STUDENT_BACKEND=ollama STUDENT=gemma3-270m-student just generate-dpo-dataset
#   RESUME=1 just generate-dpo-dataset  # Resume from last checkpoint
generate-dpo-dataset:
    #!/usr/bin/env bash
    TEACHER="{{ env('TEACHER', 'gemma3:27b') }}"
    STUDENT_BACKEND="{{ env('STUDENT_BACKEND', 'unsloth') }}"
    STUDENT="{{ env('STUDENT', 'models/gemma3-270m-student-unsloth-v1') }}"
    INPUT="{{ env('INPUT', 'data/gpt5nano/train.jsonl') }}"
    OUTPUT="{{ env('OUTPUT', 'data/gpt5nano/train_dpo.jsonl') }}"
    BATCH="{{ env('BATCH', '2') }}"
    RESUME="{{ env('RESUME', '') }}"

    echo "Generating DPO preference dataset..."
    echo "Teacher: $TEACHER"
    echo "Student backend: $STUDENT_BACKEND"
    echo "Student: $STUDENT"
    echo "Input: $INPUT"
    echo "Output: $OUTPUT"
    echo "Batch size: $BATCH (use 1-2 for stability, 4-8 for speed if you have VRAM)"

    RESUME_FLAG=""
    if [ -n "$RESUME" ]; then
        RESUME_FLAG="--resume"
        echo "Resume mode: ON"
    fi

    uv run --no-project python scripts/distillation/generate_dpo_dataset.py \
        --teacher-backend auto \
        --teacher-model "$TEACHER" \
        --student-backend "$STUDENT_BACKEND" \
        --student-model "$STUDENT" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --temperature 0.7 \
        --batch-size "$BATCH" \
        --save-every 10 \
        --ollama-base-url http://localhost:11434 \
        $RESUME_FLAG

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
    EXISTING_DATA="{{ env('EXISTING_DATA', 'data/gpt5nano/train.jsonl') }}"
    OUTPUT="{{ env('OUTPUT', 'data/gpt5nano/train_dpo_extended.jsonl') }}"
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

# Generate validation DPO preference dataset
# Configure via environment variables:
#   TEACHER=gpt-4o-mini just generate-dpo-val
#   STUDENT_BACKEND=ollama STUDENT=gemma3-270m-student just generate-dpo-val
generate-dpo-val:
    #!/usr/bin/env bash
    TEACHER="{{ env('TEACHER', 'gemma3:27b') }}"
    STUDENT_BACKEND="{{ env('STUDENT_BACKEND', 'unsloth') }}"
    STUDENT="{{ env('STUDENT', 'models/gemma3-270m-student-unsloth-v1') }}"
    INPUT="{{ env('INPUT', 'data/gpt5nano/val.jsonl') }}"
    OUTPUT="{{ env('OUTPUT', 'data/gpt5nano/val_dpo.jsonl') }}"
    BATCH="{{ env('BATCH', '2') }}"

    echo "Generating validation DPO preference dataset..."
    echo "Teacher: $TEACHER"
    echo "Student backend: $STUDENT_BACKEND"
    echo "Student: $STUDENT"
    echo "Input: $INPUT"
    echo "Output: $OUTPUT"
    echo "Batch size: $BATCH"

    uv run --no-project python scripts/distillation/generate_dpo_dataset.py \
        --teacher-backend auto \
        --teacher-model "$TEACHER" \
        --student-backend "$STUDENT_BACKEND" \
        --student-model "$STUDENT" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --temperature 0.7 \
        --batch-size "$BATCH" \
        --save-every 10 \
        --ollama-base-url http://localhost:11434

# Inspect DPO preference dataset
inspect-dpo-dataset DATASET="data/gpt5nano/train_dpo_extended.jsonl":
    uv run --no-project python scripts/distillation/inspect_dpo_dataset.py {{DATASET}}

# Train with DPO (Phase 2 distillation - RL-based refinement)
# Configure via environment variables:
#   BASE=models/my-model DATASET=data/my_dpo.jsonl just train-dpo
#   VAL_DATASET=data/gpt5nano/val_dpo.jsonl just train-dpo  # Use DPO-formatted val set
train-dpo:
    #!/usr/bin/env bash
    BASE="{{ env('BASE', 'models/gemma3-270m-student-unsloth-v1_merged') }}"
    DATASET="{{ env('DATASET', 'data/gpt5nano/train_dpo.jsonl') }}"
    VAL_DATASET="{{ env('VAL_DATASET', 'data/gpt5nano/val_dpo.jsonl') }}"
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
unsloth-train IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    echo "Starting Unsloth training in Docker..."
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
        -w /workspace \
        -e WANDB_PROJECT=summarizer \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e UV_NO_PROJECT=1 \
        {{IMAGE}} \
        bash -c "rm -rf /workspace/unsloth_compiled_cache /tmp/torchinductor_* && python scripts/training/train_unsloth_gemma3.py"

# Export trained model to GGUF for Ollama
export-gguf MODEL="./models/gemma3-270m-student-unsloth-v1" NAME="gemma3-summary-v1" QUANT="Q4_K_M" IMAGE="spark-unsloth":
    #!/usr/bin/env bash
    echo "Exporting model to GGUF..."
    docker run --rm \
        --gpus=all \
        -v $(pwd):/workspace \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -w /workspace \
        {{IMAGE}} \
        python scripts/export/export_to_gguf.py --model-path {{MODEL}} --output-name {{NAME}} --quantization {{QUANT}}

# Import GGUF model into Ollama
ollama-import NAME="gemma3-summary-v1":
    #!/usr/bin/env bash
    echo "Importing {{NAME}} into Ollama..."

    # Check if the model directory exists
    MODEL_DIR="models/gguf/{{NAME}}"
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
    echo "Creating Ollama model: {{NAME}}"

    # Import from the model directory
    cd "$MODEL_DIR"
    ollama create {{NAME}} -f Modelfile
    cd - > /dev/null

    echo ""
    echo "✅ Model imported as: {{NAME}}"
    echo "Test with: ollama run {{NAME}} 'Fix the login bug'"

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

# Setup environment (run after cloning)
setup:
    @echo "Setting up environment..."
    @echo "1. Installing dependencies with vLLM build environment..."
    ./fix.sh uv sync
    @echo ""
    @echo "2. Creating .env file if it doesn't exist..."
    @[ -f .env ] || cp .env.example .env
    @echo ""
    @echo "3. Checking if model is downloaded..."
    @just check-model || echo "Run 'just download-model' to download the base model"
    @echo ""
    @echo "Setup complete! Edit .env to add your ANTHROPIC_API_KEY"

# Clean up Python cache and build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    rm -rf .pytest_cache .mypy_cache .coverage htmlcov

# Monitor GPU usage
gpu:
    watch -n 1 nvidia-smi
