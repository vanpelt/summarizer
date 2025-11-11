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
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
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
        -e HF_TOKEN=${HF_TOKEN:-} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
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
ollama-import NAME="gemma3-summary-v1" GGUF="":
    #!/usr/bin/env bash
    echo "Importing {{NAME}} into Ollama..."

    # If GGUF file specified, use it; otherwise find any .gguf file
    if [ -n "{{GGUF}}" ]; then
        GGUF_FILE="{{GGUF}}"
    else
        GGUF_FILE=$(ls *.gguf 2>/dev/null | head -n 1)
    fi

    if [ -z "$GGUF_FILE" ] || [ ! -f "$GGUF_FILE" ]; then
        echo "Error: GGUF file not found."
        echo "Available GGUF files:"
        ls -lh *.gguf 2>/dev/null || echo "  (none)"
        echo ""
        echo "Run 'just export-gguf' to create one, or specify: just ollama-import NAME GGUF=file.gguf"
        exit 1
    fi

    echo "Using GGUF file: $GGUF_FILE"
    echo "Creating Ollama model: {{NAME}}"

    ollama create {{NAME}} -f Modelfile
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
