#!/bin/bash
# Script to set DGX Spark vLLM build environment variables
# Usage: ./fix.sh <command>
# Example: ./fix.sh uv sync

export TORCH_CUDA_ARCH_LIST=12.0
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export MAX_JOBS=16

# Run the command passed as arguments
exec "$@"
