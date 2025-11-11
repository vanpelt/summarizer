#!/bin/bash
#
export TORCH_CUDA_ARCH_LIST="12.0"
uv venv --system-site-packages /venv

uv sync --active

uv run python -m ipykernel install --user --name venv --display-name "Python (.venv)"
