#!/bin/bash
set -e
echo "Job running on $(hostname)"

# Force set environment variables to prevent user lookup issues
export HOME=/tmp
export USER=condor_user
export OPENAI_CACHE_DIR=/tmp/openai_cache
export XDG_CACHE_HOME=/tmp/cache

# Disable PyTorch compilation to avoid Triton GPU requirements
export TORCH_COMPILE_DISABLE=1
export PYTORCH_DISABLE_CUDA_COMPILE=1

# Create necessary directories
mkdir -p /tmp/openai_cache
mkdir -p /tmp/cache
mkdir -p /tmp/.local

echo "Environment setup:"
echo "HOME=$HOME"
echo "USER=$USER" 
echo "OPENAI_CACHE_DIR=$OPENAI_CACHE_DIR"

# Create and activate virtual environment
python3 -m venv venv
. venv/bin/activate

# Upgrade pip and install setuptools first
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install dependencies from requirements.txt
pip install --no-cache-dir -r requirements.txt

# Run your code
python3 -u lora_main.py
