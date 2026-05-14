#!/bin/bash

# 1. Initialize Conda (This allows 'conda activate' to work inside a script)
# This finds where conda is installed and sources the profile script
echo "--- Creating Conda Environment ---"
conda create -n multi_llm_agent python=3.10 -y

echo "--- Activating Environment ---"
conda activate multi_llm_agent

echo "--- Installing PyTorch 2.0.1 with CUDA 12.1 ---"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

echo "--- Installing Requirements ---"
# Ensure your requirements.txt is in the same folder
pip install -r requirements.txt

echo "--- Verification ---"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
