FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch cu118 first
RUN pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118


# Install app dependencies
RUN pip install \
    transformers==4.28.1 \
    accelerate==0.19.0 \
    fastapi==0.95.1 \
    gradio==3.23.0 \
    httpx==0.24.0 \
    markdown-it-py==2.2.0 \
    numpy==1.24.3 \
    prompt-toolkit==3.0.38 \
    pydantic==1.10.7 \
    requests==2.30.0 \
    rich==13.3.5 \
    rouge==1.0.1 \
    sentencepiece==0.1.99 \
    shortuuid==1.0.11 \
    tiktoken==0.4.0 \
    tokenizers==0.13.3 \
    uvicorn==0.22.0 \
    bitsandbytes==0.38.1 \
    peft==0.3.0 \
    langchain==0.0.229 \
    deepspeed==0.9.2 \
    sentence_transformers==2.2.2 \
    tensorboard \
    openai \
    scipy \
    termcolor \
    flask \
    flask_cors

# Optional quick sanity check
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"




CMD ["bash"]
