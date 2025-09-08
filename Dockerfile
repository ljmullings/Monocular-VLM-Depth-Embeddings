FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy environment and requirements
COPY environment.yml pyproject.toml ./

# Install conda environment
RUN conda env update -n base -f environment.yml

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_DIR=/workspace/runs

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/runs /workspace/.cache

# Expose ports for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]
