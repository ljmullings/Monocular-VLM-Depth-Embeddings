#!/bin/bash

# Training script for distance head

# Set default GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Default config
CONFIG=${1:-"configs/base.yaml"}

echo "Training distance head..."
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create run directory
RUN_DIR="runs/train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# Run training
python bin/train_head.py \
    --config "$CONFIG" \
    --output_dir "$RUN_DIR" \
    --log_dir "$RUN_DIR/logs"

echo "Training completed. Results saved to: $RUN_DIR"
