#!/bin/bash

# Set PS3 environment variables
export NUM_LOOK_CLOSE=${NUM_LOOK_CLOSE:-2}
export NUM_TOKEN_LOOK_CLOSE=${NUM_TOKEN_LOOK_CLOSE:-2048}
export SELECT_NUM_EACH_SCALE=${SELECT_NUM_EACH_SCALE:-"256+512"}

# Default values
CONFIG=${1:-"configs/base.yaml"}
IMAGE=${2:-"assets/sample.jpg"}
TEXT=${3:-"How far is the red car?"}

echo "Running MVDE inference..."
echo "Config: $CONFIG"
echo "Image: $IMAGE" 
echo "Text: $TEXT"
echo "PS3 Settings: NUM_LOOK_CLOSE=$NUM_LOOK_CLOSE, NUM_TOKEN_LOOK_CLOSE=$NUM_TOKEN_LOOK_CLOSE"

# Run inference
python bin/run_infer.py \
    --config "$CONFIG" \
    --image "$IMAGE" \
    --text "$TEXT" \
    --output "runs/inference_$(date +%Y%m%d_%H%M%S).json"
