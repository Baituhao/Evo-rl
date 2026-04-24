#!/bin/bash

# Multi-GPU distributed SARM RA-BC weight computation
# Uses Accelerate to distribute episodes across multiple GPUs

set -e

# Configuration
DATASET_REPO_ID="lerobot/aloha_sim_insertion_human"
REWARD_MODEL_PATH="pepijn223/sarm_single_uni4"
HEAD_MODE="sparse"  # or "dense" or "both"
STRIDE=5  # Compute every 5 frames, interpolate the rest
NUM_WORKERS=4  # Data loading threads per GPU
PREFETCH_SIZE=32
NUM_VISUALIZATIONS=5
OUTPUT_DIR="./sarm_viz"
MAIN_PORT=29500

# Multi-GPU settings
NUM_GPUS=8
NUM_MACHINES=1

echo "Starting distributed SARM RA-BC computation..."
echo "Dataset: ${DATASET_REPO_ID}"
echo "Model: ${REWARD_MODEL_PATH}"
echo "GPUs: ${NUM_GPUS}"
echo "Stride: ${STRIDE}"
echo "Workers per GPU: ${NUM_WORKERS}"

accelerate launch \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    --num_machines=${NUM_MACHINES} \
    --main_process_port=${MAIN_PORT} \
    src/lerobot/policies/sarm/compute_rabc_weights_distributed.py \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --reward-model-path "${REWARD_MODEL_PATH}" \
    --head-mode "${HEAD_MODE}" \
    --stride ${STRIDE} \
    --num-workers ${NUM_WORKERS} \
    --prefetch-size ${PREFETCH_SIZE} \
    --num-visualizations ${NUM_VISUALIZATIONS} \
    --output-dir "${OUTPUT_DIR}"

echo "Done! Check ${OUTPUT_DIR} for visualizations."
