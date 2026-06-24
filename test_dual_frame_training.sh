#!/bin/bash
# Quick single-GPU test before full 8-GPU training
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE="test_dual_frame"

source /mnt/data/syk/.bashrc
conda activate evo-rl-pi

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"
export CUDA_VISIBLE_DEVICES="0"

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/test_${TIMESTAMP}.log"

echo "Testing dual-frame RISE TD implementation (single GPU, 3 steps)..."
python -m lerobot.scripts.lerobot_value_train \
    --config_path="bash/config/0616_pistar06_td_value_train.json" \
    --value.type="pistar_06_td" \
    --dataset.root="/mnt/cpfs/syk/datasets/advantige_dataset/20260601103414_20260425-20260528" \
    --dataset.image_center_crop="[800,800]" \
    --batch_size=2 \
    --num_workers=1 \
    --steps=3 \
    --output_dir="outputs/${DATE}/test_${TIMESTAMP}" \
    2>&1 | tee "${LOG_FILE}"

echo
echo "Check log for:"
echo "  1. 'loss_td' appears in metrics (TD loss computed)"
echo "  2. No OOM errors"
echo "  3. Shapes logged correctly"
echo
echo "If passed, run full training with bash/0616_pistar06_td_value_train.sh"
