#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm train 0421,在datasets/openarm_data_260306_260319_sft上训练,参照lerobot的sarm训练参数，"
DATE="0421_sarm"
STAGE1="sarm_train" 
STAGE1_TAG="0421"

source /mnt/data/syk/.bashrc
conda activate evo-rl


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



# sarm train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/0421_sarm_train.json" \
    --policy.image_downsample_antialias=true \
    --policy.image_downsample_size="[480, 640]" \
    --output_dir="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
    --dataset.repo_id="my/rollout" \
    --dataset.root="datasets/openarm_data_260306_260319_sft" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
