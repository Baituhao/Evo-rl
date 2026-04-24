#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：pistar06 train on openarm_data_260306_260319_sft"
DATE="0410_pistar06"
STAGE1="value_train" 
STAGE1_TAG="0410_pistar06"
VALUE_INFER_TAG="0410_pistar06"
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


    # --multi_gpu \
    # --num_processes=8 \
# value train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_value_train \
    --config_path="bash/config/value_train_pistar06_0410.json" \
    --value.type="pistar06" \
    --dataset.root="datasets/openarm_data_260306_260319_sft" \
    --output_dir="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


