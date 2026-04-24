#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm annotation"
DATE="0417_sarm_annotation"
STAGE1="sarm_annotation" 
STAGE1_TAG="0417_test"

source /mnt/data/syk/.bashrc
conda activate evo-rl


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


sarm annotation
setsid nohup \
    python analysis/convert_frame_subtasks_to_sarm_annotations.py \
    --dataset-root datasets/openarm_data_260306_260319_sft_with_subtask \
    --sparse-mode mirror_dense \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh


# sarm verify 
setsid nohup \
    python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
    --dataset-root datasets/openarm_data_260306_260319_sft_with_subtask \
    --visualize-only \
    --visualize-type both \
    --num-visualizations 5 \
    --video-key observation.images.head \
    --output-dir outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP} \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"

