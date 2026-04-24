#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：voc 计算 pistar06 0410,1400-1899 episodes"
DATE="0421_voc_pistar06"
STAGE1="voc_compute" 
STAGE1_TAG="pistar06_410"

source /mnt/data/syk/.bashrc
conda activate evo-rl


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



# voc compute
setsid nohup \
    python /mnt/data/syk/Evo-RL/analysis/compute_voc_from_pistar06_progress.py \
    --progress-parquet "datasets/1_openarm_data_260306_260319_sft_style_hg/data/chunk-000/file-000.parquet" \
    --value-column "complementary_info.value_0410_pistar06_80k" \
    --episodes "1400-1899" \
    --output-dir "outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


