
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm_0421_lerobot_config_trained_infer_on_full infer on dataset datasets/openarm_data_260306_260319_sft,"
DATE="0424_sarm_0421_lerobot_config_infer_on_full"
STAGE2="sarm_infer" 
STAGE2_TAG="sarm_0421_lerobot_config"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
# value infer
setsid nohup \
    python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id="datasets/openarm_data_260306_260330_sft_style_hg" \
    --reward-model-path="outputs/0421_sarm_train&infer_workflow/sarm_train_0421_20260422_105514/checkpoints/005000/pretrained_model" \
    --head-mode="sparse" \
    --stride=9 \
    --num-visualizations=10 \
    --prefetch-size=32 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_0421_lerobot_config_full_progress.parquet" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




