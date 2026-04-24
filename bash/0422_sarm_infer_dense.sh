
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm-0417 infer on datasets/openarm_data_260306_260319_sft_with_subtask,"
DATE="0422_sarm_infer"
STAGE2="sarm_infer_dense_test" 
STAGE2_TAG="sarm_0417"
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
    --dataset-repo-id="datasets/openarm_data_260306_260319_sft_with_subtask" \
    --reward-model-path="outputs/0417_sarm/sarm_train_0417_test_20260417_202855/checkpoints/050000/pretrained_model" \
    --head-mode="dense" \
    --stride=9 \
    --episode-indices 0 \
    --num-visualizations=1 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260319_sft_with_subtask/sarm_progress_0417_dense_test.parquet" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




