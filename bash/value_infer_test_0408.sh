
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：value infer example，value train 2"
DATE="test"
STAGE2="value_infer" 
STAGE2_TAG="example"
VALUE_INFER_TAG="value_infer_test_tag"
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
    lerobot-value-infer --config_path="bash/config/value_infer_default_config.json" \
    --dataset.root="outputs/merge_rollout_0_20260330_203044_1" \
    --dataset.episodes=[10,20,23,46,51,54,57,67,70,73,86,100,106,115,143,149,179,187,194,204,214,270,283,288,306,314,316,341,376,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082] \
    --inference.checkpoint_path="outputs/value_train_2_20260330_214753/checkpoints/080000/pretrained_model" \
    --acp.advantage_field="complementary_info.advantage_${VALUE_INFER_TAG}" \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    --acp.value_field="complementary_info.value_${VALUE_INFER_TAG}" \
    --output_dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




