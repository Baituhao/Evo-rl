
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：value viz example"
DATE="test"
STAGE2="value_viz" 
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
    python -m lerobot.scripts.lerobot_value_indicator_show \
    --config_path="bash/config/value_viz_default_config.json" \
    --dataset.root="outputs/policy_rollout_1_1_20260330_165819/dataset" \
    --inference.checkpoint_path="outputs/value_train_2_20260330_214753/checkpoints/080000/pretrained_model" \
    --dataset.episodes=0 \
    --viz.episodes=0 \
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


