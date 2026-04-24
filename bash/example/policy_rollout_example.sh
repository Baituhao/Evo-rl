#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：policy rollout example"
DATE="test"
STAGE4="policy_rollout" 
STAGE4_TAG="example"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE4} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
# policy rollout 
setsid nohup \
    lerobot-rollout --config_path="bash/config/policy_rollout_default_config.json" \
    --policy.path="outputs/policy_train_0401_20260401_153418/checkpoints/120000/pretrained_model" \
    --output_dir="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}" \
    --acp.enable=true \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE4} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE4} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



