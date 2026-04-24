#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：policy eval example"
DATE="test"
STAGE4="policy_eval" 
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
# policy eval 
setsid nohup \
    lerobot-eval --config_path="bash/config/policy_eval_default_config.json" \
    --policy.path="outputs/0403_value01_rectify/policy_train_0403-rectify-bs64-10k_20260408_110305/checkpoints/010000/pretrained_model" \
    --output_dir="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}" \
    --acp.enable=true \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE4} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE4} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



