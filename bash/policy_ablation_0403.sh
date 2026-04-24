#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：policy ablation 0403 bs64-10k,在 原始专家数据集上训练，不带标签推理"
DATE="0403_ablation"

STAGE3="policy_train" 
STAGE3_TAG="bs64-10k-ablation"

STAGE4="policy_rollout" 
STAGE4_TAG="bs64-10k-ablation"

source /mnt/data/syk/.bashrc
conda activate evo-rl
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-29501}"

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/policy_ablation_0403_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"

# policy train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/policy_train_default_config.json" \
    --policy.compile_model=true \
    --acp.enable=false \
    --dataset.omit_failed=false \
    --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE3} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE4} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
# policy rollout 
setsid nohup \
    lerobot-rollout --config_path="bash/config/policy_rollout_default_config.json" \
    --policy.path="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}/checkpoints/010000/pretrained_model" \
    --output_dir="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}" \
    --acp.enable=false \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE4} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE4} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



