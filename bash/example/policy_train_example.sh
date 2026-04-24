#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：policy train example"
DATE="test"
STAGE3="policy_train" 
STAGE3_TAG="example"
VALUE_INFER_TAG="pistar06_80k"
source /mnt/data/syk/.bashrc
conda activate evo-rl-pi


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"

# policy train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/policy_train_default_config.json" \
    --policy.compile_model=true \
    --acp.enable=true \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    --dataset.omit_failed=false \
    --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
    --dataset.repo_id="my/rollout" \
    --dataset.root="outputs/merge_rollout_0_20260330_203044" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE3} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


