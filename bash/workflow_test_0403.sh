#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：value train example"
DATE="test"
STAGE1="value_train" 
STAGE1_TAG="example"

STAGE2="value_infer" 
STAGE2_TAG="example"

STAGE3="policy_train" 
STAGE3_TAG="example"

STAGE4="policy_rollout" 
STAGE4_TAG="example"

VALUE_INFER_TAG="value_infer_test_tag"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/workflow_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE1} begin" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

# 1. value train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_value_train \
    --config_path="bash/config/value_train_default_config.json" \
    --value.type="value01" \
    --steps=1000 \
    --dataset.root="outputs/policy_rollout_1_1_20260330_165819/dataset" \
    --output_dir="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE2} begin" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# 2. value infer
setsid nohup \
    lerobot-value-infer --config_path="bash/config/value_infer_default_config.json" \
    --dataset.root="outputs/policy_rollout_1_1_20260330_165819/dataset" \
    --inference.checkpoint_path="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}/checkpoints/001000/pretrained_model" \
    --acp.advantage_field="complementary_info.advantage_${VALUE_INFER_TAG}" \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    --acp.value_field="complementary_info.value_${VALUE_INFER_TAG}" \
    --output_dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE3} begin" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# 3. policy train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/policy_train_default_config.json" \
    --policy.compile_model=false \
    --acp.enable=true \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    --steps=100 \
    --dataset.omit_failed=false \
    --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
    --dataset.repo_id="my/rollout" \
    --dataset.root="outputs/policy_rollout_1_1_20260330_165819/dataset" \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE3} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE3} completed" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE4} begin" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# 4. policy rollout 
setsid nohup \
    lerobot-rollout --config_path="bash/config/policy_rollout_default_config.json" \
    --policy.path="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}/checkpoints/000100/pretrained_model" \
    --output_dir="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}" \
    --acp.enable=true \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE4} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE4} completed" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
