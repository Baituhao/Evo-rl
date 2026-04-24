#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：value01 实验的 0408，10k step， 用 policy 100k step "
DATE="0408_exp1_value01_10k"
STAGE1="value_train" 
STAGE1_TAG="value01-0407-bs8-10k"

STAGE2="value_infer" 
STAGE2_TAG="value01-0407-bs8-10k"

STAGE3="policy_train" 
STAGE3_TAG="policy-0408-bs8-100k-value01-0407-bs8-10k"

STAGE4="policy_rollout" 
STAGE4_TAG="policy-0408-bs8-100k-value01-0407-bs8-10k"

VALUE_INFER_TAG="value01-0407-bs8-10k"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/workflow_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE1} begin" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

# # 1. value train
# setsid nohup \
#     accelerate launch \
#     --multi_gpu \
#     --num_processes=8 \
#     --num_machines=1 \
#     --main_process_port="${MAIN_PORT}" \
#     -m lerobot.scripts.lerobot_value_train \
#     --config_path="bash/config/value_train_value01_0407.json" \
#     --value.type="value01" \
#     --dataset.root="outputs/merge_rollout_0_20260330_203044" \
#     --output_dir="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
#     >> "${LOG_FILE}" 2>&1 &

# echo "Started ${STAGE1} (logs: ${LOG_FILE})"
# source bash/wait_monitor.sh

# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE1} completed" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE2} begin" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# # 2. value infer
# setsid nohup \
#     lerobot-value-infer --config_path="bash/config/value_infer_default_config.json" \
#     --dataset.root="outputs/merge_rollout_0_20260330_203044" \
#     --inference.checkpoint_path="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}/checkpoints/010000/pretrained_model" \
#     --acp.advantage_field="complementary_info.advantage_${VALUE_INFER_TAG}" \
#     --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
#     --acp.value_field="complementary_info.value_${VALUE_INFER_TAG}" \
#     --output_dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
#     >> "${LOG_FILE}" 2>&1 &

# echo "Started ${STAGE2} (logs: ${LOG_FILE})"
# source bash/wait_monitor.sh

# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE2} completed" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE3} begin" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# # 3. policy train
# setsid nohup \
#     accelerate launch \
#     --multi_gpu \
#     --num_processes=2 \
#     --num_machines=1 \
#     --main_process_port="${MAIN_PORT}" \
#     -m lerobot.scripts.lerobot_train \
#     --config_path="bash/config/policy_train_default_config.json" \
#     --policy.compile_model=true \
#     --acp.enable=true \
#     --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
#     --dataset.omit_failed=false \
#     --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
#     --dataset.repo_id="my/rollout" \
#     --dataset.root="outputs/merge_rollout_0_20260330_203044" \
#     >> "${LOG_FILE}" 2>&1 &

# echo "Started ${STAGE3} (logs: ${LOG_FILE})"
# source bash/wait_monitor.sh

# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
# echo "${STAGE3} completed" >> "${LOG_FILE}"
# echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE4} begin" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"


# 4. policy eval 
setsid nohup \
    lerobot-eval --config_path="bash/config/policy_eval_default_config.json" \
    --policy.path="outputs/0408_exp1_value01_10k/policy_train_policy-0408-bs8-100k-value01-0407-bs8-10k_20260409_103432/checkpoints/100000/pretrained_model" \
    --output_dir="outputs/${DATE}/${STAGE4}_${STAGE4_TAG}_${TIMESTAMP}" \
    --acp.enable=true \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    >> "${LOG_FILE}" 2>&1 &

echo "Started ${STAGE4} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
echo "${STAGE4} completed" >> "${LOG_FILE}"
echo "$(printf '%*s' 80 | tr ' ' '=')" >> "${LOG_FILE}"
