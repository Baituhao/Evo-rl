#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：0420 policy train with sarm rabc subtasks inference test"
DATE="0420_workflow_test"

STAGE2="sarm_infer" 
STAGE2_TAG="0417_sarm"

STAGE3="policy_train" 
STAGE3_TAG="pi05_200k_sarm_rabc_subtasks_32bs"

source /mnt/data/syk/.bashrc
conda activate evo-rl-pi


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}.log"
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
    --episode-indices="$(seq -s, 100 101)" \
    --num-visualizations=10 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260319_sft_with_subtask/sarm_progress_subtasks_0417.parquet" \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



# echo "====================================================================================================" >> "${LOG_FILE}"
# echo "${STAGE3} begin" >> "${LOG_FILE}"
# echo "====================================================================================================" >> "${LOG_FILE}"




# # policy train
# setsid nohup \
#     accelerate launch \
#     --multi_gpu \
#     --num_processes=8 \
#     --num_machines=1 \
#     --main_process_port="${MAIN_PORT}" \
#     -m lerobot.scripts.lerobot_train \
#     --config_path="bash/config/0416_sarm_rabc_policy_train.json" \
#     --policy.compile_model=true \
#     --rabc_progress_path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_progress_subtasks_0417.parquet" \
#     --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
#     --dataset.repo_id="my/rollout" \
#     --dataset.root="datasets/openarm_data_260306_260330_sft_style_hg" \
#     >> "${LOG_FILE}" 2>&1 &


# echo "Started ${STAGE3} (logs: ${LOG_FILE})"
# source bash/wait_monitor.sh

# echo "====================================================================================================" >> "${LOG_FILE}"
# echo "${STAGE3} completed" >> "${LOG_FILE}"
# echo "====================================================================================================" >> "${LOG_FILE}"


