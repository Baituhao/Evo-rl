#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：0416 policy train with sarm rabc，batch size 32 * 8 GPUs * 100k steps, num_workers=2"
DATE="0416_workflow_resume_2nw"

STAGE2="sarm_infer" 
STAGE2_TAG="0410_sarm"

STAGE3="policy_train" 
STAGE3_TAG="pi05_200k_sarm_rabc"

source /mnt/data/syk/.bashrc
conda activate evo-rl-pi


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"








# echo "====================================================================================================" >> "${LOG_FILE}"
# echo "${STAGE2} begin" >> "${LOG_FILE}"
# echo "====================================================================================================" >> "${LOG_FILE}"
# # value infer
# setsid nohup \
#     python src/lerobot/policies/sarm/compute_rabc_weights.py \
#     --dataset-repo-id="datasets/openarm_data_260306_260330_sft_style_hg" \
#     --reward-model-path="outputs/0410_sarm/sarm_train_0410_20260411_211319/checkpoints/050000/pretrained_model" \
#     --head-mode="sparse" \
#     --stride=9 \
#     --num-visualizations=10 \
#     --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
#     >> "${LOG_FILE}" 2>&1 &


# echo "Started ${STAGE2} (logs: ${LOG_FILE})"
# source bash/wait_monitor.sh

# echo "====================================================================================================" >> "${LOG_FILE}"
# echo "${STAGE2} completed" >> "${LOG_FILE}"
# echo "====================================================================================================" >> "${LOG_FILE}"



echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




# policy train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/0416_sarm_rabc_policy_train.json" \
    --policy.compile_model=true \
    --use_rabc=true \
    --num_workers=2 \
    --rabc_progress_path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_progress.parquet" \
    --output_dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
    --dataset.repo_id="my/rollout" \
    --dataset.root="datasets/openarm_data_260306_260330_sft_style_hg" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE3} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE3} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"


