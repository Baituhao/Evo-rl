#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm train 0421,在datasets/openarm_data_260306_260319_sft上训练,参照lerobot的sarm训练参数，并在sft上infer，全量上infer"
DATE="0421_sarm_train&infer_workflow"
STAGE1="sarm_train" 
STAGE1_TAG="0421"

STAGE2="sarm_infer_on_sft" 
STAGE2_TAG="0421_sarm"

STAGE3="sarm_infer_on_full" 
STAGE3_TAG="0421_sarm"

source /mnt/data/syk/.bashrc
conda activate evo-rl


export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs ; DEBUG to enable ACP debug logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29501}"


mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"



# sarm train
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port="${MAIN_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="bash/config/0421_sarm_train.json" \
    --policy.image_downsample_antialias=true \
    --policy.image_downsample_size="[480, 640]" \
    --output_dir="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}" \
    --dataset.repo_id="my/rollout" \
    --dataset.root="datasets/openarm_data_260306_260319_sft" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE1} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE1} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE1="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}.log"

echo "====================================================================================================" >> "${LOG_FILE1}"
echo "${STAGE2} begin" >> "${LOG_FILE1}"
echo "====================================================================================================" >> "${LOG_FILE1}"
# value infer
setsid nohup \
    python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id="datasets/openarm_data_260306_260319_sft" \
    --reward-model-path="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}/checkpoints/005000/pretrained_model" \
    --head-mode="sparse" \
    --stride=9 \
    --num-visualizations=10 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260319_sft/sarm_progress_0421_sft.parquet" \
    >> "${LOG_FILE1}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE1})"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE2="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}.log"

echo "====================================================================================================" >> "${LOG_FILE2}"
echo "${STAGE3} begin" >> "${LOG_FILE2}"
echo "====================================================================================================" >> "${LOG_FILE2}"
# value infer
setsid nohup \
    python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id="datasets/openarm_data_260306_260330_sft_style_hg" \
    --reward-model-path="outputs/${DATE}/${STAGE1}_${STAGE1_TAG}_${TIMESTAMP}/checkpoints/005000/pretrained_model" \
    --head-mode="sparse" \
    --stride=9 \
    --num-visualizations=10 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE3}_${STAGE3_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_progress_0421_full.parquet" \
    >> "${LOG_FILE2}" 2>&1 &


echo "Started ${STAGE3} (logs: ${LOG_FILE2})"

source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE1}"
echo "${STAGE2} completed" >> "${LOG_FILE1}"
echo "====================================================================================================" >> "${LOG_FILE1}"

echo "====================================================================================================" >> "${LOG_FILE2}"
echo "${STAGE3} completed" >> "${LOG_FILE2}"
echo "====================================================================================================" >> "${LOG_FILE2}"