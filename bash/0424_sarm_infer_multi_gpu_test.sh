
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm-0410 infer on dataset datasets/openarm_data_260306_260319_sft,multi gpu test"
DATE="0424_sarm_infer_multi_gpu_test"
STAGE2="sarm_infer" 
STAGE2_TAG="0410_sarm"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PORT="${MAIN_PORT:-29389}"

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
# value infer
setsid nohup \
    accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port=${MAIN_PORT} \
    -m lerobot.policies.sarm.compute_rabc_weights \
    --dataset-repo-id="datasets/openarm_data_260306_260330_sft_style_hg" \
    --reward-model-path="outputs/0410_sarm/sarm_train_0410_20260411_211319/checkpoints/050000/pretrained_model" \
    --head-mode="sparse" \
    --stride=1 \
    --num-visualizations=10 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_progress_0410_multi_gpu_test.parquet" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




