
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm infer on dataset  datasets/1_openarm_data_260306_260330_sft_style_hg 2300-2799"
DATE="0416_sarm_infer"
STAGE2="sarm_infer" 
STAGE2_TAG="0410_sarm"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"


echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"

# value infer
setsid nohup \
    python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id="datasets/openarm_data_260306_260330_sft_style_hg" \
    --reward-model-path="outputs/0410_sarm/sarm_train_0410_20260411_211319/checkpoints/050000/pretrained_model" \
    --episode-indices="$(seq -s, 2300 2799)" \
    --head-mode="sparse" \
    --stride=9 \
    --num-visualizations=0 \
    --num-workers=1 \
    --prefetch-size=16 \
    --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    --output-path="datasets/openarm_data_260306_260330_sft_style_hg/sarm_progress_2300_2799.parquet" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




