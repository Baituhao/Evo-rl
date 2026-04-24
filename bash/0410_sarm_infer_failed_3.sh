
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="实验描述：sarm infer on failed 3"
DATE="0410_sarm_infer_failed"
STAGE2="0410_sarm_infer" 
STAGE2_TAG="0410_sarm_infer"
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
  --dataset-repo-id="datasets_oss/failed-0402-03002/2041348667125948417" \
  --reward-model-path="outputs/0410_sarm/sarm_train_0410_20260411_211319/checkpoints/050000/pretrained_model" \
  --episode-indices 0 \
  --head-mode="sparse" \
  --task "fold the towel" \
  --stride=9 \
  --output-dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




