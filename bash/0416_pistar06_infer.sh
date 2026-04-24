
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DESC="pistar06 infer on  datasets/1_openarm_data_260306_260330_sft_style_hg 2300-2799"
DATE="0416_pistar06_infer"
STAGE2="value_infer" 
STAGE2_TAG="0410_pistar06_80k"
VALUE_INFER_TAG="0410_pistar06_80k"
source /mnt/data/syk/.bashrc
conda activate evo-rl

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export LEROBOT_LOG_LEVEL="INFO"  # set to INFO to disable ACP debug logs

mkdir -p "outputs/${DATE}"
LOG_FILE="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}.log"
echo "[${TIMESTAMP}] ${EXPERIMENT_DESC}" >> "${LOG_FILE}"

echo "copy datasets to datasets"
cp -r datasets/openarm_data_260306_260330_sft_style_hg datasets/1_openarm_data_260306_260330_sft_style_hg
echo "datasets copied"

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} begin" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"
# value infer
setsid nohup \
    lerobot-value-infer --config_path="bash/config/value_infer_default_config.json" \
    --dataset.root="datasets/1_openarm_data_260306_260330_sft_style_hg" \
    --dataset.episodes="$(printf "[%s]" "$(seq -s, 2300 2799)")" \
    --viz.enable=false \
    --inference.checkpoint_path="outputs/0410_pistar06/value_train_0410_pistar06_20260415_135052/checkpoints/080000/pretrained_model" \
    --acp.advantage_field="complementary_info.advantage_${VALUE_INFER_TAG}" \
    --acp.indicator_field="complementary_info.acp_indicator_${VALUE_INFER_TAG}" \
    --acp.value_field="complementary_info.value_${VALUE_INFER_TAG}" \
    --output_dir="outputs/${DATE}/${STAGE2}_${STAGE2_TAG}_${TIMESTAMP}" \
    >> "${LOG_FILE}" 2>&1 &


echo "Started ${STAGE2} (logs: ${LOG_FILE})"
source bash/wait_monitor.sh

echo "====================================================================================================" >> "${LOG_FILE}"
echo "${STAGE2} completed" >> "${LOG_FILE}"
echo "====================================================================================================" >> "${LOG_FILE}"




