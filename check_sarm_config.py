#!/usr/bin/env python
"""检查 SARM 模型的配置参数"""

from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

model_path = "outputs/0605_sarm_train_infer_workflow/sarm_train_0605_20260611_013211/checkpoints/005000/pretrained_model"

try:
    model = SARMRewardModel.from_pretrained(model_path)
    config = model.config

    print("=" * 60)
    print("SARM 模型配置")
    print("=" * 60)
    print(f"n_obs_steps: {config.n_obs_steps}")
    print(f"frame_gap: {config.frame_gap}")
    print(f"max_rewind_steps: {config.max_rewind_steps}")
    print(f"\nobservation_delta_indices (前9个观察帧):")
    obs_indices = config.observation_delta_indices[:config.n_obs_steps + 1]
    print(f"  {obs_indices}")

    half_steps = config.n_obs_steps // 2
    print(f"\n双向采样:")
    print(f"  过去帧数: {half_steps}")
    print(f"  当前帧: 1")
    print(f"  未来帧数: {half_steps}")
    print(f"  总帧数: {half_steps * 2 + 1}")

    time_span = config.frame_gap * half_steps / 30.0
    print(f"\n时间跨度 (假设30fps):")
    print(f"  从 -{time_span:.1f} 秒到 +{time_span:.1f} 秒")
    print(f"  总共: {time_span * 2:.1f} 秒")

    print(f"\n边界影响:")
    print(f"  开头 {half_steps * config.frame_gap} 帧: 缺少完整的过去信息")
    print(f"  结尾 {half_steps * config.frame_gap} 帧: 缺少完整的未来信息")

    print("\n" + "=" * 60)

except Exception as e:
    print(f"错误: {e}")
    print("\n请检查模型路径是否正确")
