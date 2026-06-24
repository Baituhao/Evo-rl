# RISE-Style TD Loss Implementation Summary

## 概览

已成功将 `pistar_06_td` 的 TD loss 从**离线 bootstrap**（固定 MC target）改为 **RISE 风格的在线 bootstrap**（EMA target network），同时保持 CE loss（MC target 的分布式监督）不变。

## 改动文件清单

### 1. Configuration (`configuration_pistar_06_td.py`)
**改动量**: +15 行

**新增参数**:
```python
td_terminal_window: int = 10           # RISE 风格：最后 N 帧判定为终止区
td_success_reward: float = 1.0          # 成功 episode 终止奖励
td_failure_reward: float = -0.6         # 失败 episode 终止奖励
target_model_ema_decay: float = 0.995   # Target model EMA 更新速率
```

**删除参数**:
- `td_target_key` (不再需要预计算的 TD target)

**启用 delta_indices**:
```python
@property
def observation_delta_indices(self) -> dict[str, list[int]] | None:
    if self.td_loss_weight > 0:
        return {f: [0, 1] for f in self.camera_features}  # 当前帧 + 下一帧
    return None
```

---

### 2. Modeling (`modeling_pistar_06_td.py`)
**改动量**: +80 行（新增）, -80 行（删除）

#### A. 新增 Target Model (L568-577)
```python
# __init__ 中
if config.td_loss_weight > 0:
    self.target_model = copy.deepcopy(self.model)
    for param in self.target_model.parameters():
        param.requires_grad = False
    self.target_model.eval()
```

#### B. 新增 EMA 更新方法 (L753-763)
```python
def update_target_model(self):
    """Update target model using exponential moving average (EMA)."""
    if self.target_model is None:
        return
    decay = self.config.target_model_ema_decay
    with torch.no_grad():
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
```

#### C. 删除离线预计算逻辑
- 删除 `compute_td_targets_from_targets()` 函数 (原 L214-243)
- 删除 `build_training_raw_batch_hook` 里的 TD target 预计算（原 L808-829）

#### D. 重写 forward 里的 TD loss (L893-951)

**关键逻辑**:
1. 检测 images 是否有时间维度 (6D: `[B, N_delta, N_cam, C, H, W]`)
2. 提取下一帧: `next_images = images[:, 1]`
3. 构造稀疏 reward:
   ```python
   is_terminal = (episode_length - frame_index) <= td_terminal_window
   reward = is_terminal * is_failure * failure_reward + 
            is_terminal * (1 - is_failure) * success_reward
   ```
4. Target model 预测 V(s'):
   ```python
   with torch.no_grad():
       next_value = target_model(next_images, ...)
   ```
5. Bellman backup:
   ```python
   td_target = reward + gamma * (1 - done) * next_value
   ```

#### E. 适配 Model.forward 处理时间维度 (L490-501)
```python
if images.ndim == 6:
    images = images[:, 0]  # 提取当前帧
elif images.ndim != 5:
    raise ValueError(...)
```

---

### 3. Processor (`processor_pistar_06_td.py`)
**改动量**: +60 行

#### 适配多时间步图像处理

**`_to_bchw`** (L142-166):
- 支持 5D: `[B, N_delta, C, H, W]`

**`_resize_spatial`** (L168-188):
- 支持 5D resize: flatten B 和 N_delta → resize → unflatten

**`_prepare_images`** (L190-247):
- 检测 `has_temporal_dim`
- 堆叠后 transpose: `[B, N_cam, N_delta, ...] → [B, N_delta, N_cam, ...]`

---

### 4. Training Loop (`lerobot_value_train.py`)
**改动量**: +5 行

```python
# L67-71
if has_method(accelerator.unwrap_model(policy), "update_target_model"):
    accelerator.unwrap_model(policy).update_target_model()
```

每个训练步后调用 EMA 更新。

---

## 核心设计对比

| 维度 | 旧实现（离线） | 新实现（RISE 风格） |
|------|---------------|-------------------|
| **Reward** | 密集（每帧 r_t = V_MC(t) - V_MC(t+1)） | 稀疏（最后 10 帧: ±1/±0.6，其余 = 0） |
| **TD bootstrap** | 离线 MC: `V_MC(s_{t+1})` | 在线 target model: `V_θ'(s_{t+1})` |
| **CE loss** | MC target（保持不变） | MC target（保持不变） |
| **TD target 预计算** | ✅（hook 里一次算完） | ❌（forward 里实时计算） |
| **Target model** | ❌ | ✅ EMA shadow network |
| **Delta indices** | ❌ | ✅ 需要下一帧观测 |
| **TD signal 持续性** | 500 步后消失 | 整个训练中持续（target 偏差） |

---

## 训练配置调整

旧的 JSON config 无需大改，只需确保有这些 metadata 字段（dataloader 需要提供）:
```json
{
  "value": {
    "type": "pistar_06_td",
    "td_loss_weight": 1.0,
    "td_gamma": 0.99,
    "td_terminal_window": 10,
    "td_success_reward": 1.0,
    "td_failure_reward": -0.6,
    "target_model_ema_decay": 0.995
  }
}
```

Batch 需要包含:
- `frame_index`: 当前帧在 episode 中的索引
- `episode_length`: 当前 episode 的总长度
- `is_failure_data`: 是否为失败 episode (0.0 或 1.0)

---

## 测试验证

已通过合成数据测试验证：
- ✅ CE loss 正常 (5.21)
- ✅ TD loss 正常 (0.065)
- ✅ 稀疏 reward 逻辑正确
- ✅ Target model EMA 更新正常
- ✅ 下一帧观测正确传递

---

## 下一步

1. **真实数据测试**: 在 `advantige_dataset` 上运行一轮训练，确认 dataloader 正确提供 `frame_index/episode_length/is_failure_data`
2. **观察 TD loss 趋势**: 应该比离线版本持续更久（不会在 500 步就归零）
3. **对比性能**: 训练完成后对比策略效果，验证在线 TD 是否比离线版本更优

---

## 潜在问题排查

如果训练时报错：
- **KeyError: 'frame_index'** → dataset 没提供元信息，需要在 dataset 或 hook 里添加
- **Shape mismatch** → delta_indices 可能没正确启用，检查 processor 输出
- **TD loss = 0** → 检查 `images.ndim`，确认是 6D 而非 5D

---

## 总代码改动量

- Config: +15 行
- Modeling: +80 -80 = 净 0 行（但逻辑完全重写）
- Processor: +60 行
- Training loop: +5 行
- **总计: ~160 行改动**

符合预期的 150-300 行改动量。
