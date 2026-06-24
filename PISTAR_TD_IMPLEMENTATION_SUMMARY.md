# Pistar_06_td TD Loss 实现总结（TL;DR）

> 完整文档见 [PISTAR_TD_IMPLEMENTATION_GUIDE.md](./PISTAR_TD_IMPLEMENTATION_GUIDE.md)

## 一句话总结

在 Pistar 06 value model 上**增量添加** RISE-style online TD bootstrap loss，通过 EMA target network 提供独立于 MC 的持续监督信号。

---

## 核心修改（3 个文件）

### 1. `src/lerobot/datasets/factory.py` L130-132

**问题**：硬编码 `cfg.policy`，value training 时读不到 `observation_delta_indices`

**修复**：
```python
model_cfg = getattr(cfg, 'value', None) or cfg.policy
delta_timestamps = resolve_delta_timestamps(model_cfg, ds_meta)
```

### 2. `src/lerobot/scripts/lerobot_value_train.py` L258-268

**问题**：Preprocessor 删除 `frame_index`, `episode_length` 等 TD metadata

**修复**：
```python
batch = value_target_raw_batch_hook(batch, step)  # Hook 注入
td_metadata = {k: batch.get(k) for k in ['frame_index', 'episode_length', 'is_failure_data'] if k in batch}
batch = preprocessor(batch)
batch.update(td_metadata)  # 恢复
```

### 3. `src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py`

**新增功能**：
- L576-582：初始化 `target_model = deepcopy(model)`
- L726-734：每步 EMA 更新 `θ_target ← 0.995·θ_target + 0.005·θ_main`
- L976-1024：计算 TD loss `MSE(V_main(s), r + γ·V_target(s'))`

---

## EMA Target Network 原理（5 点）

### 1. 为什么需要 Target Network？

**问题**：TD learning 的"移动目标"
```
y_td = r + γ·V(s')  ← V(s) 和 V(s') 是同一个网络
更新 V(s) → V(s') 也变 → target 跟着动 → 训练不稳定
```

**解决**：用滞后副本计算 target
```
V_main(s) ← 梯度更新（快）
V_target(s') ← EMA 追踪（慢，滞后 ~200 步）
y_td = r + γ·V_target(s')  ← 短期内稳定
```

### 2. EMA 更新公式

```python
# 每个训练步执行一次
θ_target ← 0.995·θ_target + 0.005·θ_main

# 等价写法（RISE 原版）
θ_target ← (1-τ)·θ_target + τ·θ_main, τ=0.005
```

### 3. 有效记忆窗口

```
θ_target^(t) = Σ_{k=0}^{t} 0.005 × 0.995^k × θ_main^(t-k)

窗口 N = 1/τ = 1/0.005 = 200 步

含义：target 是 main 过去 ~200 步的指数加权平均
```

### 4. 为什么持续监督？

```
步 1000: V_main 拟合 MC → CE loss 饱和
        但 V_target 还是步 900 的参数
        
步 1001: V_main 继续优化 → V_target 刚追到步 901

TD_loss = (V_main - (r + γ·V_target))² ≠ 0 ✓

只要训练继续，target 永远滞后 → TD error 持续非零
```

### 5. 与硬更新对比

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **硬更新** (DQN 2013) | 每 N 步：`θ_t ← θ_m` | 简单 | 突变，不稳定 |
| **EMA** (DDPG 2015+) | 每步：`θ_t ← 0.995·θ_t + 0.005·θ_m` | 平滑，鲁棒 | （无明显缺点） |

**结论**：EMA 是现代 RL 标准（TD3/SAC/RISE 都用）

---

## 数据流（4 步）

```
1. Dataset Factory
   cfg.value.observation_delta_indices = [0, 1]
   → delta_timestamps = {obs.state: [0.0, 0.033], obs.images: [0.0, 0.033]}
   → Dataset 返回 [B, 2, ...] (dual-frame)

2. Processor
   images: [B, 2, N_cam, C, H, W]
   state: [B, 2, D] → 生成 2B prompts (前 B 个 current, 后 B 个 next)

3. Training Hook
   注入 episode_length, frame_index, is_failure_data

4. Forward
   检测 is_dual_frame → 分离 current/next
   → V_main(s), V_target(s') → TD loss
```

---

## TD Loss 计算（稀疏 reward）

```python
# 距离 episode 结束的帧数
frames_to_end = episode_length - frame_index

# 终止窗口掩码（最后 10 帧）
is_terminal = (frames_to_end <= 10).float()

# 稀疏 reward
reward = is_terminal * (success * 1.0 + failure * (-0.6))

# Bellman backup
td_target = reward + 0.99 * (1 - done) * V_target(s')

# TD loss
td_loss = MSE(V_main(s), td_target)

# 总损失
loss = ce_loss + 1.0 * td_loss
```

**设计**（RISE-style）：
- 非终止帧：`reward=0`，纯 bootstrap
- 终止窗口（最后 10 帧）：注入稀疏 reward
- 最后一帧：`done=1`，截断 bootstrap

---

## 验证清单

### ✅ 正常日志应包含

```bash
# 1. Hook 注入成功
INFO: Hook injected TD metadata: episode_length shape=torch.Size([16]), frame_index shape=torch.Size([16])

# 2. Dual-frame 检测通过
WARNING: TD loss check: frame_index=True, episode_length=True, is_failure=True

# 3. TD loss 非零
INFO: step:50 loss:5.290 ce:5.271 td:0.020 grdn:6.960
INFO: step:100 loss:4.920 ce:4.887 td:0.033 grdn:6.346
```

### ❌ 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `td:0.000` | `is_dual_frame=False` | 检查 factory.py 是否用 `cfg.value` |
| `frame_index=False` | Metadata 被删除 | 检查 train script 是否保存/恢复 |
| TD 持续上升 | Reward 超出 bin 范围 | 改 `success_reward=0.0, failure_reward=-1.0` |

---

## 配置参数（JSON）

```json
{
  "value": {
    "type": "pistar_06_td",
    "td_loss_weight": 1.0,           // TD 权重（0=禁用）
    "td_gamma": 0.99,                // 折扣因子
    "td_terminal_window": 10,        // 终止窗口帧数
    "td_success_reward": 1.0,        // 成功 reward ⚠️ 建议改 0.0
    "td_failure_reward": -0.6,       // 失败 reward
    "target_model_ema_decay": 0.995, // EMA 衰减率（对应 τ=0.005）
    
    "bin_min": -1.0,
    "bin_max": 0.0,                  // Value 支持范围
    ...
  }
}
```

---

## 下一步优化

### 1. 调整 reward 尺度（重要）

**当前问题**：`td_success_reward=1.0` 超出 `[bin_min, bin_max]=[-1, 0]`

**解决方案 A**（推荐）：对齐 value 范围
```json
"td_success_reward": 0.0,   // MC target ≈ 0（最好结果）
"td_failure_reward": -1.0,  // MC target ≈ -1（最坏结果）
```

**解决方案 B**：扩展 bin 范围
```json
"bin_min": -1.0,
"bin_max": 1.0,  // 允许正 value
```

### 2. 调整 TD 权重

**当前**：`td_loss_weight=1.0` → TD 只占总 loss ~1%（太小）

**建议**：
```json
"td_loss_weight": 10.0,  // TD 占 ~10%，真正影响训练
```

或扩大终止窗口：
```json
"td_terminal_window": 50,  // 增加稀疏 reward 覆盖率
```

### 3. 监控 EMA 效果（可选）

在 `update_target_model()` 中添加：
```python
param_distance = sum((p - tp).pow(2).sum().item() 
                     for p, tp in zip(main_params, target_params))
rms_distance = (param_distance / total_params) ** 0.5
logging.info(f"Target lag: {rms_distance:.6f}")
```

预期：距离在 200 步后稳定在 ~0.006 量级

---

## 与 RISE 原版对比

| 维度 | RISE (JAX/Flax) | 我们 (PyTorch) |
|------|-----------------|---------------|
| **Value model** | PaliGemma | Pistar 06 (SigLIP+Gemma) |
| **EMA 公式** | `(1-τ)·old + τ·new` | `decay·old + (1-decay)·new` |
| **数学等价** | ✅ τ=0.005 | ✅ decay=0.995 |
| **Reward 设定** | `success=1.0, failure=-1.0` | `success=1.0, failure=-0.6` |
| **架构清晰度** | 复制整个 policy | 只复制 `self.model` ✅ 更简洁 |
| **调试友好度** | 基础 | 增加 dual-frame/metadata 检查 ✅ |

---

## 参考文献

- **RISE** (2024): [Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967)
- **DDPG** (2015): Continuous control with deep reinforcement learning (首次引入 soft target update)
- **TD3** (2018): Addressing Function Approximation Error (τ=0.005 标准配置)
- **SAC** (2018): Soft Actor-Critic (同样采用 τ=0.005)

---

**生成时间**：2026-06-17  
**版本**：v1.0  
**作者**：Kiro (Claude Code)
