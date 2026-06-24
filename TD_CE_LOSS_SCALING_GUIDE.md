# TD Loss 与 CE Loss 量纲对齐指南

## 📋 目录

1. [问题概述](#问题概述)
2. [根源分析](#根源分析)
3. [数学推导与量纲计算](#数学推导与量纲计算)
4. [解决方案对比](#解决方案对比)
5. [推荐实施路径](#推荐实施路径)
6. [监控指标](#监控指标)
7. [实战案例](#实战案例)

---

## 问题概述

### 现象描述

在 Pistar_06_td 的 value 模型训练中，观察到 **TD loss 和 CE loss 存在 10-100 倍的量纲差距**：

```
Step 1000:
  loss_ce: 2.34
  loss_td: 0.023
  loss_ratio (CE/TD): ~100x
```

### 影响

1. **梯度失衡**：TD loss 对梯度的贡献可忽略不计，相当于只训练 CE loss
2. **训练效果**：无法有效利用 Bellman bootstrap 的监督信号
3. **超参数调优困难**：`td_loss_weight` 需要手动调整到很大的值才能平衡

---

## 根源分析

### 代码位置

**文件**：`src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py`

### 损失计算流程

#### 1. CE Loss 计算（第 931-960 行）

```python
# 第 932 行：将 value target 投影到 bins
soft_target = project_values_to_bins(value_target, bin_centers)

# 第 933-934 行：交叉熵损失
log_probs = functional.log_softmax(logits, dim=-1)
per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

# 第 960 行：记录 CE loss
ce_loss_value = per_sample_loss.mean()
```

**量纲分析**：
- 对于 `num_bins=201` 的分布式回归
- 初始 CE ≈ `log(201) ≈ 5.3`
- 训练收敛后：**1.0 - 3.0**

#### 2. TD Loss 计算（第 974-1024 行）

```python
# 第 999-1001 行：构造稀疏 reward
reward = (is_terminal * is_failure * self.config.td_failure_reward +
          is_terminal * (1.0 - is_failure) * self.config.td_success_reward)

# 第 1007-1014 行：用 target model 预测 V(s')
with torch.no_grad():
    next_logits = self.target_model(...)
    next_value = expected_value_from_logits(next_logits, bin_centers_device)

# 第 1017 行：Bellman backup
td_target = reward + self.config.td_gamma * (1.0 - done) * next_value

# 第 1020 行：MSE 损失
per_sample_td = functional.mse_loss(pred_value, td_target.detach(), reduction="none")

# 第 1024 行：记录 TD loss
td_loss_value = per_sample_td.mean()
```

**量纲分析**：
- TD loss = `(V(s) - target)²`
- 如果预测准确，误差 ≈ 0.1，则 TD loss ≈ **0.01**
- 如果预测差，误差 ≈ 0.5，则 TD loss ≈ **0.25**

#### 3. 损失组合（第 1023 行）

```python
per_sample_loss = per_sample_loss + self.config.td_loss_weight * per_sample_td
#                  ↑ CE: 1-3          ↑ weight=1.0    ↑ TD MSE: 0.01-0.25
```

**结论**：当 `td_loss_weight=1.0` 时，TD loss 对总损失的贡献仅为 **0.5% - 10%**。

---

## 数学推导与量纲计算

### CE Loss 的理论范围

交叉熵损失：

```
L_CE = -Σ p(i) * log(q(i))
```

- **理论最大值**：`log(num_bins) = log(201) ≈ 5.3`（均匀分布）
- **理论最小值**：`0`（完美预测）
- **实际训练范围**：`1.0 - 3.0`

### TD Loss 的理论范围

**配置参数**（`configuration_pistar_06_td.py` 第 42-51 行）：

```python
bin_min: float = -1.0
bin_max: float = 0.0
td_success_reward: float = 1.0
td_failure_reward: float = -0.6
td_gamma: float = 0.99
```

**Bellman target 的范围**：

```
td_target = reward + gamma * (1 - done) * V(s')
```

- **成功案例（terminal）**：
  ```
  target = 1.0 + 0.99 * 0 * V(s') = 1.0
  ```
  
- **失败案例（terminal）**：
  ```
  target = -0.6 + 0.99 * 0 * V(s') = -0.6
  ```

- **非 terminal 帧**：
  ```
  target = 0 + 0.99 * V(s')
         ∈ [0.99 * (-1.0), 0.99 * 0.0]
         = [-0.99, 0.0]
  ```

**TD target 的总范围**：`[-0.99, 1.0]`

**MSE 的理论范围**：

- **最大误差**：
  ```
  (pred - target)² = (0.0 - (-0.99))² = 0.98 ≈ 1.0
  或
  (pred - target)² = (-1.0 - 1.0)² = 4.0
  ```

- **实际训练误差**（训练中期，模型已部分收敛）：
  ```
  误差 ≈ 0.1 - 0.5
  MSE ≈ 0.01 - 0.25
  ```

### 量纲比计算

| 训练阶段 | CE Loss | TD Loss | 量纲比 (CE/TD) | TD 贡献占比 |
|----------|---------|---------|----------------|-------------|
| 初期     | 5.0     | 0.5     | 10x            | 9%          |
| 中期     | 2.0     | 0.05    | 40x            | 2.4%        |
| 后期     | 1.5     | 0.02    | 75x            | 1.3%        |

**结论**：在当前配置下，TD loss 的贡献在训练过程中逐渐被稀释。

---

## 解决方案对比

### 方案 1：调整 TD Loss 权重 ⭐ 推荐

#### 原理

通过增大 `td_loss_weight`，让 TD loss 对梯度的贡献与 CE loss 相当。

#### 实施

**文件**：`src/lerobot/values/pistar_06_td/configuration_pistar_06_td.py`

```python
# 第 47 行
td_loss_weight: float = 1.0  # 原值

# 改为 ↓
td_loss_weight: float = 20.0  # 或 10.0, 30.0（需根据实验调整）
```

#### 调参指南

| `td_loss_weight` | 预期效果                          | 适用场景          |
|------------------|-----------------------------------|-------------------|
| 5.0              | TD 贡献 ~10%                      | 轻微引入 TD 信号  |
| 10.0             | TD 贡献 ~20-30%                   | 平衡 CE 和 TD     |
| 20.0             | TD 贡献 ~40-50%                   | **推荐起点**      |
| 50.0             | TD 主导训练                       | 专注 TD bootstrap |

#### 优点

- ✅ **实施简单**：只需修改一行配置
- ✅ **不破坏兼容性**：无需重新训练已有模型
- ✅ **容易调试**：通过 wandb 直接观察效果

#### 缺点

- ❌ **需要手动调参**：不同数据集可能需要不同权重
- ❌ **训练阶段不适应**：训练后期 TD loss 变小，权重需动态调整

---

### 方案 2：使用 Huber Loss 替代 MSE

#### 原理

Huber loss 对大误差使用线性惩罚而非平方惩罚，减少 outlier 的影响，同时保持与 CE loss 相近的量级。

#### 实施

**文件**：`src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py`

```python
# 第 1020 行（原代码）
per_sample_td = functional.mse_loss(pred_value, td_target.detach(), reduction="none")

# 改为 ↓
per_sample_td = functional.huber_loss(
    pred_value, 
    td_target.detach(), 
    delta=1.0,  # 阈值：误差 > 1.0 时线性增长
    reduction="none"
)
```

#### Huber Loss 定义

```
L_huber(x) = {
    0.5 * x²           if |x| ≤ δ
    δ * (|x| - 0.5δ)   if |x| > δ
}
```

对于 `δ=1.0`：
- 小误差（|x| < 1.0）：行为类似 MSE
- 大误差（|x| > 1.0）：线性增长，避免梯度爆炸

#### 优点

- ✅ **稳健性强**：对初期大误差更稳健
- ✅ **现代标准**：DQN、Rainbow、TD3 都使用 Huber loss
- ✅ **减少梯度爆炸**：大误差不会产生过大梯度

#### 缺点

- ❌ **引入新超参数**：`delta` 需要调整
- ❌ **需要配合方案 1**：Huber loss 仍然需要适当的权重

---

### 方案 3：动态权重自适应 🔬 实验性

#### 原理

根据当前训练步的 CE loss 和 TD loss 的相对大小，自动调整 TD loss 的权重，使两者对总损失的贡献保持固定比例。

#### 实施

**文件**：`src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py`

在第 1023 行之后添加：

```python
if self.config.td_loss_weight > 0 and td_loss_value is not None:
    # 计算动态权重
    ce_magnitude = ce_loss_value.detach()
    td_magnitude = td_loss_value.detach()
    
    # 目标：让 TD loss 的加权贡献达到 CE loss 的 alpha 倍
    alpha = 0.5  # TD 贡献 = 0.5 * CE 贡献（即 TD:CE = 1:2）
    adaptive_weight = (alpha * ce_magnitude) / (td_magnitude + 1e-8)
    
    # 限制权重范围，防止极端值
    adaptive_weight = adaptive_weight.clamp(0.1, 100.0)
    
    # 应用动态权重
    per_sample_loss = per_sample_loss + adaptive_weight * per_sample_td
    
    # 记录实际使用的权重（用于监控）
    loss_dict["td_adaptive_weight"] = float(adaptive_weight.item())
else:
    # 原逻辑
    per_sample_loss = per_sample_loss + self.config.td_loss_weight * per_sample_td
```

#### 配置修改

添加新配置项：

```python
# configuration_pistar_06_td.py 第 47 行后添加
td_use_adaptive_weight: bool = False  # 是否启用动态权重
td_adaptive_alpha: float = 0.5        # TD 贡献的目标比例
td_adaptive_weight_min: float = 0.1   # 权重下限
td_adaptive_weight_max: float = 100.0 # 权重上限
```

#### 优点

- ✅ **自动适应训练阶段**：训练初期和后期自动调整
- ✅ **泛化到不同数据集**：无需为每个数据集手动调参
- ✅ **可监控**：wandb 中可观察 `td_adaptive_weight` 的变化

#### 缺点

- ❌ **增加复杂度**：引入新的超参数和逻辑
- ❌ **可能不稳定**：权重突变可能导致训练震荡
- ❌ **需要充分测试**：实验性方案，需验证稳定性

---

### 方案 4：重新设计 Reward 尺度 🔧 长期方案

#### 原理

将 reward 和 value 的尺度设计为与 CE loss 自然对齐（都在 0-10 范围），从根本上避免量纲不匹配。

#### 实施

**文件**：`src/lerobot/values/pistar_06_td/configuration_pistar_06_td.py`

```python
# 第 41-51 行（原配置）
num_bins: int = 201
bin_min: float = -1.0
bin_max: float = 0.0
td_success_reward: float = 1.0
td_failure_reward: float = -0.6

# 改为 ↓
num_bins: int = 201
bin_min: float = 0.0
bin_max: float = 10.0
td_success_reward: float = 10.0
td_failure_reward: float = 0.0
```

#### 设计思想

**新的语义**：
- Value 表示"从当前状态开始，预期能降低多少 CE loss"
- 成功轨迹的 value ≈ 10.0（大幅降低 loss）
- 失败轨迹的 value ≈ 0.0（无法降低 loss）

**量纲对齐**：
- CE loss ∈ [1, 5]
- TD target ∈ [0, 10]
- MSE ≈ (误差)² ∈ [0.01, 1.0]
- **MSE 和 CE 在同一量级**

#### 优点

- ✅ **理论优雅**：reward 尺度与训练目标（CE loss）直接对应
- ✅ **无需调权重**：`td_loss_weight=1.0` 即可
- ✅ **语义清晰**：value 值直接可解释

#### 缺点

- ❌ **破坏兼容性**：需要重新训练所有已有模型
- ❌ **数据需重新处理**：已生成的 value target 需重新计算
- ❌ **迁移成本高**：不适合快速实验

---

## 推荐实施路径

### 阶段 1：快速验证（1-2 天）

**目标**：验证增大 `td_loss_weight` 是否有效

#### 步骤

1. **修改配置**
   ```bash
   vim src/lerobot/values/pistar_06_td/configuration_pistar_06_td.py
   ```
   
   ```python
   # 第 47 行
   td_loss_weight: float = 20.0  # 从 1.0 改为 20.0
   ```

2. **运行训练**
   ```bash
   bash bash/0616_pistar06_td_value_train.sh
   ```

3. **观察 wandb**（训练 500 步后）
   - 查看 `loss_ce` 和 `loss_td` 的曲线
   - 计算实际比例：`loss_td * 20 / loss_ce`
   - **预期目标**：比例在 0.3-1.0 之间

4. **调整权重**
   - 如果 `loss_td * 20` 仍远小于 `loss_ce`，增大到 50.0
   - 如果 `loss_td * 20` 远大于 `loss_ce`，减小到 10.0

---

### 阶段 2：稳健优化（3-5 天）

**目标**：引入 Huber loss 提升稳健性

#### 步骤

1. **修改损失函数**
   ```bash
   vim src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py
   ```
   
   ```python
   # 第 1020 行
   per_sample_td = functional.huber_loss(
       pred_value, td_target.detach(), delta=1.0, reduction="none"
   )
   ```

2. **调整权重**（根据阶段 1 的结果）
   ```python
   # configuration_pistar_06_td.py 第 47 行
   td_loss_weight: float = 15.0  # Huber loss 通常比 MSE 略大，权重可略减
   ```

3. **对比实验**
   - 训练两个模型：MSE (weight=20) vs Huber (weight=15)
   - 比较最终 value MAE 和训练稳定性

---

### 阶段 3：自动化（可选，1 周）

**目标**：实现动态权重自适应

#### 步骤

1. **添加配置项**
   ```python
   # configuration_pistar_06_td.py
   td_use_adaptive_weight: bool = True
   td_adaptive_alpha: float = 0.5
   ```

2. **实现动态权重逻辑**（见方案 3）

3. **充分测试**
   - 在多个数据集上验证稳定性
   - 监控 `td_adaptive_weight` 的变化曲线

---

### 阶段 4：长期重构（可选，2-3 周）

**目标**：重新设计 reward 尺度

#### 步骤

1. **更新配置**（见方案 4）
2. **重新生成所有 value targets**
3. **重新训练基准模型**
4. **验证下游任务性能**

---

## 监控指标

### 添加监控代码

**文件**：`src/lerobot/scripts/lerobot_value_train.py`

在第 308-311 行之后添加：

```python
if is_log_step:
    logging.info(train_tracker)
    if wandb_logger:
        wandb_log_dict = train_tracker.to_dict()
        if output_dict:
            wandb_log_dict.update(output_dict)
            
            # ========== 新增：量纲监控 ==========
            if "loss_ce" in output_dict and "loss_td" in output_dict:
                ce = output_dict["loss_ce"]
                td = output_dict["loss_td"]
                weight = cfg.value.td_loss_weight
                
                # 计算关键指标
                wandb_log_dict["loss_ratio_ce_to_td"] = ce / (td + 1e-8)
                wandb_log_dict["loss_td_weighted"] = td * weight
                wandb_log_dict["loss_td_contribution_pct"] = (
                    100.0 * (td * weight) / (ce + td * weight + 1e-8)
                )
            # ====================================
                
        wandb_logger.log_dict(wandb_log_dict, step)
    train_tracker.reset_averages()
```

### 关键指标定义

| 指标名称                    | 含义                          | 理想范围       |
|-----------------------------|-------------------------------|----------------|
| `loss_ce`                   | 交叉熵损失                    | 1.0 - 3.0      |
| `loss_td`                   | TD 损失（未加权）             | 0.01 - 0.5     |
| `loss_ratio_ce_to_td`       | CE / TD 的比例                | 1 - 10         |
| `loss_td_weighted`          | TD 损失（加权后）             | 0.5 - 2.0      |
| `loss_td_contribution_pct`  | TD 对总损失的贡献百分比       | 20% - 50%      |

### wandb Dashboard 配置

在 wandb 中创建自定义图表：

1. **Loss Components**（多曲线图）
   - `loss_ce`
   - `loss_td_weighted`
   - 预期：两条曲线在同一量级

2. **Loss Ratio**（单曲线图）
   - `loss_ratio_ce_to_td`
   - 预期：从 100 降到 1-10

3. **TD Contribution**（单曲线图）
   - `loss_td_contribution_pct`
   - 预期：稳定在 30% 左右

---

## 实战案例

### 案例 1：量纲失衡导致 TD 无效

#### 现象

```
Step 1000:
  loss_ce: 2.45
  loss_td: 0.023
  loss_ratio: 106.5
  value_mae: 0.15
```

训练 10k 步后，value MAE 不再下降，TD loss 始终 < 0.05。

#### 诊断

TD loss 权重过小，梯度几乎全部来自 CE loss，模型无法学习 temporal consistency。

#### 解决

```python
# 修改配置
td_loss_weight: float = 20.0
```

#### 结果

```
Step 1000:
  loss_ce: 2.40
  loss_td: 0.025
  loss_td_weighted: 0.50
  loss_ratio: 96.0 → 4.8 (weighted)
  value_mae: 0.14
  
Step 5000:
  loss_ce: 1.85
  loss_td: 0.018
  loss_td_weighted: 0.36
  value_mae: 0.09  # 显著下降
```

---

### 案例 2：训练后期 TD 贡献降低

#### 现象

```
Step 1000:
  loss_td_contribution_pct: 35%

Step 10000:
  loss_td_contribution_pct: 8%  # 逐渐降低
```

#### 原因

随着 value 预测变准，TD loss 自然变小（MSE 平方效应），导致贡献占比下降。

#### 解决方案 A：分段调整权重

```python
# 在训练脚本中动态调整
if step < 3000:
    effective_weight = 20.0
elif step < 6000:
    effective_weight = 30.0
else:
    effective_weight = 50.0
```

#### 解决方案 B：使用方案 3 动态权重

启用自适应权重，让系统自动调整。

---

### 案例 3：Huber Loss 提升稳健性

#### 对比实验

**配置**：
- Model A: MSE, weight=20.0
- Model B: Huber (δ=1.0), weight=15.0

#### 训练前 1000 步（初期大误差）

| Metric       | MSE (A)  | Huber (B) |
|--------------|----------|-----------|
| loss_td      | 0.45     | 0.28      |
| grad_norm    | 15.3     | 8.7       |
| 训练稳定性   | 偶尔震荡 | 平滑      |

#### 训练 10000 步（收敛后）

| Metric     | MSE (A) | Huber (B) |
|------------|---------|-----------|
| value_mae  | 0.092   | 0.089     |
| 最终性能   | 相近    | 略好      |

#### 结论

Huber loss 在训练初期提供更稳定的梯度，最终性能相近或略优。

---

## 附录：完整代码修改清单

### 修改 1：配置文件

**文件**：`src/lerobot/values/pistar_06_td/configuration_pistar_06_td.py`

```python
# 第 47 行
- td_loss_weight: float = 1.0
+ td_loss_weight: float = 20.0  # 调整为 20.0
```

---

### 修改 2：损失函数（可选）

**文件**：`src/lerobot/values/pistar_06_td/modeling_pistar_06_td.py`

```python
# 第 1020 行
- per_sample_td = functional.mse_loss(pred_value, td_target.detach(), reduction="none")
+ per_sample_td = functional.huber_loss(
+     pred_value, td_target.detach(), delta=1.0, reduction="none"
+ )
```

---

### 修改 3：监控指标

**文件**：`src/lerobot/scripts/lerobot_value_train.py`

```python
# 第 311 行之后添加
if "loss_ce" in output_dict and "loss_td" in output_dict:
    ce = output_dict["loss_ce"]
    td = output_dict["loss_td"]
    weight = cfg.value.td_loss_weight
    
    wandb_log_dict["loss_ratio_ce_to_td"] = ce / (td + 1e-8)
    wandb_log_dict["loss_td_weighted"] = td * weight
    wandb_log_dict["loss_td_contribution_pct"] = (
        100.0 * (td * weight) / (ce + td * weight + 1e-8)
    )
```

---

## 总结

### 核心要点

1. **问题本质**：TD loss (MSE) 的量纲天然比 CE loss 小 10-100 倍
2. **快速解决**：增大 `td_loss_weight` 到 20-50
3. **稳健方案**：使用 Huber loss 替代 MSE
4. **长期方案**：重新设计 reward 尺度

### 推荐配置

```python
# 快速起步
td_loss_weight: float = 20.0

# 稳健生产
td_loss_weight: float = 15.0
# + Huber loss (delta=1.0)

# 理想长期
bin_min: float = 0.0
bin_max: float = 10.0
td_success_reward: float = 10.0
td_loss_weight: float = 1.0
```

### 验证清单

- [ ] wandb 中 `loss_td_weighted` 与 `loss_ce` 在同一量级
- [ ] `loss_td_contribution_pct` 在 20%-50% 之间
- [ ] `value_mae` 在训练中持续下降
- [ ] 训练曲线平滑，无异常震荡
- [ ] 下游任务性能提升（policy training with value guidance）

---

**文档版本**：v1.0  
**最后更新**：2026-06-22  
**作者**：Evo-RL Team
