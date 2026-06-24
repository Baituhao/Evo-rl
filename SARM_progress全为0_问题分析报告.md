# SARM 推理 progress 全为 0 问题分析报告

> 分析日期：2026-06-10
> 相关文件：`bash/0605_sarm_infer.sh`、`bash/0605_sarm_train_infer_workflow.sh`、`bash/config/0601_sarm_train.json`
> 数据集：`datasets_oss/sarm_datatset/20260601103424_20260429-20260519`

---

## 一、问题现象

SARM 推理输出的 `progress_sparse` 全部为 `0.0`。

推理产物 `outputs/0605_sarm_infer/sarm_infer_sarm_0421_20260605_132108/sarm_2002_20260605_132108.parquet`：

| 统计项 | 值 |
|---|---|
| 帧数 | 3271 |
| progress_sparse min / max / mean | 0 / 0 / 0 |
| unique values | `[0.0]` |

---

## 二、根本原因

**config 声明 state 为 16 维，但数据集 stats 中 state 为 21 维，且第 16–20 维为恒定值（std=0）。旧代码对完整 21 维做 MEAN_STD 归一化，std=0 维度归一化后数值爆炸，导致训练从第 1 步起 loss 即为 NaN，模型权重全部被污染为 NaN，推理结果恒为 0。**

---

## 三、完整证据链

### 1. 推理结果恒为 0 的直接原因：模型权重全是 NaN

对 checkpoint 做权重检查：

| checkpoint | NaN 参数 / 总参数 |
|---|---|
| `002500/pretrained_model` | 218 / 222 |
| `005000/pretrained_model` | 218 / 222 |

随机输入下，模型输出 `tau_pred = [nan, nan, ...]`。

推理时的计算链（`modeling_sarm.py:calculate_rewards`）：

```
raw_reward = stage_idx(0) + tau_pred(nan) = nan
normalize_stage_tau(nan) → clamp(0,1) → 0.0
```

因此 `progress_sparse` 全部为 0。**推理脚本本身没有问题。**

### 2. 模型权重为何全 NaN：训练从第 1 步就 loss=nan

训练日志 `outputs/0601_sarm_train&infer_workflow/sarm_train_0601_20260601_190226.log`：

```
step:5  loss:nan grdn:nan
step:10 loss:nan grdn:nan
...
step:5000 loss:nan grdn:nan   (全程 1000 行 step 日志均为 nan)
```

当天更早的一次训练（`154135`）同样全程 NaN。NaN 从第一个 log 点（step 5）即出现，说明训练第 1 步即发散。

### 3. NaN 来源：输入数据含 inf/nan（实验复现）

用干净的随机初始化模型，逐一向输入注入污染，复现 NaN：

| 注入污染 | 第 1 步 loss |
|---|---|
| 全部正常 | 0.068 ✓ |
| video_features 含 inf | **nan** |
| text_features 含 inf | **nan** |
| state_features 含 inf | **nan** |
| state_features 含 nan | **nan** |
| sparse_targets 含 nan | **nan** |

证明：只要任一路输入含 inf/nan，loss 第 1 步即 NaN，与训练日志一致。

### 4. 锁定污染来源：state 归一化

数据集 `meta/stats.json` 中 `observation.state`（21 维）：

```
std==0 的维度: [16, 17, 18, 19, 20]
std min: 0.0   max: 2.74
mean / std 本身无 nan/inf
```

config 声明与数据集实际维度不一致：

| 来源 | state 维度 |
|---|---|
| `bash/config/0601_sarm_train.json` | **16** |
| 数据集 `meta/info.json` | **21** |

### 5. 旧代码无截断逻辑，导致归一化数值爆炸

`git show HEAD:.../processor_sarm.py`（即 0601 训练时所用代码）：

- `TruncateStateProcessorStep` 计数 = **0**（不存在）
- `_truncate_stats_to_feature_dim` 计数 = **0**（不存在）
- Normalizer 之前不对 state 做截断

旧代码直接对 21 维 state 做 `MEAN_STD` 归一化。归一化公式 `denom = std + 1e-8`：

```
std=0 维度 → denom = 1e-8
(x - mean) / 1e-8
```

模拟：若恒定维度真实值与 mean 有 1e-3 的浮点差异：

```
(1e-3) / 1e-8 = 1e5
```

归一化结果爆炸到 1e5 量级，进入 transformer 后数值溢出 → loss=nan → 梯度=nan → 权重被污染 → 全程 NaN。

---

## 四、为何 0421 训练正常、0601 异常

| | 0421（正常） | 0601（全 NaN） |
|---|---|---|
| 数据集 | `openarm_data_260306_260319_sft` | `sarm_datatset/20260601103424_...`（21 维含 std=0） |
| state 维度匹配 | 匹配 | 不匹配（16 vs 21） |
| 截断逻辑 | — | 旧代码无截断 |
| image 预处理 | `image_downsample_size=[480,640]` | `image_center_crop=[800,800]` |
| step:5 loss | 0.171（正常收敛） | nan |

> 注：`image_center_crop=[800,800]` 经验证与本次 NaN **无关**——CLIP 对该尺寸图像编码正常输出。NaN 的唯一根因是 state 归一化。

---

## 五、当前工作区已包含修复

当前工作区的 `src/lerobot/policies/sarm/processor_sarm.py`（**未提交**改动）已加入两处修复：

1. `TruncateStateProcessorStep(target_dim=declared_state_dim)` — 把输入 state 从 21 维截断到声明的 16 维。
2. `_truncate_stats_to_feature_dim()` — 把 normalizer 的 stats 同步截断到 16 维。

用真实 stats + 当前代码实跑验证：

```
原始 stats std (21维): [..., 0.0, 0.0, 0.0, 0.0, 0.0]   (16-20 维 std=0)
截断后 std (16维):      [...]                              (无 0 维)
截断后 std==0 dims: []
```

那 5 个 std=0 的维度被完整丢弃，归一化分母全部正常，不再爆炸。

---

## 六、结论与建议

### 结论

1. `progress_sparse` 全为 0 是因为模型权重全 NaN，而非推理脚本问题。
2. 权重全 NaN 是因为训练从第 1 步即 loss=nan。
3. 根因是 **16 维 / 21 维 state 维度不匹配**：数据集 state 第 16–20 维 std=0，旧代码未截断即做 MEAN_STD 归一化，数值爆炸引发 NaN。

### 关于 0605 新训练（`bash/0605_sarm_train_infer_workflow.sh`）

**不会重现该问题**，前提是：

- 跑训练时使用**当前工作区的 `processor_sarm.py`**（含截断修复），不要用 `git stash` / `git checkout` 还原这些未提交改动。
- 复用的 config `bash/config/0601_sarm_train.json` 中 state 仍声明为 16 维（截断目标），与修复逻辑配套。

### 建议

1. 将 `processor_sarm.py` 中的 state 截断修复**单独提交**，避免未提交改动被误丢失。
2. `image_center_crop=[800,800]` 与本问题无关，可保留。
3. 训练启动后检查首个 step 的 loss 是否为有限值，作为快速验证手段。
4. （可选）从数据源头确认 state 第 16–20 维是否为无效/占位维度，必要时在数据集生成阶段移除。

---

## 附：快速复检命令

检查数据集 state stats 是否含 std=0：

```python
import json, numpy as np
stats = json.load(open('<dataset>/meta/stats.json'))
std = np.array(stats['observation.state']['std'])
print('std==0 dims:', np.where(std == 0)[0].tolist())
print('config 声明维度需与截断目标一致')
```

检查 checkpoint 权重是否含 NaN：

```python
import torch
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
m = SARMRewardModel.from_pretrained('<ckpt>/pretrained_model')
n = sum(1 for _, p in m.named_parameters() if torch.isnan(p).any())
print(f'NaN 参数: {n} / {sum(1 for _ in m.named_parameters())}')
```
