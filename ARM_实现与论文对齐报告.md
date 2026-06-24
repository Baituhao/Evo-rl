# ARM 实现与论文对齐报告

> 模块路径：`src/lerobot/policies/arm/`
> 对应论文：*ARM: Advantage Reward Modeling for Long-Horizon Manipulation* (arXiv 2604.03037v2)
> 报告日期：2026-06-02

本报告依据当前仓库代码（`modeling_arm.py` / `configuration_arm.py` /
`processor_arm.py` / `arm_utils.py` / `compute_rabc_weights.py`）整理，逐项给出
实现细节，并标注与论文的**对齐 / 未对齐**情况，供汇报使用。

---

## 1. 总览

ARM 是一个**长程操作任务的奖励模型**（reward model），不是动作策略。它通过单个
因果 Transformer 编码器，对一段历史帧窗口建模，输出两类信号：

1. **Multi-frame Advantage（多帧优势）**：对相邻帧之间的"转移"做三分类
   （退步 / 停滞 / 前进），监督任务进展方向。
2. **Task Completion（任务完成）**：对窗口中**当前帧**预测是否处于成功终止态。

两个头联合训练，最终用于为下游 RA-BC（Reward-Aware Behavior Cloning）提供
逐帧 progress 权重。

整体数据流：

```
数据集帧窗口 ──► CLIP(ViT-B/32) 冻结编码 ──► 三路 MLP 加性融合 ──►
8 层因果 Transformer ──► {Advantage Head, Completion Head} ──► 联合损失
```

---

## 2. 模型架构实现细节

代码位置：`modeling_arm.py`。

### 2.0 架构流水线图（含张量形状）

```
ARM 架构流水线（window_size=5, d_model=512, batch=B）
═══════════════════════════════════════════════════════════════════════════

输入（一个因果窗口，帧间隔 gap=30 → 1Hz）
┌─────────────────────────────────────────────────────────────────────────┐
│  图像帧序列          本体状态序列         任务文本(goal)                    │
│  (B,5,C,H,W)         (B,5,21)             "pick up the cube"               │
└──────┬───────────────────┬────────────────────┬──────────────────────────┘
       │                    │                    │
       ▼  [CLIP ViT-B/32]   │  [pad→32]          ▼  [CLIP text, 冻结]
   冻结·no_grad             │                 冻结·no_grad
       │                    │                    │
   (B,5,512)            (B,5,32)              (B,512)
   video_features       state_features        text_features
       │                    │                    │
══════ Processor 边界（CLIP 编码在此完成）══════════════════════════════════
       │                    │                    │
       ▼ visual_mlp         ▼ state_mlp          ▼ goal_mlp
  Linear512→512        Linear32→512         Linear512→512
  ReLU                 ReLU                 ReLU
  Linear512→512        Linear512→512        Linear512→512
   (B,5,512)            (B,5,512)             (B,512)
       │                    │                    │ unsqueeze+expand
       │                    │                    ▼ (B,5,512)
       └──────────┬─────────┴──────────┬─────────┘
                  ▼  逐元素相加 (加性融合)
            x = v + s + g            (B,5,512)
                  │
                  ▼  + pos_embedding[:, :5]   (1,5,512) 可学习
            x                        (B,5,512)
                  │
        ┌─────────▼──────────────────────────────────┐
        │   因果 Transformer Encoder  ×8 层            │
        │   heads=8, FFN=2048, dropout=0.1            │
        │   causal_mask(上三角) + pad_mask             │
        └─────────┬──────────────────────────────────┘
                  ▼  LayerNorm
            h                        (B,5,512)
                  │
        ┌─────────┴───────────────────────┐
        ▼                                  ▼
┌───────────────────────────┐   ┌──────────────────────────────┐
│  Advantage Head            │   │  Task Completion Head        │
│  拼接相邻态 (h_i,h_{i+1})   │   │  仅取当前帧 h[:,-1,:]         │
│  cat → (B,4,1024)          │   │  (B,512)                     │
│  Linear 1024→3             │   │  Linear 512→1                │
│  → (B,4,3)                 │   │  → (B,1)                     │
│  T=5 帧 → T-1=4 个转移      │   │  每窗口 1 个输出              │
└───────────┬────────────────┘   └───────────┬──────────────────┘
            ▼                                 ▼
   adv_logits (B,4,3)                comp_logits (B,)
   三分类:{退步,停滞,前进}            当前帧是否成功终止态
            │                                 │
   训练 ▼ CrossEntropy(L_int)        训练 ▼ FocalLoss(L_succ, α=γ=2)
            └──────────────┬──────────────────┘
                           ▼
              L = λ_int·L_int + λ_succ·L_succ      (λ=1.0/1.0)
═══════════════════════════════════════════════════════════════════════════
推理 calculate_rewards:
   softmax(adv_logits) → tri-state 标签{-1,0,+1} + 概率 (B,3)
   sigmoid(comp_logits) → progress 占位 reward (B,)
   ⚠ 论文 Sec3.3 全局进度反向累加重建：未实现
```

### 2.0.1 可训练参数量

CLIP 编码器在 Processor 中冻结（`no_grad`），不计入。可训练参数共
**26,556,420（≈26.56 M）**：

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| visual_mlp (512→512→512) | 525,312 | 2.0% |
| state_mlp (32→512→512) | 279,552 | 1.1% |
| goal_mlp (512→512→512) | 525,312 | 2.0% |
| pos_embedding (1,5,512) | 2,560 | 0.0% |
| Transformer ×8 | 25,219,072 | 94.9% |
| LayerNorm | 1,024 | 0.0% |
| **ARMEncoder 小计** | **26,552,832** | **99.99%** |
| Advantage Head (Linear 1024→3) | 3,075 | 0.01% |
| Completion Head (Linear 512→1) | 513 | 0.00% |
| **可训练合计** | **26,556,420** | **100%** |

主体参数集中在 8 层 Transformer（约 94.9%），两个输出头极轻量（合计 < 4K）。

### 2.1 输入与多模态加性融合（`ARMEncoder`）

每个窗口包含 `window_size` 帧（默认 5）。对每一帧 `i`，三路模态各经一个
两层 MLP（`Linear → ReLU → Linear`）投影到 `d_model`，然后**逐元素相加**：

```
v_i = visual_mlp(CLIP_image_i)      # (B, T, 512) -> (B, T, D)
s_i = state_mlp(state_i)            # (B, T, 32)  -> (B, T, D)
g   = goal_mlp(CLIP_text)           # (B, 512)    -> (B, D)，再 broadcast 到 T
x_i = v_i + s_i + g                 # 加性融合
x_i = x_i + pos_embedding[:, :T]    # 可学习位置编码
```

- 视觉/文本特征：CLIP ViT-B/32（512 维），在 processor 中冻结编码（见 §4）。
- 状态：`pad_state_to_max_dim` 把本体状态零填充/截断到 `max_state_dim=32`。
- 位置编码：`nn.Parameter(1, window_size, D)`，`trunc_normal_(std=0.02)` 初始化。

### 2.2 因果 Transformer 主干

```python
encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, 4*d_model,
                                           dropout, batch_first=True)
transformer   = nn.TransformerEncoder(encoder_layer, num_layers)  # 8 层
```

- `num_layers=8`、`num_heads=8`、FFN 维度 `4*d_model`、`dropout=0.1`。
- **因果掩码**：`torch.triu(ones(T,T), diagonal=1)` 上三角，保证帧 `i` 只能看到
  `≤ i` 的帧，符合论文"causal / 单向"设定。
- **padding 掩码**：`arange(T) >= lengths`，对越界帧屏蔽。当前实现里所有帧都有效
  （`lengths = window_size`，边界帧通过 clamp 复制），所以 pad 掩码实际不裁剪。
- 输出经 `LayerNorm` 得到 `h: (B, T, D)`。

### 2.3 Multi-frame Advantage Head（多帧优势头）

```python
pair = torch.cat([h[:, :-1, :], h[:, 1:, :]], dim=-1)  # (B, T-1, 2D)
logits = Linear(2D, 3)(pair)                            # (B, T-1, 3)
```

- 对**相邻隐藏态对** `(h_i, h_{i+1})` 拼接后线性三分类，描述这段**转移**的优势。
- `T` 帧 → `T-1` 个转移标签。三类语义：
  `0=退步(-1)`、`1=停滞(0)`、`2=前进(+1)`。
- `T<2` 时返回空张量 `(B, 0, 3)`，避免边界报错。

### 2.4 Task Completion Head（任务完成头）

```python
logits = Linear(D, 1)(h[:, -1, :])   # (B, 1)
```

- **只对当前帧**（窗口最右 `h[:, -1, :]`）预测一个标量 logit，判断
  `s_t` 是否为成功终止态。**每个窗口 1 个输出**，不是逐帧。
- 这是本次明确按论文修正过的点（原 SARM 实现是逐帧 `(B, T)`）。

---

## 3. 模型训练实现细节

代码位置：`modeling_arm.py::ARMRewardModel.forward` + `arm_utils.py`。

### 3.1 前向与损失

输入 batch 的 `observation` 含：`video_features (B,T,512)`、`text_features (B,512)`、
`state_features (B,T,32)`、`lengths (B,)`、`advantage_targets (B,T-1)`、
`completion_targets (B,)`。

**Advantage 损失（L_int）**——交叉熵：

```python
adv_logits = advantage_head(h)                  # (B, T-1, 3)
trans_valid = frame_valid[:, :-1] & frame_valid[:, 1:]   # 转移两端都有效
adv_loss = F.cross_entropy(adv_logits[trans_valid], targets[trans_valid])
```

- 仅对有效转移计损；无有效转移时 `adv_loss = adv_logits.sum()*0`（保持计算图）。

**Completion 损失（L_succ）**——Focal Loss：

```python
comp_logits = completion_head(h).squeeze(-1)    # (B,)
# 兼容 (B,) 或 (B,T)：若为 (B,T) 取最后一帧
comp_loss = focal_loss(comp_logits, completion_targets,
                       alpha=focal_alpha, gamma=focal_gamma)
```

Focal Loss 实现（`arm_utils.focal_loss`）：

```
ce   = BCEWithLogits(logits, target)        # = -log(p_t)
p_t  = p*target + (1-p)*(1-target)
loss = mean( alpha * (1-p_t)^gamma * ce )
```

注意：这里 `alpha` 是论文式中的**整体缩放系数**（=2.0），不是按类别平衡项。

**联合损失**：

```
L_ARM = lambda_int * L_int + lambda_succ * L_succ      # 1.0 * + 1.0 *
```

### 3.2 训练标签的来源

标签生成在 processor（§4）完成，模型只消费。两个来源：

1. **优先：数据集 `tri_state` 列**（用户即将提供）。每帧一个值 `{-1,0,+1}`，
   processor 映射 `+1→{0,1,2}`，并丢弃第 0 帧得到 `(B, T-1)` 转移标签。
2. **回退：线性 progress 启发式**（占位）。当数据集无 `tri_state` 时，
   按 `frame_position / episode_length` 计算线性进度，用阈值法派生标签。

Completion 标签目前**始终**由线性 progress 派生（`progress ≥ 1-ε` 视为完成，
取窗口最后一帧），尚无专门的完成标签列。

### 3.3 训练超参（`configuration_arm.py`）

| 超参 | 代码取值 | 论文 (Table 6) | 是否一致 |
|---|---|---|---|
| 视觉/文本编码器 | CLIP ViT-B/32（冻结） | CLIP ViT-B/32 | ✅ |
| 窗口大小 window_size | 5 | 5 帧 @ 1Hz | ✅ |
| 帧间隔 frame_gap | 30（30fps→1Hz） | 1Hz | ✅ |
| Transformer 层数 | 8 | 8 | ✅ |
| 注意力头数 | 8 | — | （论文未细列） |
| hidden_dim / d_model | 512 | — | （与 CLIP 512 对齐） |
| max_state_dim | 32 | — | 工程设定 |
| dropout | 0.1 | — | — |
| batch_size | 64 | 64 | ✅ |
| 优化器 | AdamW | AdamW | ✅ |
| 学习率 lr | 5e-5 | 5e-5 | ✅ |
| weight_decay | 1e-3 | 1e-3 | ✅ |
| betas / eps | (0.9,0.999)/1e-8 | — | 默认 |
| warmup steps | 1000 | 1000 | ✅ |
| 调度器 | Cosine decay | Cosine | ✅ |
| decay steps | 50000 | — | 工程设定 |
| λ_int | 1.0 | 1.0 | ✅ |
| λ_succ | 1.0 | 1.0 | ✅ |
| Focal α | 2.0 | 2.0 | ✅ |
| Focal γ | 2.0 | 2.0 | ✅ |
| completion ε | 1e-3 | 1e-3 | ✅ |
| progress_delta_threshold | 0.01 | — | 仅启发式回退用 |
| 优势类别数 | 3 | 3（tri-state） | ✅ |
| 训练 epoch | 由训练脚本控制 | 2 | ⚠️ 见 §7 |
| 精度 FP16 | 由训练框架控制 | FP16 | ⚠️ 见 §7 |

---

## 4. 数据处理与 Processor 实现细节

代码位置：`processor_arm.py`。

### 4.1 CLIP 编码（冻结，在 Processor 中完成）

```
图像 (B, T, C, H, W) ──► 可选 downsample ──► CLIP.get_image_features ──► (B, T, 512)
文本 task string      ──► CLIP.tokenizer   ──► CLIP.get_text_features  ──► (B, 512)
```

- `clip_batch_size=64`：批量送入 CLIP 防 OOM，内部拆分。
- `image_downsample_size`：可选先缩放（bilinear + antialias），再送 CLIP 预处理。
- 编码全程 `torch.no_grad()`。

### 4.2 tri_state 标签传递链

由于 LeRobot `batch_to_transition` 的 `COMPLEMENTARY_DATA` 白名单会丢弃非标准列，
特别实现了以下三段链路：

1. **resolve_delta_timestamps（`datasets/factory.py`）**：新增
   `extra_delta_timestamps_keys` 钩子，将 `tri_state` 列按与
   `observation_delta_indices` 相同的时间戳窗口化，使 DataLoader 返回
   `(T,)` 的窗口标签。

2. **`arm_batch_to_transition`（`processor_arm.py`）**：自定义 `to_transition`，
   在 `batch_to_transition` 后把 `batch["tri_state"]` 手动存入
   `COMPLEMENTARY_DATA`，绕过白名单丢弃。

3. **`ARMEncodingProcessorStep.__call__`**：从 `comp_data` 取出
   `tri_state`，映射 `{-1,0,+1} → {0,1,2}`（`+1`），丢弃第 0 帧，
   得到 `advantage_targets: (B, T-1)`；completion 标签仍走线性 progress 回退。

当数据集无 `tri_state` 列时，三条链路全部无动作，自动回退到线性 progress 标签。

### 4.3 状态预处理

- 本体状态通过 `NormalizerProcessorStep` 按 `MEAN_STD` 归一化。
- `pad_state_to_max_dim`：裁剪或零填充到 `max_state_dim=32`。

---

## 5. 模型推理接口

代码位置：`modeling_arm.py::ARMRewardModel.calculate_rewards`。

### 5.1 调用签名

```python
rewards, (labels, probs), confidence = model.calculate_rewards(
    text_embeddings,      # (B, 512) 或 (512,) 单样本
    video_embeddings,     # (B, T, 512)
    state_features=None,  # (B, T, 32) 或 None
    lengths=None,
    return_all_frames=False,
    return_stages=True,
    return_confidence=True,
)
```

### 5.2 各输出含义

| 返回值 | 形状（批量/单样本） | 含义 |
|---|---|---|
| `rewards` | `(B,)` / 标量 | 当前帧任务完成概率 `sigmoid(C_t)` |
| `labels`（`return_stages=True`） | `(B,)` / 标量 | 最新转移的三分类预测 `{-1,0,+1}` |
| `probs`（`return_stages=True`） | `(B, 3)` / `(3,)` | 对应三类 softmax 概率 |
| `confidence`（`return_confidence=True`） | `(B,)` / 标量 | 同 `rewards`（复用完成概率） |

- `return_all_frames=True`：`labels/probs` 扩展到全部 `T-1` 个转移，形状变为
  `(B, T-1, 3)` / `(T-1, 3)`。
- `frame_index` 参数：指定取第几个转移（默认最后一个），边界安全。
- 单样本路径（`text_embeddings.dim()==1`）自动 unsqueeze/squeeze，输出去掉 batch 维。

### 5.3 compute_rabc_weights.py 批量推理

`compute_rabc_weights.py` 对整个数据集逐帧推理，输出 `arm_progress.parquet`，
字段：`index`、`episode_index`、`frame_index`、`progress_sparse`（=任务完成概率）。
支持 stride 抽帧 + 线性插值加速（默认 `stride=9`）。
可视化输出：每集一张 PNG（进度曲线 + 三分类堆叠图 + 采样帧缩略图），可选 MP4。

---

## 6. 与论文对齐总表

| 论文要素 | 论文设定 | 当前实现 | 对齐情况 |
|---|---|---|---|
| 多模态融合 | 加性 `x=MLP(v)+MLP(s)+MLP(g)` | 三路 MLP 逐元素相加 | ✅ 对齐 |
| 主干 | 单个因果 Transformer，8 层 | 8 层 `TransformerEncoder`+因果掩码 | ✅ 对齐 |
| 视觉/文本 | CLIP ViT-B/32 冻结 | CLIP ViT-B/32，`no_grad` | ✅ 对齐 |
| 窗口 | 5 帧 @ 1Hz | window=5, gap=30 | ✅ 对齐 |
| Advantage 头 | 转移三分类，T→T-1 | `Linear(2D,3)` 拼接相邻态 | ✅ 对齐 |
| Completion 头 | **当前帧 1 个输出** | `Linear(D,1)` 取 `h[:,-1]` | ✅ 对齐（本次修正） |
| 优势损失 | CrossEntropy | `F.cross_entropy` | ✅ 对齐 |
| 完成损失 | Focal Loss | `focal_loss` BCE 实现 | ✅ 对齐 |
| 联合损失权重 | λ_int=λ_succ=1.0 | 1.0 / 1.0 | ✅ 对齐 |
| Focal α/γ | 2.0 / 2.0 | 2.0 / 2.0 | ✅ 对齐 |
| 完成阈值 ε | 1e-3 | 1e-3 | ✅ 对齐 |
| 优化/调度 | AdamW 5e-5, wd1e-3, warmup1000, cosine | 完全一致 | ✅ 对齐 |
| batch | 64 | 64 | ✅ 对齐 |
| 优势标签 tri-state | `{-1,0,+1}` | `{-1,0,+1}`→`{0,1,2}` | ✅ 对齐 |
| **Global Progress 重建（Sec 3.3）** | C_t 锚定 + Δŷ 反向累加 | **未实现**，用 sigmoid(C_t) 占位 | ❌ 未对齐 |
| **优势标签真实来源** | 数据集真值 | 适配就绪，**数据集 tri_state 列待提供** | ⚠️ 待数据 |
| **完成标签来源** | 真终止态标注 | 仍由线性 progress 启发式派生 | ⚠️ 占位 |
| epoch / FP16 | 2 epoch / FP16 | 由外部训练脚本控制，**未在 config 固化** | ⚠️ 需脚本核对 |

---

## 7. 未对齐 / 待办项（汇报重点）

1. **Global Progress Reconstruction（论文 Sec 3.3）— 未实现。**
   论文用完成头在终止帧锚定 `P_T=1.0`，再沿优势预测 `Δŷ` 反向累加重建逐帧
   全局进度。当前 `calculate_rewards` 直接用 `sigmoid(C_t)` 作为 progress 占位，
   `compute_rabc_weights.py` 的 `progress_sparse` 也是这个占位值。
   → 这是最主要的功能性缺口，需后续实现累加重建。

2. **优势真值标签待接入。** tri_state 适配链路（factory→processor→model）已全部
   就绪并验证通过；一旦数据集
   `datasets_oss/advantige_dataset/20260601103414_20260425-20260528` 补上
   `tri_state` 列即可直接用真值训练。当前缺该列时走线性 progress 占位标签。

3. **完成标签仍为启发式。** Completion 头的训练目标来自线性 progress 阈值
   （`progress ≥ 1-ε`），而非真实终止态标注。若有真终止帧标注应替换。

4. **epoch=2 与 FP16 未在 config 固化。** 这两项由外部训练脚本/框架控制，
   需在实际训练脚本中确认与论文一致。

5. **completion 复用为 confidence。** 推理中 `confidence` 直接复用完成概率，
   语义上等价于 reward，非独立不确定度估计。

---

## 8. 验证状态

模型实例化、前向（兼容 `(B,)` 与 `(B,T)` 完成标签）、`calculate_rewards`
各形状（批量 / 单样本 / `return_all_frames`）、processor 标签生成形状，
均已在 `evo-rl` conda 环境下跑通（`ALL OK`）。导入路径
`from lerobot.policies.arm.modeling_arm import ARMRewardModel` 正常。
