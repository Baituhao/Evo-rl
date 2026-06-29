# Streaming Inference with Checkpoint Resume

## 概述

这个功能实现了**真正的流式推理**，具有以下特点：

### ✅ 特性

1. **真流式**：按 episode 逐个处理，立即写入，释放内存
2. **断点续推**：崩溃后可以从上次位置继续，无需重新推理已完成的 episode
3. **内存可控**：峰值内存 = 单个 episode（~100 帧 ≈ 几 MB）
4. **容错性强**：checkpoint 机制保证进度不丢失

### 📊 与全量模式对比

| 模式 | 内存占用 | 断点续推 | NCCL 超时风险 | 适用场景 |
|------|---------|----------|--------------|---------|
| 全量内存 (`streaming_write=false`) | ~2 MB | ❌ | ✅ 低 | 小数据集（< 50 万帧） |
| **流式模式 (`streaming_write=true`)** | **~5 MB** | **✅ 支持** | **✅ 低** | **大数据集，长时间推理** |

---

## 使用方法

### 1. 启用流式推理

在启动脚本中添加参数：

```bash
accelerate launch \
    -m lerobot.scripts.lerobot_value_infer \
    --acp.streaming_write=true \      # 启用流式推理（默认已启用）
    --acp.write_mode=sidecar \
    # ... 其他参数
```

### 2. 使用全量内存模式

如果想使用全量内存模式（更快但无断点续推）：

```bash
accelerate launch \
    -m lerobot.scripts.lerobot_value_infer \
    --acp.streaming_write=false \     # 禁用流式，使用全量内存
    --acp.write_mode=sidecar \
    # ... 其他参数
```

---

## 工作流程

### Phase 1: 推理阶段（按 episode 流式处理）

```
For each episode:
  1. 从 DataLoader 中筛选该 episode 的数据
  2. 推理该 episode 的所有帧 → 得到 values
  3. 计算该 episode 的 advantages
  4. 立即写入 episode parquet 文件
  5. 更新 checkpoint
  6. 释放内存
```

**输出**：
- `<dataset>/advantage/<tag>/episodes/ep_0000000.parquet`
- `<dataset>/advantage/<tag>/episodes/ep_0000001.parquet`
- ...
- `<dataset>/advantage/<tag>/checkpoint.json`

### Phase 2: 合并阶段（计算全局阈值）

```
1. 读取所有 episode parquet 文件（只读 advantage 列）
2. 计算全局阈值（按 task 计算 quantile）
3. 流式写入最终 parquet（添加 indicator 列）
4. 清理中间文件
```

**输出**：
- `<dataset>/advantage/<tag>/frames.parquet` （最终结果）

---

## 断点续推

### 自动恢复

如果推理过程崩溃（OOM、NCCL 超时、手动终止等），**重新运行相同的命令**即可：

```bash
# 第一次运行（推理了 100 个 episode 后崩溃）
bash bash/run_inference.sh

# 崩溃后，直接重新运行相同命令
bash bash/run_inference.sh
# 会自动跳过已完成的 100 个 episode，从第 101 个继续
```

### Checkpoint 文件

`checkpoint.json` 记录推理进度：

```json
{
  "total_episodes": 7958,
  "completed_episodes": [0, 1, 2, ..., 100],
  "last_update": "2026-06-29T12:34:56",
  "status": "inference",  // "inference" | "merging" | "completed"
  "config_hash": "a1b2c3d4e5f6g7h8",
  "total_frames_processed": 12500,
  "start_time": "2026-06-29T10:00:00",
  "current_episode": 100
}
```

### 查看进度

```bash
# 查看 checkpoint
cat /mnt/cpfs_b5/shenjian/datasets/fold_0623/advantage/<tag>/checkpoint.json | jq .

# 查看已完成的 episode 数量
cat checkpoint.json | jq '.completed_episodes | length'

# 查看进度百分比
cat checkpoint.json | jq '(.completed_episodes | length) / .total_episodes * 100'
```

### 手动干预

```bash
# 清理 checkpoint 从头开始（慎用）
rm -rf /mnt/cpfs_b5/shenjian/datasets/fold_0623/advantage/<tag>/

# 只清理中间文件，保留 checkpoint
rm -rf /mnt/cpfs_b5/shenjian/datasets/fold_0623/advantage/<tag>/episodes/
# 然后重新运行，会从断点继续
```

---

## 目录结构

```
<dataset>/advantage/<tag>/
├── checkpoint.json              # 断点文件
├── episodes/                    # 临时 episode 文件（推理完成后自动清理）
│   ├── ep_0000000.parquet      # [index, value, advantage, task_index]
│   ├── ep_0000001.parquet
│   └── ...
└── frames.parquet               # 最终结果 [index, value, advantage, indicator]
```

---

## 性能特点

### 内存占用

- **峰值内存**：单个 episode 的数据（~100 帧）
  - Value 预测：~400 KB
  - Advantage 计算：~400 KB
  - Parquet 写入缓冲：~1 MB
  - **总计**：~5 MB / episode

- **对比**：
  - 全量模式：~2 MB（但无断点续推）
  - 假流式：~15 MB（还会累积）
  - 真流式：~5 MB（可扩展到无限大数据集）

### 时间开销

以 fold_0623 数据集为例（125,836 帧，7,958 episodes）：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 推理循环 | ~38 小时 | 与其他模式相同 |
| Episode 写入 | < 1 秒/episode | 即时写入，不阻塞 |
| Checkpoint 更新 | < 0.1 秒 | JSON 写入，极快 |
| 合并阶段 | ~2 分钟 | 读取 + 计算阈值 + 写入 |
| **总耗时** | **~38 小时** | 开销几乎可忽略 |

---

## 故障排查

### 1. 配置冲突错误

**错误**：`Config hash mismatch!`

**原因**：checkpoint 记录的配置与当前运行的配置不一致

**解决**：
```bash
# 方案 A：使用新的 output 目录
--output_dir=/path/to/new/output

# 方案 B：删除旧的 checkpoint（会从头开始）
rm checkpoint.json
```

### 2. Episode 文件损坏

**症状**：合并阶段报错 `Error reading episode parquet`

**解决**：
```bash
# 删除损坏的 episode 文件
rm episodes/ep_0001234.parquet

# 重新运行，会重新推理该 episode
bash bash/run_inference.sh
```

### 3. NCCL 超时（多节点）

**症状**：`NCCL timeout after 600s`

**原因**：真流式推理在主进程写入时，其他进程在等待

**解决**：已经内置 30 分钟超时，正常情况不会超时

---

## 测试

### 快速测试（10 个 episode）

```bash
bash bash/test_true_streaming.sh
```

### 测试断点续推

```bash
# 1. 运行推理（故意只跑 5 个 episode）
accelerate launch -m lerobot.scripts.lerobot_value_infer \
    --acp.true_streaming=true \
    --dataset.episodes="[0,1,2,3,4,5,6,7,8,9]" \
    # ... 其他参数

# 2. 手动终止（Ctrl+C）

# 3. 检查 checkpoint
cat checkpoint.json | jq '.completed_episodes'
# 输出：[0, 1, 2, 3, 4]

# 4. 重新运行相同命令
accelerate launch -m lerobot.scripts.lerobot_value_infer \
    # 完全相同的参数
# 输出：Resuming from checkpoint: 5/10 episodes completed
#      Episode 0-4 already completed, skipping
#      Starting episode 5...
```

---

## 限制与注意事项

### 1. 只支持 sidecar 模式

真流式推理**只支持 `write_mode=sidecar`**，不支持 `in_place` 模式。

### 2. 多节点同步开销

在多节点环境下，每个 episode 完成后都需要 `wait_for_everyone()`，会增加同步开销。

**建议**：单节点多 GPU 或数据并行场景使用。

### 3. Episode 文件占用磁盘

- 7,958 episodes × ~100 KB/episode ≈ **800 MB**
- 推理完成后会自动清理

### 4. 配置不可变

Checkpoint 记录了配置 hash，重新运行时配置必须一致。

如需改配置，请：
- 使用新的 `output_dir`
- 或删除旧的 checkpoint

---

## 实现细节

### 核心文件

- `src/lerobot/common/inference_checkpoint.py` - Checkpoint 管理
- `src/lerobot/scripts/value_infer_streaming.py` - 流式推理核心逻辑
- `src/lerobot/scripts/lerobot_value_infer.py` - 主脚本集成
- `src/lerobot/configs/value.py` - 配置定义

### 关键函数

- `run_streaming_inference_with_resume()` - 主入口
- `_infer_single_episode_streaming()` - 单 episode 推理
- `_merge_episodes_and_write_indicators()` - 合并阶段

---

## 版本历史

- **v0.2.0** (2026-06-29): 实现真正的流式推理 + 断点续推
- **v0.1.0** (2026-06-29): 修复 NCCL 超时问题
