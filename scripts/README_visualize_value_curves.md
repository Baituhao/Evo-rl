# Value Curves Visualization Tool

可视化工具，将三条 value 曲线（pistar06, pistar06-td, RISE）叠加在 episode 视频上。

## 功能

- 从数据集 parquet 读取 pistar06 和 pistar06-td 的 value
- 从单独的 npy 文件读取 RISE value
- **自动归一化**：pistar06 和 pistar06-td 的值通过长度感知归一化变换到 RISE 的值域空间
  - 公式：`value' = value * clamp(8151/episode_frames, 1.0, 7.31) * 2 + 1`
  - 短 episode (≤1115帧) 使用 scale 7.31（上限）
  - 中等 episode (1115-8151帧) 的 scale 随长度递减
  - 长 episode (≥8151帧) 使用 scale 1.0（下限），不压缩
- 视频自动下采样到指定高度（降低文件大小）
- 三条曲线平行显示在视频下方
- 当前帧位置用竖线标记

## 使用方法

```bash
python scripts/visualize_value_curves.py \
    --dataset-root datasets_oss/fold_0602 \
    --rise-dir outputs_oss/fold_0602_step60000_values \
    --episode 1 \
    --output outputs/viz_test/value_comparison_ep0001.mp4 \
    --video-height 360 \
    --fps 30
```

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--dataset-root` | ✓ | 数据集根目录，包含 data/ 和 videos/ |
| `--rise-dir` | ✓ | RISE value npy 文件目录 |
| `--episode` | ✓ | Episode ID（整数） |
| `--output` | ✓ | 输出视频路径 |
| `--video-height` | | 视频下采样高度（默认360，原始1080p） |
| `--fps` | | 输出视频帧率（默认30） |
| `--pistar06-field` | | pistar06 value 字段名（默认 `complementary_info.value_pi06-0605`） |
| `--pistar06-td-field` | | pistar06-td value 字段名（默认 `complementary_info.value_pistar06-td`） |
| `--video-key` | | 视频 key（默认 `observation.images.head`） |
| `--norm-max-frames` | | 归一化最大帧数（默认 8151，来自 advantige_dataset） |
| `--norm-min-frames` | | 归一化最小帧数（默认 1115，来自 advantige_dataset） |

## 批量渲染示例

渲染多个 episode:

```bash
for ep in 1 1001 2001 3001 4001 4501; do
    python scripts/visualize_value_curves.py \
        --dataset-root datasets_oss/fold_0602 \
        --rise-dir outputs_oss/fold_0602_step60000_values \
        --episode $ep \
        --output outputs/viz_test/value_comparison_ep$(printf "%04d" $ep).mp4 \
        --video-height 360
done
```

## 输出示例

生成的视频布局：
```
┌─────────────────────────┐
│                         │
│    Episode Video        │  ← 下采样到 360p
│    (360×640)            │
│                         │
├─────────────────────────┤
│  Value Predictions      │
│  [曲线图]               │  ← 三条曲线 + 当前帧标记
│  - pistar06 (蓝, ×2+1)  │
│  - pistar06-td (绿, ×2+1)│
│  - RISE (红)            │
└─────────────────────────┘
```

## 性能

- Episode 4501（377 帧）：约 30 秒
- Episode 1（4106 帧）：约 5 分钟
- 输出文件大小：360p 约为原始 1080p 的 1/10

## 归一化公式详解

pistar06 和 pistar06-td 使用长度感知归一化映射到 RISE 值域：

```
scale_factor = clamp(MAX_FRAMES / episode_frames, 1.0, MAX_FRAMES / MIN_FRAMES)
value' = value * scale_factor * 2 + 1
```

其中 `MAX_FRAMES=8151`, `MIN_FRAMES=1115` (基于 advantige_dataset 统计)

**不同长度 episode 的 scale factor**：

| Episode 长度 | Raw Scale | Clamped Scale | 说明 |
|-------------|-----------|---------------|------|
| 377 帧 | 21.62 | 7.31 | 短 episode，clamp 到上限 |
| 1115 帧 | 7.31 | 7.31 | 刚好在上限 |
| 4106 帧 | 1.99 | 1.99 | 中等长度，在范围内 |
| 8151 帧 | 1.00 | 1.00 | 参考长度，刚好下限 |
| 10454 帧 | 0.78 | 1.00 | 超长 episode，clamp 到下限 |

**Clamp 范围 [1.0, 7.31] 的意义**：
- **下限 1.0**：确保长 episode 不被压缩，保持原始动态范围
- **上限 7.31**：防止极短 episode 过度放大导致数值溢出

## 依赖

- opencv-python (cv2)
- matplotlib
- numpy
- pyarrow
- pandas
- tqdm
- lerobot (for dataset metadata)

所有依赖应该已经在 `evo-rl-pi` conda 环境中。
