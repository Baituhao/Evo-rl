#!/usr/bin/env python3
"""
Visualize value curves from multiple sources overlaid on episode video.

Usage:
    python scripts/visualize_value_curves.py \
        --dataset-root datasets_oss/fold_0602 \
        --rise-dir outputs_oss/fold_0602_step60000_values \
        --episode 1 \
        --output outputs/value_comparison_ep0001.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

matplotlib.use("Agg")


def load_episode_data_from_parquet(dataset_root: Path, episode_id: int):
    """Load episode frames from parquet files."""
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))

    episode_data = []
    for pq_file in data_files:
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        ep_mask = df["episode_index"] == episode_id
        if ep_mask.any():
            episode_data.append(df[ep_mask])

    if not episode_data:
        raise ValueError(f"Episode {episode_id} not found in dataset")

    import pandas as pd
    ep_df = pd.concat(episode_data, ignore_index=True).sort_values("frame_index")
    return ep_df


def load_rise_values(rise_dir: Path, episode_id: int) -> np.ndarray:
    """Load RISE value predictions from npy file."""
    npy_path = rise_dir / f"episode_{episode_id:04d}_values.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"RISE values not found: {npy_path}")
    return np.load(npy_path)


def load_video_frames(video_path: Path, target_height: int = 360) -> list[np.ndarray]:
    """Load and downsample video frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample
        h, w = frame.shape[:2]
        if h > target_height:
            scale = target_height / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return frames


def plot_value_curves(
    frame_indices: np.ndarray,
    pistar06_values: np.ndarray,
    pistar06_td_values: np.ndarray,
    rise_values: np.ndarray,
    current_frame: int,
    fig_width: int = 12,
    fig_height: int = 3,
    dpi: int = 100,
) -> np.ndarray:
    """Plot three value curves with current frame marker."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Plot all three curves
    ax.plot(frame_indices, pistar06_values, label="pistar06", color="blue", linewidth=1.5, alpha=0.8)
    ax.plot(frame_indices, pistar06_td_values, label="pistar06-td", color="green", linewidth=1.5, alpha=0.8)
    ax.plot(frame_indices, rise_values, label="RISE", color="red", linewidth=1.5, alpha=0.8)

    # Mark current frame
    ax.axvline(current_frame, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Frame Index", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(f"Value Predictions (Frame {current_frame})", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(frame_indices[0], frame_indices[-1])

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return img


def compose_frame(video_frame: np.ndarray, curve_frame: np.ndarray) -> np.ndarray:
    """Stack video frame on top of curve plot."""
    # Resize curve to match video width
    v_h, v_w = video_frame.shape[:2]
    c_h, c_w = curve_frame.shape[:2]

    if c_w != v_w:
        scale = v_w / c_w
        new_h = int(c_h * scale)
        curve_frame = cv2.resize(curve_frame, (v_w, new_h), interpolation=cv2.INTER_AREA)

    # Stack vertically
    return np.vstack([video_frame, curve_frame])


def main():
    parser = argparse.ArgumentParser(description="Visualize value curves over episode video")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--rise-dir", type=Path, required=True, help="Directory with RISE .npy files")
    parser.add_argument("--episode", type=int, required=True, help="Episode ID to visualize")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument("--video-height", type=int, default=360, help="Downsample video to this height")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--pistar06-field", type=str, default="complementary_info.value_pi06-0605")
    parser.add_argument("--pistar06-td-field", type=str, default="complementary_info.value_pistar06-td")
    parser.add_argument("--video-key", type=str, default="observation.images.head",
                        help="Video key in dataset")
    parser.add_argument("--norm-max-frames", type=int, default=8151,
                        help="Max episode frames for normalization (default: 8151)")
    parser.add_argument("--norm-min-frames", type=int, default=1115,
                        help="Min episode frames for normalization (default: 1115)")
    args = parser.parse_args()

    print(f"Loading episode {args.episode} data...")

    # Load parquet data
    ep_df = load_episode_data_from_parquet(args.dataset_root, args.episode)
    frame_indices = ep_df["frame_index"].values

    if args.pistar06_field not in ep_df.columns:
        print(f"ERROR: Field '{args.pistar06_field}' not found in dataset")
        print(f"Available fields: {[c for c in ep_df.columns if 'value' in c.lower()]}")
        sys.exit(1)

    if args.pistar06_td_field not in ep_df.columns:
        print(f"ERROR: Field '{args.pistar06_td_field}' not found in dataset")
        sys.exit(1)

    pistar06_values = ep_df[args.pistar06_field].values
    pistar06_td_values = ep_df[args.pistar06_td_field].values

    # Normalize pistar06 values to RISE space with episode-length-aware scaling
    # Formula: value' = value * clamp(MAX_FRAMES/episode_frames, 1, MAX_FRAMES/MIN_FRAMES) * 2 + 1
    episode_total_frames = len(ep_df)
    max_scale = args.norm_max_frames / args.norm_min_frames
    scale_factor = max(1.0, min(args.norm_max_frames / episode_total_frames, max_scale))

    pistar06_values = pistar06_values * scale_factor * 2 + 1
    pistar06_td_values = pistar06_td_values * scale_factor * 2 + 1

    print(f"Episode length: {episode_total_frames} frames")
    print(f"Normalization scale factor: {scale_factor:.4f} "
          f"(clamped to [1.0, {max_scale:.4f}])")

    # Load RISE values
    print(f"Loading RISE values from {args.rise_dir}...")
    rise_values = load_rise_values(args.rise_dir, args.episode)

    if len(rise_values) != len(frame_indices):
        print(f"WARNING: RISE values length ({len(rise_values)}) != episode length ({len(frame_indices)})")
        # Pad or truncate
        if len(rise_values) < len(frame_indices):
            rise_values = np.pad(rise_values, (0, len(frame_indices) - len(rise_values)),
                                 mode='edge')
        else:
            rise_values = rise_values[:len(frame_indices)]

    # Load video
    print("Loading video frames...")
    # Find video file path from dataset metadata
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata('my/rollout', root=args.dataset_root)
    video_rel_path = meta.get_video_file_path(args.episode, args.video_key)
    video_path = args.dataset_root / video_rel_path

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    video_frames = load_video_frames(video_path, target_height=args.video_height)

    if len(video_frames) != len(frame_indices):
        print(f"WARNING: Video frames ({len(video_frames)}) != data frames ({len(frame_indices)})")
        min_len = min(len(video_frames), len(frame_indices))
        video_frames = video_frames[:min_len]
        frame_indices = frame_indices[:min_len]
        pistar06_values = pistar06_values[:min_len]
        pistar06_td_values = pistar06_td_values[:min_len]
        rise_values = rise_values[:min_len]

    print(f"Rendering {len(video_frames)} frames...")

    # Setup output video
    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_composed = compose_frame(
        video_frames[0],
        plot_value_curves(frame_indices, pistar06_values, pistar06_td_values, rise_values,
                         frame_indices[0], fig_width=12, fig_height=3)
    )
    out_h, out_w = first_composed.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_w, out_h))

    # Render each frame
    for i, (video_frame, frame_idx) in enumerate(tqdm(zip(video_frames, frame_indices),
                                                       total=len(video_frames))):
        curve_img = plot_value_curves(
            frame_indices, pistar06_values, pistar06_td_values, rise_values,
            frame_idx, fig_width=12, fig_height=3
        )
        composed = compose_frame(video_frame, curve_img)
        out.write(composed)

    out.release()
    print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
