#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compute ARM progress values for RA-BC (Reward-Aware Behavior Cloning) weighting.

This script processes all frames in a dataset with ARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses multi-output extraction: each ARM query returns progress for 9 frames, so we only
need ~num_frames/30 queries instead of one per frame (~30x speedup).

Usage:
    # Full RA-BC computation with visualizations
    python src/lerobot/policies/arm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/arm_single_uni4

    # Faster computation with stride (compute every 5 frames, interpolate the rest)
    python src/lerobot/policies/arm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/arm_single_uni4 \\
        --stride 5

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/policies/arm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/arm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 5

The output is saved to the dataset's local cache directory as 'arm_progress.parquet'.
"""

import argparse
import logging
from pathlib import Path
from queue import Queue
from threading import Thread

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.arm.modeling_arm import ARMRewardModel
from lerobot.policies.arm.processor_arm import make_arm_pre_post_processors

# Advantage head class labels (tri-state), used for visualization.
ADVANTAGE_LABELS = ["Negative", "Neutral", "Positive"]


class PrefetchIterator:
    """Single-threaded prefetch wrapper using a background thread."""

    def __init__(self, iterable, prefetch_size=2):
        self.iterable = iter(iterable)
        self.queue = Queue(maxsize=prefetch_size)
        self.thread = Thread(target=self._producer, daemon=True)
        self.thread.start()

    def _producer(self):
        try:
            for item in self.iterable:
                self.queue.put(item)
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


def parse_episode_indices(value: str) -> list[int]:
    """Parse comma-separated episode indices string into a list of ints."""
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid episode indices '{value}': {e}") from e


class FrameSubset(Dataset):
    """Thin Dataset wrapper that maps a list of absolute frame indices to dataset items."""

    def __init__(self, dataset: LeRobotDataset, frame_indices: list[int], image_key: str, state_key: str):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.image_key = image_key
        self.state_key = state_key

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, i):
        idx = self.frame_indices[i]
        sample = self.dataset[idx]
        item = {
            self.image_key: sample[self.image_key],
            "_frame_idx": idx,
        }
        if self.state_key in sample:
            item[self.state_key] = sample[self.state_key]
        return item


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    """Read reward_model_path from parquet metadata if available."""
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def _resolve_dataset_kwargs(dataset_repo_id: str) -> dict:
    """Return LeRobotDataset kwargs, using local root when the path exists on disk."""
    local_path = Path(dataset_repo_id)
    if local_path.exists() and (local_path / "meta" / "info.json").exists():
        logging.info(f"Detected local dataset at '{local_path}', loading offline.")
        return {"repo_id": local_path.name, "root": str(local_path)}
    return {"repo_id": dataset_repo_id}


def load_arm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    device: str = "cuda",
) -> tuple[LeRobotDataset, ARMRewardModel, any]:
    """
    Load ARM model, dataset, and preprocessor.

    Returns:
        Tuple of (dataset, reward_model, preprocessor)
    """
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = ARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices

    logging.info(f"Loading dataset: {dataset_repo_id}")
    ds_kwargs = _resolve_dataset_kwargs(dataset_repo_id)
    temp_dataset = LeRobotDataset(**ds_kwargs, download_videos=True)
    fps = temp_dataset.fps

    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }
    dataset = LeRobotDataset(**ds_kwargs, delta_timestamps=delta_timestamps)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    preprocess, _ = make_arm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    return dataset, reward_model, preprocess


def to_numpy_image(img) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        # Take center frame for bidirectional sampling
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # Handle normalized images (may have negative values or values > 1)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames.

    Same as arm_inference_visualization.py
    """
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # Progress plot
    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    # Sample frames
    ax_frames.axis("off")
    num_sample = 8
    sample_indices = np.linspace(0, len(frames) - 1, num_sample, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[idx])][:12]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _render_chart_to_image(
    progress_preds: np.ndarray,
    stage_preds: np.ndarray,
    stage_labels: list[str],
    current_step: int,
    width: int,
    height: int,
) -> Image.Image:
    """Render progress + stage probability chart using matplotlib (same style as PNG visualization)."""
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    # Only show data up to current_step
    last_step = min(current_step, len(progress_preds) - 1)
    visible_indices = frame_indices[: last_step + 1]
    visible_progress = progress_preds[: last_step + 1]
    visible_stages = stage_preds[: last_step + 1, :]

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)
    ax_progress = fig.add_subplot(gs[0])
    ax_stages = fig.add_subplot(gs[1])

    # Progress plot
    ax_progress.plot(visible_indices, visible_progress, linewidth=2, color="#2E86AB")
    ax_progress.fill_between(visible_indices, 0, visible_progress, alpha=0.3, color="#2E86AB")
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress", fontsize=9)
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.grid(True, alpha=0.3)
    ax_progress.set_xlim(0, len(progress_preds) - 1)
    ax_progress.tick_params(labelsize=8)

    # Stage predictions
    ax_stages.stackplot(
        visible_indices,
        *[visible_stages[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    ax_stages.set_xlabel("Frame", fontsize=9)
    ax_stages.set_ylabel("Stage Probability", fontsize=9)
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=7)
    ax_stages.grid(True, alpha=0.3)
    ax_stages.set_xlim(0, len(progress_preds) - 1)
    ax_stages.tick_params(labelsize=8)

    # Convert to PIL Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return Image.fromarray(buf, mode="RGBA").convert("RGB")


def render_episode_video(
    dataset: "LeRobotDataset",
    episode_idx: int,
    image_key: str,
    progress_preds: np.ndarray,
    stage_preds: np.ndarray,
    stage_labels: list[str],
    scheme: str,
    output_path: "Path",
    fps: int = 30,
):
    """Render video with frame on top and dynamic chart below (PNG visualization style)."""
    try:
        import av
    except ImportError:
        logging.warning("PyAV not installed, skipping video rendering. Install with: pip install av")
        return

    ep = dataset.meta.episodes[episode_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    num_frames = ep_end - ep_start

    # Load all frames
    frames: list[Image.Image] = []
    for i in range(num_frames):
        frame_idx = ep_start + i
        sample = dataset[frame_idx]
        img = to_numpy_image(sample[image_key])
        frames.append(Image.fromarray(img))

    if len(frames) == 0:
        logging.warning(f"No frames to render for episode {episode_idx}")
        return

    # Determine layout: frame on top, chart below
    frame_width = frames[0].width
    frame_height = frames[0].height
    chart_height = max(300, frame_height // 2)  # Chart takes ~1/3 of total height
    total_height = frame_height + chart_height

    # Compose frames with dynamic charts
    composed_frames: list[Image.Image] = []
    for i in tqdm(range(num_frames), desc=f"Rendering video ep{episode_idx}", leave=False):
        # Render chart for current step
        chart_img = _render_chart_to_image(
            progress_preds=progress_preds,
            stage_preds=stage_preds,
            stage_labels=stage_labels,
            current_step=i,
            width=frame_width,
            height=chart_height,
        )

        # Combine frame + chart vertically
        combined = Image.new("RGB", (frame_width, total_height))
        combined.paste(frames[i], (0, 0))
        combined.paste(chart_img, (0, frame_height))
        composed_frames.append(combined)

    # Encode to video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vcodec = "libx264"
    video_options = {"g": "2", "crf": "23"}
    pix_fmt = "yuv420p"

    with av.open(str(output_path), "w") as output:
        out_stream = output.add_stream(vcodec, fps, options=video_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = composed_frames[0].width
        out_stream.height = composed_frames[0].height
        for pil_img in composed_frames:
            av_frame = av.VideoFrame.from_image(pil_img.convert("RGB"))
            for packet in out_stream.encode(av_frame):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)

    print(f"Saved video: {output_path}")


def visualize_arm_predictions(
    dataset: LeRobotDataset,
    reward_model: ARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: Path,
    num_display_frames: int = 5,
    stride: int = 1,
    render_video: bool = False,
    video_fps: int = 15,
    task_name: str | None = None,
):
    """
    Visualize ARM predictions for multiple episodes.

    Computes predictions for every frame by default. With stride > 1, computes predictions
    every N frames and interpolates (progress + stage probabilities) for visualization.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured
        reward_model: Loaded ARM model
        preprocess: Preprocessor from make_arm_pre_post_processors
        episode_indices: List of episode indices to visualize
        head_mode: "sparse", "dense", or "both"
        output_dir: Directory to save visualizations
        num_display_frames: Number of frames to display in thumbnail strip (default: 5)
        stride: Compute predictions every N frames, interpolate the rest (default: 1)
        render_video: Also render per-frame overlay mp4 video (default: False)
        video_fps: Frame rate of rendered video (default: 15)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    device = reward_model.device

    # Causal window: the target frame is the last (rightmost) frame.
    target_idx = reward_model.config.window_size - 1

    # Single-head ARM: one visualization pass. The "stages" plot shows the
    # tri-state advantage class probabilities (negative/neutral/positive).
    schemes_to_viz = ["advantage"]

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        task = task_name if task_name is not None else dataset[ep_start].get("task", "perform the task")
        num_frames = ep_end - ep_start

        # Select frames for display thumbnails (evenly sampled from begin to end)
        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames
            else list(range(ep_start, ep_end))
        )
        viz_frames = {}

        # Load display frames up-front (stride mode might skip them otherwise).
        for frame_idx in display_indices:
            sample = dataset[frame_idx]
            viz_frames[frame_idx] = to_numpy_image(sample[image_key])

        # Initialize storage for each scheme
        scheme_data = {}
        for scheme in schemes_to_viz:
            num_stages = reward_model.config.num_advantage_classes
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan),
                "viz_stages": np.full((num_frames, num_stages), np.nan),
                "viz_gt_progress": np.full(num_frames, np.nan),
                "viz_gt_stages": np.full(num_frames, np.nan),
                "target_key": "completion_targets",
                "num_stages": num_stages,
                "temporal_props": None,
                "subtask_names": ADVANTAGE_LABELS,
            }

        if stride > 1:
            logging.info(f"Visualization stride={stride}: inferring every {stride} frames and interpolating")

        # Process frames one at a time to avoid memory buildup
        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                image_key: sample[image_key],
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)
                video_features = processed["video_features"].to(device)
                text_features = processed["text_features"].to(device)
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                lengths = processed.get("lengths")

                for scheme in schemes_to_viz:
                    sd = scheme_data[scheme]

                    # Ground truth: binary task-completion target at the target frame.
                    if stride == 1 and sd["target_key"] in processed:
                        gt_target = processed[sd["target_key"]][0, target_idx].cpu().item()
                        sd["viz_gt_progress"][local_idx] = float(gt_target)

                    # Predictions
                    reward, stage_probs = reward_model.calculate_rewards(
                        text_embeddings=text_features,
                        video_embeddings=video_features,
                        state_features=state_features,
                        lengths=lengths,
                        return_all_frames=True,
                        return_stages=True,
                    )

                    # Handle both tensor and numpy outputs
                    if isinstance(reward, torch.Tensor):
                        reward = reward.cpu().numpy()
                        stage_probs = stage_probs.cpu().numpy()

                    if reward.ndim == 2:
                        sd["viz_progress"][local_idx] = reward[0, target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[0, target_idx, :]
                    else:
                        sd["viz_progress"][local_idx] = reward[target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[target_idx, :]

                # Clear GPU memory after each frame
                del processed, video_features, text_features
                if state_features is not None:
                    del state_features

            torch.cuda.empty_cache()

        # Interpolate predictions back to per-frame arrays for smooth visualization.
        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]

                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(
                        valid_idx, sd["viz_progress"][valid_idx], all_local
                    )

                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for s in range(sd["num_stages"]):
                        stage_interp[:, s] = interpolate_progress(
                            valid_idx, sd["viz_stages"][valid_idx, s], all_local
                        )

                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    nz = row_sums.squeeze(-1) > 0
                    stage_interp[nz] = stage_interp[nz] / row_sums[nz]
                    sd["viz_stages"] = stage_interp
                else:
                    # No valid points: keep NaNs/zeros; visualization will be empty.
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        # Generate visualization for each head
        ordered_viz_frames = [viz_frames[idx] for idx in sorted(display_indices)]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            viz_path = output_dir / f"arm_prediction_ep{episode_idx}_{scheme}.png"

            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx})",
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None,
                gt_stages=sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None,
            )

            if render_video:
                video_path = output_dir / f"arm_prediction_ep{episode_idx}_{scheme}.mp4"
                render_episode_video(
                    dataset=dataset,
                    episode_idx=episode_idx,
                    image_key=image_key,
                    progress_preds=sd["viz_progress"],
                    stage_preds=sd["viz_stages"],
                    stage_labels=stage_labels,
                    scheme=scheme,
                    output_path=video_path,
                    fps=video_fps,
                )

        # Clear memory between episodes
        torch.cuda.empty_cache()

    logging.info(f"Visualizations saved to: {output_dir.absolute()}")


def generate_all_frame_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    """Generate all frame indices, ordered by offset for cache-friendly access.

    Orders frames as: [0, 30, 60...], [1, 31, 61...], ..., [29, 59, 89...]
    This groups frames that share similar temporal windows together.
    """
    num_frames = ep_end - ep_start
    indices = []
    for offset in range(frame_gap):
        for frame_rel in range(offset, num_frames, frame_gap):
            indices.append(ep_start + frame_rel)
    return indices


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate values to fill in gaps (robust to NaNs / edge cases)."""
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def _collate_passthrough(batch):
    """Collate that stacks tensors but leaves non-tensor values as lists."""
    out = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def compute_arm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    num_visualizations: int = 5,
    output_dir: str = "./arm_viz",
    stride: int = 1,
    episode_indices: list[int] | None = None,
    num_workers: int = 0,
    prefetch_size: int = 2,
    render_video: bool = False,
    task_name: str | None = None,
):
    """
    Compute ARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained ARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: ARM head to use ("sparse", "dense", or "both")
        device: Device to use for inference
        num_visualizations: Number of episodes to visualize (0 to skip)
        output_dir: Directory to save visualizations
        stride: Compute progress every N frames, interpolate the rest (default: 1 = every frame)
        episode_indices: List of episode indices to process. If None, processes all episodes.
        num_workers: Number of DataLoader worker processes (default: 0 = single process, recommended for video datasets)
        prefetch_size: Number of batches to prefetch (default: 2, works in both single/multi-process modes)
        task_name: Override the task description string for all episodes. If None, uses the task from dataset metadata.
    """
    dataset, reward_model, preprocess = load_arm_resources(dataset_repo_id, reward_model_path, device)

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames

    # Resolve which episodes to process
    all_episode_set = set(range(num_episodes))
    if episode_indices is not None:
        invalid = set(episode_indices) - all_episode_set
        if invalid:
            raise ValueError(f"Episode indices out of range [0, {num_episodes - 1}]: {sorted(invalid)}")
        episodes_to_process = sorted(set(episode_indices))
        logging.info(
            f"Processing {len(episodes_to_process)}/{num_episodes} selected episodes "
            f"(out of {total_frames} total frames)"
        )
    else:
        episodes_to_process = list(range(num_episodes))
        logging.info(f"Processing {total_frames} frames across {num_episodes} episodes")

    # Single-head ARM: always compute one progress column (kept under the
    # 'progress_sparse' name for downstream RABCWeights compatibility).
    compute_sparse = True
    compute_dense = False

    # Storage arrays
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_progress_sparse = [] if compute_sparse else None
    all_progress_dense = [] if compute_dense else None

    if stride > 1:
        logging.info(f"Using stride={stride}: computing every {stride} frames, interpolating the rest")
    if prefetch_size > 0:
        mode = f"multi-process (num_workers={num_workers})" if num_workers > 0 else "single-threaded"
        logging.info(f"Prefetch enabled ({mode}): prefetch_size={prefetch_size}")

    # Process selected episodes
    for episode_idx in tqdm(episodes_to_process, desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Get task description
        task = task_name if task_name is not None else dataset[ep_start].get("task", "perform the task")

        # Generate frames to compute (with stride applied)
        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            # Only compute every stride-th frame (relative to episode start)
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            # Always include last frame for better interpolation at episode end
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        center_idx = reward_model.config.window_size - 1  # Last frame in causal window

        # Dictionary to collect results
        frame_results = {}

        # Build a DataLoader over the frames to compute for this episode (enables prefetching)
        frame_subset = FrameSubset(dataset, compute_indices, image_key, state_key)
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "collate_fn": _collate_passthrough,
        }
        if num_workers > 0:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["prefetch_factor"] = prefetch_size
        frame_loader = DataLoader(frame_subset, **loader_kwargs)

        # Wrap with prefetch iterator for single-threaded mode
        if num_workers == 0 and prefetch_size > 0:
            frame_loader = PrefetchIterator(frame_loader, prefetch_size=prefetch_size)

        for batch in tqdm(frame_loader, desc=f"  Ep {episode_idx}", leave=False):
            query_idx = int(batch["_frame_idx"][0])
            try:
                item = {
                    image_key: batch[image_key][0],
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                if state_key in batch:
                    item[state_key] = batch[state_key][0]

                with torch.no_grad():
                    processed = preprocess(item)
                    video_features = processed["video_features"].to(device)
                    text_features = processed["text_features"].to(device)
                    state_features = processed.get("state_features")
                    if state_features is not None:
                        state_features = state_features.to(device)
                    lengths = processed.get("lengths")

                    sparse_val = np.nan
                    dense_val = np.nan

                    # Compute progress prediction for the target (last causal) frame
                    if compute_sparse:
                        sparse_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                        )
                        sparse_val = float(
                            sparse_progress[0, center_idx]
                            if sparse_progress.ndim == 2
                            else sparse_progress[center_idx]
                        )

                    frame_results[query_idx] = (sparse_val, dense_val)

            except Exception as e:
                logging.warning(f"Failed to process frame {query_idx}: {e}")

        # Interpolate to get values for all frames
        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        # All frame indices for this episode
        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            # Interpolate progress values
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            # No interpolation needed
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        # Store results for all frames
        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            all_indices.append(frame_idx)
            all_episode_indices.append(episode_idx)
            all_frame_indices.append(local_idx)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    all_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    all_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    all_progress_dense.append(frame_results[frame_idx][1])
                else:
                    all_progress_dense.append(np.nan)

    # Create output table
    table_data = {
        "index": np.array(all_indices, dtype=np.int64),
        "episode_index": np.array(all_episode_indices, dtype=np.int64),
        "frame_index": np.array(all_frame_indices, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(all_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(all_progress_dense, dtype=np.float32)

    # Sort by index
    df = pa.table(table_data).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)
    final_table = pa.Table.from_pandas(df, preserve_index=False)

    # Add metadata with reward model path
    metadata = {b"reward_model_path": reward_model_path.encode()}
    final_table = final_table.replace_schema_metadata(metadata)

    # Determine output path
    output_path = Path(dataset.root) / "arm_progress.parquet" if output_path is None else Path(output_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")

    # Print statistics
    if "progress_sparse" in df.columns:
        valid = df["progress_sparse"].dropna()
        logging.info(
            f"Sparse progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    if "progress_dense" in df.columns:
        valid = df["progress_dense"].dropna()
        logging.info(
            f"Dense progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    # Visualize episodes after processing
    if num_visualizations > 0:
        viz_pool = episodes_to_process
        viz_episodes = viz_pool[: min(num_visualizations, len(viz_pool))]
        logging.info(f"Generating {len(viz_episodes)} visualizations...")
        visualize_arm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
            render_video=render_video,
            task_name=task_name,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute ARM progress values for RA-BC weighting or visualize ARM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full RA-BC computation with visualizations
    python src/lerobot/policies/arm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/arm_single_uni4

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/policies/arm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/arm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 10
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID or local path",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to pretrained ARM model (reads from existing parquet metadata if not provided)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for parquet. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="ARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    # Visualization options
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize ARM predictions (no RA-BC computation)",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5, set to 0 to skip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./arm_viz",
        help="Output directory for visualizations (default: ./arm_viz)",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        default=False,
        help="Enable mp4 video rendering alongside PNG visualizations",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload progress file to the dataset repo on HuggingFace Hub",
        default=False,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=9,
        help="Compute progress every N frames, interpolate the rest (default: 1 = every frame)",
    )
    parser.add_argument(
        "--episode-indices",
        type=parse_episode_indices,
        default=None,
        metavar="IDX1,IDX2,...",
        help="Comma-separated episode indices to process (e.g. 0,1,5). Processes all episodes if not set.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0 = single process, recommended for video datasets)",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=2,
        help="Number of batches to prefetch (default: 2, works in both single/multi-process modes)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Override task description for all episodes (useful when dataset task field is 'default' or empty)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Try to get reward_model_path from parquet metadata if not provided
    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        # Load dataset to find parquet path
        ds_kwargs = _resolve_dataset_kwargs(args.dataset_repo_id)
        temp_dataset = LeRobotDataset(**ds_kwargs, download_videos=False)
        parquet_path = Path(temp_dataset.root) / "arm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)"
            )

    # Handle visualize-only mode
    if args.visualize_only:
        dataset, reward_model, preprocess = load_arm_resources(
            args.dataset_repo_id, reward_model_path, args.device
        )
        logging.info(f"Visualization-only mode: visualizing {args.num_visualizations} episodes")
        ep_pool = (
            args.episode_indices if args.episode_indices is not None else list(range(dataset.num_episodes))
        )
        viz_episodes = ep_pool[: min(args.num_visualizations, len(ep_pool))]
        visualize_arm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=args.head_mode,
            output_dir=Path(args.output_dir),
            stride=args.stride,
            render_video=args.render_video,
            task_name=args.task_name,
        )
        print(f"\nVisualizations saved to: {Path(args.output_dir).absolute()}")
        return

    # Full RABC computation (compute_arm_progress loads model/dataset itself)
    output_path = compute_arm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        episode_indices=args.episode_indices,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        render_video=args.render_video,
        task_name=args.task_name,
    )

    print(f"\nARM progress values saved to: {output_path}")

    # Upload to Hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        hub_path = "arm_progress.parquet"

        print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hub_path,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(
            f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
        )

        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print("  rabc_head_mode: sparse  # or dense")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
