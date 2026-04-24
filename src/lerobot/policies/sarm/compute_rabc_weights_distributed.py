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
Multi-GPU distributed version of compute_rabc_weights.py using Accelerate.

Usage:
    accelerate launch \
        --multi_gpu \
        --num_processes=8 \
        --num_machines=1 \
        --main_process_port=29500 \
        src/lerobot/policies/sarm/compute_rabc_weights_distributed.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4 \
        --stride 5 \
        --num-workers 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from tqdm import tqdm

from lerobot.policies.sarm.compute_rabc_weights import (
    generate_all_frame_indices,
    get_reward_model_path_from_parquet,
    interpolate_progress,
    iter_samples_with_prefetch,
    load_sarm_resources,
    parse_episode_indices,
    resolve_dataset_reference,
    visualize_sarm_predictions,
)


def compute_sarm_progress_distributed(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    num_visualizations: int = 5,
    output_dir: str = "./sarm_viz",
    stride: int = 1,
    episode_indices: list[int] | None = None,
    task_override: str | None = None,
    num_workers: int = 1,
    prefetch_size: int = 32,
):
    """
    Multi-GPU distributed SARM progress computation using Accelerate.

    Each GPU processes a subset of episodes independently, then results are gathered.
    """
    # Initialize accelerator
    accelerator = Accelerator()

    # Only main process logs
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load resources (each process loads independently)
    dataset, reward_model, preprocess = load_sarm_resources(
        dataset_repo_id,
        reward_model_path,
        device=accelerator.device
    )

    # Set to eval mode
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    # Prepare model with accelerator
    reward_model = accelerator.prepare(reward_model)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames

    if accelerator.is_main_process:
        logging.info(f"Processing {total_frames} frames across {num_episodes} episodes")
        logging.info(f"Using {accelerator.num_processes} GPUs")

    # Determine episodes to process
    selected_episodes = episode_indices if episode_indices is not None else list(range(num_episodes))

    # Split episodes across GPUs
    episodes_per_process = np.array_split(selected_episodes, accelerator.num_processes)
    local_episodes = episodes_per_process[accelerator.process_index].tolist()

    if accelerator.is_main_process:
        logging.info(f"Selected {len(selected_episodes)} episode(s)")
        if stride > 1:
            logging.info(f"Using stride={stride}: computing every {stride} frames, interpolating the rest")
        if num_workers > 1:
            logging.info(f"Using threaded frame prefetch: num_workers={num_workers}, prefetch_size={prefetch_size}")

    logging.info(f"Process {accelerator.process_index}: processing {len(local_episodes)} episodes")

    # Determine which heads to compute
    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    # Local storage for this process
    local_indices = []
    local_episode_indices = []
    local_frame_indices = []
    local_progress_sparse = [] if compute_sparse else None
    local_progress_dense = [] if compute_dense else None

    # Process local episodes
    for episode_idx in tqdm(
        local_episodes,
        desc=f"GPU {accelerator.process_index}",
        disable=not accelerator.is_local_main_process
    ):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        task = task_override if task_override is not None else dataset[ep_start].get("task", "perform the task")

        # Generate frames to compute
        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        center_idx = reward_model.config.n_obs_steps // 2
        frame_results = {}

        sample_iter = iter_samples_with_prefetch(
            dataset=dataset,
            frame_indices=compute_indices,
            num_workers=num_workers,
            prefetch_size=prefetch_size,
        )

        for query_idx, sample in sample_iter:
            try:
                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]

                with torch.no_grad():
                    processed = preprocess(batch)
                    video_features = processed["video_features"].to(accelerator.device)
                    text_features = processed["text_features"].to(accelerator.device)
                    state_features = processed.get("state_features")
                    if state_features is not None:
                        state_features = state_features.to(accelerator.device)
                    lengths = processed.get("lengths")

                    sparse_val = np.nan
                    dense_val = np.nan

                    if compute_sparse:
                        sparse_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="sparse",
                        )
                        sparse_val = float(
                            sparse_progress[0, center_idx]
                            if sparse_progress.ndim == 2
                            else sparse_progress[center_idx]
                        )

                    if compute_dense:
                        dense_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="dense",
                        )
                        dense_val = float(
                            dense_progress[0, center_idx]
                            if dense_progress.ndim == 2
                            else dense_progress[center_idx]
                        )

                    frame_results[query_idx] = (sparse_val, dense_val)

            except Exception as e:
                logging.warning(f"Process {accelerator.process_index}: Failed to process frame {query_idx}: {e}")

        # Interpolate to get values for all frames
        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        # Store results
        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            local_indices.append(frame_idx)
            local_episode_indices.append(episode_idx)
            local_frame_indices.append(local_idx)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    local_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    local_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    local_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    local_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    local_progress_dense.append(frame_results[frame_idx][1])
                else:
                    local_progress_dense.append(np.nan)

    # Gather results from all processes
    all_indices = accelerator.gather_for_metrics(torch.tensor(local_indices, dtype=torch.int64)).cpu().numpy()
    all_episode_indices = accelerator.gather_for_metrics(torch.tensor(local_episode_indices, dtype=torch.int64)).cpu().numpy()
    all_frame_indices = accelerator.gather_for_metrics(torch.tensor(local_frame_indices, dtype=torch.int64)).cpu().numpy()

    if compute_sparse:
        all_progress_sparse = accelerator.gather_for_metrics(
            torch.tensor(local_progress_sparse, dtype=torch.float32)
        ).cpu().numpy()
    else:
        all_progress_sparse = None

    if compute_dense:
        all_progress_dense = accelerator.gather_for_metrics(
            torch.tensor(local_progress_dense, dtype=torch.float32)
        ).cpu().numpy()
    else:
        all_progress_dense = None

    # Only main process saves results
    if accelerator.is_main_process:
        # Create output table
        table_data = {
            "index": all_indices.astype(np.int64),
            "episode_index": all_episode_indices.astype(np.int64),
            "frame_index": all_frame_indices.astype(np.int64),
        }
        if compute_sparse:
            table_data["progress_sparse"] = all_progress_sparse.astype(np.float32)
        if compute_dense:
            table_data["progress_dense"] = all_progress_dense.astype(np.float32)

        # Sort by index
        df = pa.table(table_data).to_pandas()
        df = df.sort_values("index").reset_index(drop=True)
        final_table = pa.Table.from_pandas(df, preserve_index=False)

        # Add metadata
        metadata = {b"reward_model_path": reward_model_path.encode()}
        final_table = final_table.replace_schema_metadata(metadata)

        # Determine output path
        output_path = Path(dataset.root) / "sarm_progress.parquet" if output_path is None else Path(output_path)

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

        # Visualize episodes
        if num_visualizations > 0:
            if episode_indices is not None:
                viz_episodes = episode_indices
            else:
                viz_episodes = list(range(min(num_visualizations, num_episodes)))
            logging.info(f"Generating {len(viz_episodes)} visualizations...")
            visualize_sarm_predictions(
                dataset=dataset,
                reward_model=reward_model.module if hasattr(reward_model, 'module') else reward_model,
                preprocess=preprocess,
                episode_indices=viz_episodes,
                head_mode=head_mode,
                output_dir=Path(output_dir),
                stride=stride,
                task_override=task_override,
            )

        return output_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU distributed SARM progress computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--reward-model-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--head-mode", type=str, default="sparse", choices=["sparse", "dense", "both"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument("--episode-indices", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./sarm_viz")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-size", type=int, default=32)
    parser.add_argument("--push-to-hub", action="store_true", default=False)

    args = parser.parse_args()

    # Get reward model path
    reward_model_path = args.reward_model_path
    resolved_repo_id, dataset_root = resolve_dataset_reference(args.dataset_repo_id)
    if reward_model_path is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        temp_dataset = LeRobotDataset(resolved_repo_id, root=dataset_root, download_videos=False)
        parquet_path = Path(temp_dataset.root) / "sarm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if not reward_model_path:
            raise ValueError("--reward-model-path is required")

    # Parse episode indices
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    temp_dataset = LeRobotDataset(resolved_repo_id, root=dataset_root, download_videos=False)
    selected_episode_indices = parse_episode_indices(args.episode_indices, temp_dataset.num_episodes)

    # Run distributed computation
    output_path = compute_sarm_progress_distributed(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        episode_indices=selected_episode_indices,
        task_override=args.task,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
    )

    # Only main process handles Hub upload
    accelerator = Accelerator()
    if accelerator.is_main_process and output_path:
        print(f"\nSARM progress values saved to: {output_path}")

        if args.push_to_hub:
            if dataset_root is not None:
                print("\nSkipping Hub upload for local dataset input.")
            else:
                from huggingface_hub import HfApi
                api = HfApi()
                hub_path = "sarm_progress.parquet"
                print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
                api.upload_file(
                    path_or_fileobj=str(output_path),
                    path_in_repo=hub_path,
                    repo_id=args.dataset_repo_id,
                    repo_type="dataset",
                )
                print(f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}")


if __name__ == "__main__":
    main()
