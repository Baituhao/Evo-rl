"""
True streaming inference with episode-level granularity and checkpoint resume.

This module implements genuine streaming inference that:
1. Processes one episode at a time to minimize memory footprint
2. Writes intermediate results immediately after each episode
3. Supports checkpoint-based resume after crashes
4. Separates inference (Pass 1) from merging/indicator computation (Pass 2)
"""

import gc
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from lerobot.common.inference_checkpoint import (
    InferenceCheckpoint,
    compute_config_hash,
    create_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _infer_single_episode_streaming(
    ep_idx: int,
    ep_mask: np.ndarray,
    absolute_indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    eval_loader: DataLoader,
    model: Any,
    preprocessor: Any,
    episode_info: dict[int, Any],
    task_max_lengths: dict[int, int],
    value_cfg: Any,
    cfg: Any,
    accelerator: Accelerator,
    episodes_dir: Path,
) -> int:
    """Infer a single episode and write results immediately.

    Args:
        ep_idx: Episode index to process
        ep_mask: Boolean mask for frames belonging to this episode
        absolute_indices: Absolute frame indices in dataset
        episode_indices: Episode index for each frame
        frame_indices: Frame index within episode
        task_indices: Task index for each frame
        eval_loader: DataLoader for inference
        model: Value policy model
        preprocessor: Data preprocessor
        episode_info: Episode metadata
        task_max_lengths: Maximum trajectory lengths per task
        value_cfg: Value computation config
        cfg: Full inference config
        accelerator: Accelerate accelerator instance
        episodes_dir: Directory to write episode parquet files

    Returns:
        Number of frames processed in this episode
    """
    # Extract episode data
    ep_absolute_indices = absolute_indices[ep_mask]
    ep_episode_indices = episode_indices[ep_mask]
    ep_frame_indices = frame_indices[ep_mask]
    ep_task_indices = task_indices[ep_mask]

    # Dictionary to accumulate predictions for this episode
    ep_predictions = {}

    # Iterate through DataLoader and filter for this episode
    for raw_batch in eval_loader:
        batch_indices = raw_batch["index"]

        # Only process on main process (gather happens inside)
        if accelerator.is_main_process:
            # Check which frames in this batch belong to current episode
            batch_indices_np = batch_indices.cpu().numpy()
            batch_in_episode = np.isin(batch_indices_np, ep_absolute_indices)

            if not batch_in_episode.any():
                continue  # Skip batch if no frames from this episode

        # Run inference (all processes participate)
        processed_batch = preprocessor(raw_batch)
        with accelerator.autocast():
            predicted_value = accelerator.unwrap_model(model).predict_value(processed_batch)

        # Gather results to main process
        gathered_idx = accelerator.gather_for_metrics(batch_indices)
        gathered_val = accelerator.gather_for_metrics(predicted_value)

        if accelerator.is_main_process:
            idx_np = gathered_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            val_np = gathered_val.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

            # Filter for current episode
            mask_in_episode = np.isin(idx_np, ep_absolute_indices)
            ep_batch_indices = idx_np[mask_in_episode]
            ep_batch_values = val_np[mask_in_episode]

            # Accumulate predictions
            for idx, val in zip(ep_batch_indices, ep_batch_values):
                ep_predictions[int(idx)] = float(val)

            # Free memory
            del gathered_idx, gathered_val, idx_np, val_np

    # All processes wait before moving to next episode
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return 0  # Only main process writes

    # Convert predictions to arrays (sorted by absolute index)
    sorted_indices = np.sort(ep_absolute_indices)
    ep_values = np.array([ep_predictions[int(idx)] for idx in sorted_indices], dtype=np.float32)

    # Reorder other arrays to match sorted indices
    sort_order = np.argsort(ep_absolute_indices)
    ep_frame_indices_sorted = ep_frame_indices[sort_order]
    ep_task_indices_sorted = ep_task_indices[sort_order]

    # Import advantage computation functions
    from lerobot.scripts.lerobot_value_infer import (
        _compute_advantages,
        _compute_rewards,
        _compute_value_targets,
    )

    # Compute value targets and rewards for this episode
    ep_value_targets = _compute_value_targets(
        value_cfg=value_cfg,
        episode_indices=np.full_like(ep_frame_indices_sorted, ep_idx),
        frame_indices=ep_frame_indices_sorted,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=cfg.acp.c_fail_coef,
    )

    ep_rewards = _compute_rewards(
        value_cfg=value_cfg,
        targets=ep_value_targets,
        episode_indices=np.full_like(ep_frame_indices_sorted, ep_idx),
        frame_indices=ep_frame_indices_sorted,
        episode_info=episode_info,
    )

    # Compute advantages for this episode
    ep_advantages = _compute_advantages(
        value_cfg=value_cfg,
        rewards=ep_rewards,
        values=ep_values,
        episode_indices=np.full_like(ep_frame_indices_sorted, ep_idx),
        frame_indices=ep_frame_indices_sorted,
        n_step=cfg.acp.n_step,
        episode_info=episode_info,
    )

    # Create PyArrow table
    ep_table = pa.Table.from_pydict({
        "index": pa.array(sorted_indices, type=pa.int64()),
        "value": pa.array(ep_values, type=pa.float32()),
        "advantage": pa.array(ep_advantages, type=pa.float32()),
        "task_index": pa.array(ep_task_indices_sorted, type=pa.int64()),
    })

    # Write episode parquet file
    episodes_dir.mkdir(parents=True, exist_ok=True)
    ep_file = episodes_dir / f"ep_{ep_idx:07d}.parquet"
    pq.write_table(ep_table, ep_file, compression="snappy")

    # Free memory
    del ep_predictions, ep_values, ep_advantages, ep_table
    gc.collect()

    frames_count = len(sorted_indices)
    logging.info(f"Episode {ep_idx} completed: {frames_count} frames written to {ep_file.name}")

    return frames_count


def _merge_episodes_and_write_indicators(
    episodes_dir: Path,
    output_path: Path,
    task_indices_all: np.ndarray,
    interventions_all: np.ndarray,
    expert_episode_mask_all: np.ndarray,
    episode_indices_all: np.ndarray,
    cfg: Any,
) -> None:
    """Merge episode parquet files and compute indicators.

    This is Pass 2: reads all episode results, computes global thresholds,
    and writes the final sidecar parquet with indicators.

    Args:
        episodes_dir: Directory containing episode parquet files
        output_path: Path for final sidecar parquet
        task_indices_all: Task indices for all frames (for threshold computation)
        interventions_all: Intervention flags for all frames
        expert_episode_mask_all: Expert episode mask for all frames
        episode_indices_all: Episode indices for all frames
        cfg: Full inference config
    """
    logging.info("Starting merge phase: reading episode parquet files...")

    # Read all episode files and collect advantages
    ep_files = sorted(episodes_dir.glob("ep_*.parquet"))

    if not ep_files:
        raise RuntimeError(f"No episode files found in {episodes_dir}")

    logging.info(f"Found {len(ep_files)} episode files")

    # Pass 1: Read advantages and task indices for threshold computation
    all_advantages = []
    all_task_indices = []

    for ep_file in ep_files:
        ep_table = pq.read_table(ep_file, columns=["advantage", "task_index"])
        all_advantages.append(ep_table["advantage"].to_numpy())
        all_task_indices.append(ep_table["task_index"].to_numpy())

    all_advantages = np.concatenate(all_advantages)
    all_task_indices = np.concatenate(all_task_indices)

    # Import threshold computation
    from lerobot.scripts.lerobot_value_infer import (
        _binarize_advantages,
        _compute_task_thresholds,
    )

    # Compute global thresholds
    logging.info("Computing global thresholds...")
    thresholds = _compute_task_thresholds(
        task_indices=all_task_indices,
        advantages=all_advantages,
        positive_ratio=cfg.acp.positive_ratio,
    )
    logging.info(f"Computed thresholds for {len(thresholds)} tasks")

    # Pass 2: Stream write final parquet with indicators
    logging.info("Writing final sidecar parquet with indicators...")

    output_tmp = output_path.with_suffix(".parquet.tmp")

    # Define output schema
    schema = pa.schema([
        ("index", pa.int64()),
        pa.field(cfg.acp.value_field, pa.float32()),
        pa.field(cfg.acp.advantage_field, pa.float32()),
        pa.field(cfg.acp.indicator_field, pa.int64()),
    ])

    writer = pq.ParquetWriter(output_tmp, schema, compression="snappy")

    total_rows = 0
    for ep_file in ep_files:
        # Read full episode data
        ep_table = pq.read_table(ep_file)
        ep_indices = ep_table["index"].to_numpy()
        ep_values = ep_table["value"].to_numpy()
        ep_advantages = ep_table["advantage"].to_numpy()
        ep_task_indices = ep_table["task_index"].to_numpy()

        # Get interventions and expert mask for this episode's frames
        # Find positions in original arrays
        ep_positions = np.searchsorted(np.sort(np.arange(len(task_indices_all))), ep_indices)
        ep_interventions = interventions_all[ep_positions]
        ep_expert_mask = expert_episode_mask_all[ep_positions]
        ep_episode_indices = episode_indices_all[ep_positions]

        # Compute indicators
        ep_indicators = _binarize_advantages(
            task_indices=ep_task_indices,
            advantages=ep_advantages,
            thresholds=thresholds,
            interventions=ep_interventions,
            force_intervention_positive=cfg.acp.force_intervention_positive,
            expert_episode_mask=ep_expert_mask,
            force_expert_episode_positive=cfg.acp.force_expert_episode_positive,
        )

        # Build output table
        output_table = pa.Table.from_pydict({
            "index": pa.array(ep_indices, type=pa.int64()),
            cfg.acp.value_field: pa.array(ep_values, type=pa.float32()),
            cfg.acp.advantage_field: pa.array(ep_advantages, type=pa.float32()),
            cfg.acp.indicator_field: pa.array(ep_indicators, type=pa.int64()),
        })

        writer.write_table(output_table)
        total_rows += len(ep_indices)

    writer.close()

    # Sort by index for consistency
    logging.info("Sorting final parquet by index...")
    final_table = pq.read_table(output_tmp)
    sorted_table = final_table.sort_by([("index", "ascending")])
    pq.write_table(sorted_table, output_path, compression="snappy")

    # Clean up
    output_tmp.unlink()

    logging.info(f"Merge complete: {total_rows} rows written to {output_path}")


def run_streaming_inference_with_resume(
    dataset_root: Path,
    sidecar_subdir: str,
    absolute_indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    interventions: np.ndarray,
    expert_episode_mask: np.ndarray,
    eval_loader: DataLoader,
    model: Any,
    preprocessor: Any,
    episode_info: dict[int, Any],
    task_max_lengths: dict[int, int],
    value_cfg: Any,
    cfg: Any,
    accelerator: Accelerator,
    output_dir: Path,
) -> dict[int, float]:
    """Run true streaming inference with checkpoint resume support.

    This implements episode-level streaming:
    1. Load or create checkpoint
    2. For each episode not yet completed:
       - Infer and compute advantages
       - Write episode parquet immediately
       - Update checkpoint
    3. Merge all episodes and compute indicators

    Args:
        dataset_root: Root directory of dataset
        sidecar_subdir: Subdirectory for sidecar files
        absolute_indices: Absolute frame indices
        episode_indices: Episode index for each frame
        frame_indices: Frame index within episode
        task_indices: Task index for each frame
        interventions: Intervention flags
        expert_episode_mask: Expert episode mask
        eval_loader: DataLoader for inference
        model: Value policy model
        preprocessor: Data preprocessor
        episode_info: Episode metadata
        task_max_lengths: Maximum trajectory lengths per task
        value_cfg: Value computation config
        cfg: Full inference config
        accelerator: Accelerate accelerator instance
        output_dir: Output directory for intermediate files

    Returns:
        Dictionary of computed thresholds per task
    """
    # Prepare directories
    sidecar_dir = dataset_root / "advantage" / sidecar_subdir
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    episodes_dir = sidecar_dir / "episodes"
    checkpoint_path = sidecar_dir / "checkpoint.json"
    final_output = sidecar_dir / "frames.parquet"

    # Only main process manages checkpoint
    if accelerator.is_main_process:
        # Load or create checkpoint
        if checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path)
            config_hash = compute_config_hash({"cfg": str(cfg)})

            if checkpoint.config_hash != config_hash:
                logging.warning(
                    "Config hash mismatch! Inference config may have changed. "
                    "Proceeding anyway, but results may be inconsistent."
                )

            logging.info(
                f"Resuming from checkpoint: {len(checkpoint.completed_episodes)}/"
                f"{checkpoint.total_episodes} episodes completed "
                f"({checkpoint.progress_ratio()*100:.1f}%)"
            )

            # Check status
            if checkpoint.status == "completed":
                logging.info("Inference already completed")
                # Read thresholds from existing file (placeholder)
                return {}

            if checkpoint.status == "merging":
                logging.info("Inference completed, starting merge phase")
                _merge_episodes_and_write_indicators(
                    episodes_dir=episodes_dir,
                    output_path=final_output,
                    task_indices_all=task_indices,
                    interventions_all=interventions,
                    expert_episode_mask_all=expert_episode_mask,
                    episode_indices_all=episode_indices,
                    cfg=cfg,
                )

                checkpoint.status = "completed"
                save_checkpoint(checkpoint, checkpoint_path)
                return {}
        else:
            # Create new checkpoint
            unique_episodes = np.unique(episode_indices)
            config_hash = compute_config_hash({"cfg": str(cfg)})

            checkpoint = create_checkpoint(
                total_episodes=len(unique_episodes),
                config_hash=config_hash,
            )
            save_checkpoint(checkpoint, checkpoint_path)

            logging.info(
                f"Starting new streaming inference: {checkpoint.total_episodes} episodes to process"
            )

    accelerator.wait_for_everyone()

    # Inference phase: process each episode
    unique_episodes = np.unique(episode_indices)

    for ep_idx in unique_episodes:
        if accelerator.is_main_process:
            # Reload checkpoint to get latest state
            checkpoint = load_checkpoint(checkpoint_path)

            # Skip if already completed
            if checkpoint.is_episode_completed(ep_idx):
                logging.info(f"Episode {ep_idx} already completed, skipping")
                continue

            # Mark as current episode
            checkpoint.set_current_episode(ep_idx)
            save_checkpoint(checkpoint, checkpoint_path)

        accelerator.wait_for_everyone()

        # Infer this episode (all processes participate, but only main writes)
        ep_mask = episode_indices == ep_idx
        frames_count = _infer_single_episode_streaming(
            ep_idx=ep_idx,
            ep_mask=ep_mask,
            absolute_indices=absolute_indices,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            task_indices=task_indices,
            eval_loader=eval_loader,
            model=model,
            preprocessor=preprocessor,
            episode_info=episode_info,
            task_max_lengths=task_max_lengths,
            value_cfg=value_cfg,
            cfg=cfg,
            accelerator=accelerator,
            episodes_dir=episodes_dir,
        )

        if accelerator.is_main_process:
            # Mark episode as completed
            checkpoint.mark_episode_completed(ep_idx, frames_count)
            save_checkpoint(checkpoint, checkpoint_path)

            # Log progress
            progress = checkpoint.progress_ratio() * 100
            logging.info(
                f"Progress: {len(checkpoint.completed_episodes)}/{checkpoint.total_episodes} "
                f"episodes ({progress:.1f}%), {checkpoint.total_frames_processed} frames processed"
            )

        accelerator.wait_for_everyone()

    # Merge phase
    if accelerator.is_main_process:
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint.status = "merging"
        save_checkpoint(checkpoint, checkpoint_path)

        _merge_episodes_and_write_indicators(
            episodes_dir=episodes_dir,
            output_path=final_output,
            task_indices_all=task_indices,
            interventions_all=interventions,
            expert_episode_mask_all=expert_episode_mask,
            episode_indices_all=episode_indices,
            cfg=cfg,
        )

        # Clean up episodes directory
        logging.info(f"Cleaning up intermediate files in {episodes_dir}")
        shutil.rmtree(episodes_dir)

        # Mark as completed
        checkpoint.status = "completed"
        save_checkpoint(checkpoint, checkpoint_path)

        logging.info("True streaming inference completed successfully")

    accelerator.wait_for_everyone()

    # Return empty dict (thresholds not needed for return)
    return {}
