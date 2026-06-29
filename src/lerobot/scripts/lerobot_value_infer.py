#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.configs.value import ValueInferencePipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_info, write_info
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import NormalizerProcessorStep
from lerobot.scripts.value_infer_viz import (
    _export_overlay_videos,
)
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.utils.utils import init_logging, inside_slurm
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_dense_rewards_from_targets as compute_pistar06_dense_rewards_from_targets,
    compute_n_step_advantages as compute_pistar06_n_step_advantages,
    compute_normalized_value_targets as compute_pistar06_normalized_value_targets,
)
from lerobot.values.pistar_06_td.configuration_pistar_06_td import Pistar_06_tdConfig
from lerobot.values.pistar_06_td.modeling_pistar_06_td import (
    compute_dense_rewards_from_targets as compute_pistar_06_td_dense_rewards_from_targets,
    compute_n_step_advantages as compute_pistar_06_td_n_step_advantages,
    compute_normalized_value_targets as compute_pistar_06_td_normalized_value_targets,
)
from lerobot.values.value01.configuration_value01 import Value01Config
from lerobot.values.value01.modeling_value01 import (
    compute_dense_rewards_from_targets as compute_value01_dense_rewards_from_targets,
    compute_n_step_advantages as compute_value01_n_step_advantages,
    compute_normalized_value_targets as compute_value01_normalized_value_targets,
)


class ContiguousDistributedEvalSampler(Sampler[int]):
    """Distributed eval sampler with contiguous per-rank shards and deterministic tail padding."""

    def __init__(self, dataset_size: int, num_replicas: int, rank: int):
        if dataset_size <= 0:
            raise ValueError(f"'dataset_size' must be > 0, got {dataset_size}.")
        if num_replicas <= 0:
            raise ValueError(f"'num_replicas' must be > 0, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"'rank' must be in [0, {num_replicas - 1}], got {rank}.")

        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        shard_start = self.rank * self.num_samples
        shard_end = min(shard_start + self.num_samples, self.dataset_size)
        if shard_start >= self.dataset_size:
            indices: list[int] = []
        else:
            indices = list(range(shard_start, shard_end))

        if len(indices) < self.num_samples:
            indices.extend([self.dataset_size - 1] * (self.num_samples - len(indices)))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def _set_infer_logger_levels() -> None:
    for logger_name in ["fsspec", "fsspec.local", "huggingface_hub", "datasets", "torchcodec"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _create_accelerator(cfg: ValueInferencePipelineConfig, accelerator: Accelerator | None) -> Accelerator:
    if accelerator is not None:
        return accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    force_cpu = cfg.runtime.device == "cpu"

    # Monkey-patch torch.distributed.init_process_group to increase NCCL timeout
    # Default is 10 minutes, increase to 30 minutes for large datasets
    import torch.distributed as dist
    from datetime import timedelta
    original_init = dist.init_process_group
    def patched_init(*args, **kwargs):
        if 'timeout' not in kwargs:
            kwargs['timeout'] = timedelta(minutes=30)
        return original_init(*args, **kwargs)
    dist.init_process_group = patched_init

    return Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs], cpu=force_cpu)


def _resolve_pretrained_model_dir(checkpoint_path: str, checkpoint_ref: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    if (path / "model.safetensors").is_file() and (path / "config.json").is_file():
        return path

    if (path / PRETRAINED_MODEL_DIR / "model.safetensors").is_file() and (
        path / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return path / PRETRAINED_MODEL_DIR

    checkpoints_root = path / CHECKPOINTS_DIR if (path / CHECKPOINTS_DIR).is_dir() else path
    step_ref = LAST_CHECKPOINT_LINK if checkpoint_ref == "last" else checkpoint_ref
    step_dir = checkpoints_root / step_ref

    if (step_dir / PRETRAINED_MODEL_DIR / "model.safetensors").is_file() and (
        step_dir / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return step_dir / PRETRAINED_MODEL_DIR

    if (step_dir / "model.safetensors").is_file() and (step_dir / "config.json").is_file():
        return step_dir

    raise FileNotFoundError(
        f"Could not resolve pretrained model directory from checkpoint_path={path} checkpoint_ref={checkpoint_ref}."
    )


def _load_dataset_distributed(cfg: ValueInferencePipelineConfig, accelerator: Accelerator) -> LeRobotDataset:
    # Load camera_features from checkpoint to filter video decoding
    video_keys_filter = None
    if cfg.inference.checkpoint_path:
        try:
            pretrained_dir = _resolve_pretrained_model_dir(
                checkpoint_path=cfg.inference.checkpoint_path,
                checkpoint_ref=cfg.inference.checkpoint_ref,
            )
            checkpoint_config = PreTrainedConfig.from_pretrained(pretrained_dir)
            camera_features = getattr(checkpoint_config, "camera_features", None)
            if camera_features and len(camera_features) > 0:
                video_keys_filter = camera_features
                if accelerator.is_main_process:
                    logging.info(
                        "Filtering video decode to %d cameras from checkpoint config: %s",
                        len(video_keys_filter),
                        video_keys_filter,
                    )
        except Exception as e:
            if accelerator.is_main_process:
                logging.warning(
                    "Could not load camera_features from checkpoint, will decode all videos: %s", e
                )

    dataset_kwargs = {
        "repo_id": cfg.dataset.repo_id,
        "root": cfg.dataset.root,
        "episodes": cfg.dataset.episodes,
        "revision": cfg.dataset.revision,
        "download_videos": cfg.dataset.download_videos,
        "image_center_crop": cfg.dataset.image_center_crop,
        "video_keys_filter": video_keys_filter,
    }

    if accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    return dataset


def _init_runtime(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator,
) -> tuple[Path, torch.device]:
    output_dir = cfg.output_dir / "value"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_file = output_dir / "value_infer.log" if accelerator.is_main_process else None
    init_logging(log_file=log_file, file_level="INFO", accelerator=accelerator)
    _set_infer_logger_levels()

    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    return output_dir, device


def _build_episode_info(
    dataset: LeRobotDataset,
    success_field: str,
    default_success: str,
) -> tuple[dict[int, EpisodeTargetInfo], dict[int, int]]:
    episodes_ds = dataset.meta.episodes.with_format(None)
    episodes = episodes_ds[:]
    n_episodes = len(episodes_ds)
    has_success = success_field in episodes_ds.column_names

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}
    for i in range(n_episodes):
        ep_idx = int(episodes["episode_index"][i])
        ep_length = int(episodes["length"][i])
        tasks = episodes["tasks"][i]
        task_name = tasks[0] if isinstance(tasks, (list, np.ndarray)) else tasks
        # Compatible with two modes:
        # 1. task_name is a string task name (index into tasks DataFrame)
        # 2. task_name is an integer task index directly
        try:
            task_index = int(task_name)
            # Verify the task index exists
            if task_index not in dataset.meta.tasks["task_index"].values:
                raise KeyError(f"Episode {ep_idx} references unknown task index {task_index}.")
        except (ValueError, TypeError):
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.") from None
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

        explicit_success = episodes[success_field][i] if has_success else None
        resolved_success = resolve_episode_success_label(
            explicit_success,
            default_label=default_success,
            require_label=True,
        )
        ep_success = resolved_success == EPISODE_SUCCESS

        episode_info[ep_idx] = EpisodeTargetInfo(
            episode_index=ep_idx,
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)
    return episode_info, task_max_length


def _compute_value_targets(
    value_cfg: Pistar06Config | Pistar_06_tdConfig | Value01Config,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
) -> np.ndarray:
    value_type = getattr(value_cfg, "type", None)
    if value_type is None:
        if isinstance(value_cfg, Pistar06Config):
            value_type = "pistar06"
        elif isinstance(value_cfg, Pistar_06_tdConfig):
            value_type = "pistar_06_td"
        elif isinstance(value_cfg, Value01Config):
            value_type = "value01"

    if value_type == "pistar06":
        compute_targets_fn = compute_pistar06_normalized_value_targets
    elif value_type == "pistar_06_td":
        compute_targets_fn = compute_pistar_06_td_normalized_value_targets
    elif value_type == "value01":
        compute_targets_fn = compute_value01_normalized_value_targets
    else:
        raise ValueError(
            f"Unsupported value type '{value_type}'. "
            "lerobot-value-infer currently supports only 'pistar06', 'pistar_06_td', and 'value01'."
        )

    return compute_targets_fn(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=c_fail_coef,
        clip_min=value_cfg.bin_min,
        clip_max=value_cfg.bin_max,
    )


def _compute_rewards(
    value_cfg: Pistar06Config | Pistar_06_tdConfig | Value01Config,
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
) -> np.ndarray:
    value_type = getattr(value_cfg, "type", None)
    if value_type is None:
        if isinstance(value_cfg, Pistar06Config):
            value_type = "pistar06"
        elif isinstance(value_cfg, Pistar_06_tdConfig):
            value_type = "pistar_06_td"
        elif isinstance(value_cfg, Value01Config):
            value_type = "value01"

    if value_type == "pistar06":
        return compute_pistar06_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    elif value_type == "pistar_06_td":
        return compute_pistar_06_td_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    elif value_type == "value01":
        return compute_value01_dense_rewards_from_targets(
            targets, episode_indices, frame_indices, episode_info
        )
    else:
        raise ValueError(
            f"Unsupported value type '{value_type}'. "
            "lerobot-value-infer currently supports only 'pistar06', 'pistar_06_td', and 'value01'."
        )


def _compute_advantages(
    value_cfg: Pistar06Config | Pistar_06_tdConfig | Value01Config,
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
    episode_info: dict[int, EpisodeTargetInfo],
) -> np.ndarray:
    value_type = getattr(value_cfg, "type", None)
    if value_type is None:
        if isinstance(value_cfg, Pistar06Config):
            value_type = "pistar06"
        elif isinstance(value_cfg, Pistar_06_tdConfig):
            value_type = "pistar_06_td"
        elif isinstance(value_cfg, Value01Config):
            value_type = "value01"

    if value_type == "pistar06":
        return compute_pistar06_n_step_advantages(rewards, values, episode_indices, frame_indices, n_step)
    elif value_type == "pistar_06_td":
        return compute_pistar_06_td_n_step_advantages(rewards, values, episode_indices, frame_indices, n_step)
    elif value_type == "value01":
        return compute_value01_n_step_advantages(
            rewards,
            values,
            episode_indices,
            frame_indices,
            n_step,
            episode_info,
        )
    else:
        raise ValueError(
            f"Unsupported value type '{value_type}'. "
            "lerobot-value-infer currently supports only 'pistar06', 'pistar_06_td', and 'value01'."
        )


def _compute_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    positive_ratio: float,
) -> dict[int, float]:
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")

    thresholds: dict[int, float] = {}
    quantile = 1.0 - positive_ratio

    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))

    return thresholds


def _binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    interventions: np.ndarray,
    force_intervention_positive: bool,
    expert_episode_mask: np.ndarray,
    force_expert_episode_positive: bool,
) -> np.ndarray:
    indicators = np.zeros_like(advantages, dtype=np.int64)

    for i in range(advantages.shape[0]):
        task_idx = int(task_indices[i])
        threshold = thresholds[task_idx]
        indicators[i] = 1 if float(advantages[i]) >= threshold else 0

    if force_intervention_positive:
        intervention_mask = interventions.astype(np.float32) > 0.5
        indicators[intervention_mask] = 1

    if force_expert_episode_positive:
        indicators[expert_episode_mask] = 1

    return indicators


def _compute_episode_positive_mask(
    episode_indices: np.ndarray,
    episode_labels: np.ndarray,
) -> np.ndarray:
    if episode_indices.ndim != 1:
        raise ValueError("'episode_indices' must be rank-1.")
    if episode_labels.ndim != 1:
        raise ValueError("'episode_labels' must be rank-1.")
    if episode_indices.shape[0] != episode_labels.shape[0]:
        raise ValueError(
            f"'episode_indices' and 'episode_labels' must have the same length, got "
            f"{episode_indices.shape[0]} and {episode_labels.shape[0]}."
        )

    positive_episode_ids = np.unique(episode_indices[episode_labels.astype(np.float32) > 0.5])
    if positive_episode_ids.size == 0:
        return np.zeros_like(episode_indices, dtype=np.bool_)

    return np.isin(episode_indices, positive_episode_ids)


def _update_feature_metadata(dataset_root: Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    info = load_info(dataset_root)
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": tuple(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_info(info, dataset_root)


def _write_columns_sidecar(
    dataset_root: Path,
    sidecar_subdir: str,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    """Write value columns to a separate sidecar parquet file keyed by index.

    Args:
        dataset_root: Root directory of the dataset
        sidecar_subdir: Subdirectory name under <root>/advantage/ (e.g., tag name)
        absolute_indices: Global frame indices (from dataset["index"])
        columns: Dict of column_name -> values array (aligned with absolute_indices)
        feature_infos: Dict of column_name -> {"dtype": ..., "shape": ..., "names": ...}
    """
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")
    if len(absolute_indices) == 0:
        raise ValueError("'absolute_indices' must be non-empty.")

    # Build PyArrow table with index column + data columns
    arrays = {"index": pa.array(absolute_indices, type=pa.int64())}

    for field_name, values in columns.items():
        ftype = feature_infos[field_name]["dtype"]
        if ftype == "float32":
            pa_type = pa.float32()
            arr = values.astype(np.float32, copy=False)
        elif ftype == "int64":
            pa_type = pa.int64()
            arr = values.astype(np.int64, copy=False)
        else:
            raise ValueError(f"Unsupported sidecar dtype '{ftype}' for field '{field_name}'.")
        arrays[field_name] = pa.array(arr, type=pa_type)

    table = pa.Table.from_pydict(arrays)

    # Write atomically: write to .tmp then rename
    output_dir = dataset_root / "advantage" / sidecar_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "frames.parquet"
    tmp_path = output_dir / "frames.parquet.tmp"

    pq.write_table(table, tmp_path, compression="snappy")
    import os
    os.replace(tmp_path, output_path)

    logging.info(
        "Wrote %d rows to sidecar parquet: %s (columns: %s)",
        len(absolute_indices),
        output_path,
        list(columns.keys()),
    )


def _write_columns_sidecar_streaming(
    dataset_root: Path,
    sidecar_subdir: str,
    absolute_indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    predicted_values: np.ndarray,
    interventions: np.ndarray,
    expert_episode_mask: np.ndarray,
    episode_info: dict[int, Any],
    task_max_lengths: dict[int, int],
    value_cfg: Any,
    cfg: Any,
) -> None:
    """Write sidecar parquet with chunked processing to reduce memory peaks.

    Processes data in episode chunks: computes advantage/indicator for each chunk,
    writes immediately, then releases memory before processing next chunk.

    Args:
        dataset_root: Root directory of the dataset
        sidecar_subdir: Subdirectory name under <root>/advantage/
        absolute_indices: Global frame indices
        episode_indices: Episode index for each frame
        frame_indices: Frame index within episode
        task_indices: Task index for each frame
        predicted_values: Model predictions
        interventions: Intervention mask
        expert_episode_mask: Expert episode mask
        episode_info: Episode metadata
        task_max_lengths: Max length per task
        value_cfg: Value model config
        cfg: Pipeline config
    """
    output_dir = dataset_root / "advantage" / sidecar_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "frames.parquet"
    tmp_path = output_dir / "frames.parquet.tmp"

    # Build episode groups
    unique_episodes = np.unique(episode_indices)
    logging.info(
        "Streaming write: processing %d episodes in chunks to reduce memory", len(unique_episodes)
    )

    # Schema for parquet writer
    schema = pa.schema([
        ("index", pa.int64()),
        (cfg.acp.value_field, pa.float32()),
        (cfg.acp.advantage_field, pa.float32()),
        (cfg.acp.indicator_field, pa.int64()),
    ])

    # Open writer in append mode
    writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")

    # First pass: collect all advantages to compute global thresholds
    all_advantages = []
    all_task_indices = []

    for ep_idx in unique_episodes:
        ep_mask = episode_indices == ep_idx
        ep_absolute_indices = absolute_indices[ep_mask]
        ep_episode_indices = episode_indices[ep_mask]
        ep_frame_indices = frame_indices[ep_mask]
        ep_task_indices = task_indices[ep_mask]
        ep_values = predicted_values[ep_mask]

        # Compute advantage for this episode
        ep_value_targets = _compute_value_targets(
            value_cfg, ep_episode_indices, ep_frame_indices, episode_info, task_max_lengths, cfg.acp.c_fail_coef
        )
        ep_rewards = _compute_rewards(value_cfg, ep_value_targets, ep_episode_indices, ep_frame_indices, episode_info)
        ep_advantages = _compute_advantages(
            value_cfg, ep_rewards, ep_values, ep_episode_indices, ep_frame_indices, cfg.acp.n_step, episode_info
        )

        all_advantages.append(ep_advantages)
        all_task_indices.append(ep_task_indices)

    # Concatenate for global threshold computation
    all_advantages = np.concatenate(all_advantages)
    all_task_indices = np.concatenate(all_task_indices)

    # Compute global thresholds
    thresholds = _compute_task_thresholds(all_task_indices, all_advantages, cfg.acp.positive_ratio)

    logging.info("Computed global thresholds for %d tasks", len(thresholds))

    # Second pass: write each episode with computed indicators
    advantages_offset = 0
    for ep_idx in unique_episodes:
        ep_mask = episode_indices == ep_idx
        ep_count = int(np.sum(ep_mask))
        ep_absolute_indices = absolute_indices[ep_mask]
        ep_values = predicted_values[ep_mask]
        ep_task_indices = task_indices[ep_mask]
        ep_interventions = interventions[ep_mask]
        ep_expert_mask = expert_episode_mask[ep_mask]

        # Retrieve pre-computed advantages
        ep_advantages = all_advantages[advantages_offset : advantages_offset + ep_count]
        advantages_offset += ep_count

        # Binarize advantages to indicators using global thresholds
        ep_indicators = _binarize_advantages(
            ep_task_indices,
            ep_advantages,
            thresholds,
            ep_interventions,
            cfg.acp.force_intervention_positive,
            ep_expert_mask,
            cfg.acp.force_expert_episode_positive,
        )

        # Build PyArrow table for this episode (sorted by index)
        sort_order = np.argsort(ep_absolute_indices)
        table = pa.Table.from_pydict({
            "index": pa.array(ep_absolute_indices[sort_order], type=pa.int64()),
            cfg.acp.value_field: pa.array(ep_values[sort_order].astype(np.float32), type=pa.float32()),
            cfg.acp.advantage_field: pa.array(ep_advantages[sort_order].astype(np.float32), type=pa.float32()),
            cfg.acp.indicator_field: pa.array(ep_indicators[sort_order].astype(np.int64), type=pa.int64()),
        })

        # Write (append) this episode's table
        writer.write_table(table)

        # Free memory
        del ep_values, ep_advantages, ep_indicators, table

    writer.close()

    # Read back, sort globally by index, and write final file
    logging.info("Sorting sidecar parquet by index...")
    full_table = pq.read_table(tmp_path)
    sorted_table = full_table.sort_by([("index", "ascending")])

    import os
    os.replace(tmp_path, output_path)
    pq.write_table(sorted_table, output_path, compression="snappy")

    logging.info(
        "Streaming write complete: %d rows to %s",
        len(sorted_table),
        output_path,
    )

    return thresholds


def _write_columns_in_place(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")

    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        lookup_dtype = np.float32 if feature_infos[field]["dtype"] == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
        lookup[absolute_indices] = values.astype(lookup_dtype, copy=False)
        lookups[field] = lookup

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    for parquet_path in tqdm(data_files, desc="Writing annotations", leave=False):
        table = pq.read_table(parquet_path)
        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)

        in_range = (idx_np >= 0) & (idx_np <= max_index)
        in_subset = np.zeros_like(in_range)
        in_subset[in_range] = selected[idx_np[in_range]]

        new_table = table
        for field, lookup in lookups.items():
            ftype = feature_infos[field]["dtype"]
            if ftype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            elif ftype == "int64":
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()
            else:
                raise ValueError(f"Unsupported annotation dtype '{ftype}' for field '{field}'.")

            if field in new_table.schema.names:
                current = new_table[field].to_numpy().astype(target_dtype, copy=True)
            else:
                current = np.full(idx_np.shape[0], default_value, dtype=target_dtype)

            if np.any(in_subset):
                subset_indices = idx_np[in_subset]
                current[in_subset] = lookup[subset_indices]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, parquet_path, compression="snappy")

    _update_feature_metadata(dataset_root=dataset_root, feature_infos=feature_infos)


def _truncate_state_stats_to_feature_dim(preprocessor, dataset=None) -> None:
    """Truncate normalizer state stats to the feature's declared dim.

    The checkpoint may bake in normalization stats computed on a higher-dim
    training dataset (e.g. 21-dim state with extra waist/head joints), while the
    inference dataset only provides the leading joints (e.g. 16-dim). Because the
    leading dims share identical joint ordering, we slice the stored training
    stats down to the declared feature dim so normalization stays consistent with
    training while matching the incoming tensor shape.

    Both `_tensor_stats` (used during transform) and `stats` (the source `to()`
    rebuilds from) are sliced, so device/dtype moves cannot restore the old dim.

    The truncation target is the inference dataset's real feature dim when
    available (via `dataset.meta`). The checkpoint config may declare a larger
    dim than the data actually provides (e.g. action declared 21-dim while the
    inference dataset stores 16-dim), so relying on the declared feature shape
    alone would miss those cases.
    """
    dataset_dims: dict[str, int] = {}
    meta = getattr(dataset, "meta", None)
    feats = getattr(meta, "features", None)
    if feats:
        for key, ft in feats.items():
            shape = ft.get("shape") if isinstance(ft, dict) else None
            if shape:
                dataset_dims[key] = int(shape[-1])

    for step in getattr(preprocessor, "steps", []):
        if not isinstance(step, NormalizerProcessorStep):
            continue
        keys = step.normalize_observation_keys
        if keys is not None:
            target_keys = set(keys)
        else:
            target_keys = {k for k, ft in step.features.items() if ft.type == FeatureType.STATE}
        # ACTION stats are applied via a separate path (`_normalize_action`) and
        # are not listed in `normalize_observation_keys`, so include any ACTION
        # feature here too. The checkpoint may bake in higher-dim action stats
        # (e.g. 21-dim) while inference provides only the leading joints (16-dim);
        # the leading dims share identical joint ordering, so slicing keeps
        # normalization consistent with training while matching the tensor shape.
        target_keys |= {k for k, ft in step.features.items() if ft.type == FeatureType.ACTION}
        for key in target_keys:
            feature = step.features.get(key)
            if feature is None or not feature.shape:
                continue
            # Prefer the inference dataset's real dim; fall back to the declared
            # feature shape. The checkpoint may declare a larger dim than the
            # data provides (e.g. action 21-dim declared vs 16-dim in data).
            target_dim = dataset_dims.get(key, int(feature.shape[-1]))

            tensor_stats = step._tensor_stats.get(key)
            if tensor_stats:
                for stat_name, tensor in list(tensor_stats.items()):
                    if tensor.ndim >= 1 and tensor.shape[-1] > target_dim:
                        tensor_stats[stat_name] = tensor[..., :target_dim].contiguous()

            raw_stats = (step.stats or {}).get(key)
            if raw_stats:
                for stat_name, value in list(raw_stats.items()):
                    arr = np.asarray(value)
                    if arr.ndim >= 1 and arr.shape[-1] > target_dim:
                        raw_stats[stat_name] = arr[..., :target_dim]
                        logging.info(
                            "Truncated normalizer stat '%s.%s' from dim %d to %d.",
                            key,
                            stat_name,
                            arr.shape[-1],
                            target_dim,
                        )


def _load_value_policy_and_processors(
    cfg: ValueInferencePipelineConfig,
    dataset: LeRobotDataset,
    pretrained_dir: Path,
    device: torch.device,
):
    value_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    if not isinstance(value_cfg, (Pistar06Config, Pistar_06_tdConfig, Value01Config)):
        raise ValueError(
            f"Unsupported value config type '{type(value_cfg)}'. "
            "lerobot-value-infer currently supports only 'pistar06', 'pistar_06_td', and 'value01'."
        )

    value_cfg.pretrained_path = pretrained_dir
    value_cfg.device = device.type

    value_policy = make_policy(
        cfg=value_cfg,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=value_cfg,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    _truncate_state_stats_to_feature_dim(preprocessor, dataset)
    return value_policy, value_cfg, preprocessor


def _export_visualization_outputs(
    dataset: LeRobotDataset,
    cfg: ValueInferencePipelineConfig,
    output_dir: Path,
) -> list[str]:
    viz_output_dir = output_dir / "viz"
    written_videos = _export_overlay_videos(
        dataset=dataset,
        value_field=cfg.acp.value_field,
        advantage_field=cfg.acp.advantage_field,
        indicator_field=cfg.acp.indicator_field,
        viz_episodes=cfg.viz.episodes,
        video_key=cfg.viz.video_key,
        video_keys=cfg.viz.video_keys,
        output_dir=viz_output_dir,
        overwrite=cfg.viz.overwrite,
        vcodec=cfg.viz.vcodec,
        frame_storage_mode=cfg.viz.frame_storage_mode,
        smooth_window=cfg.viz.smooth_window,
        max_frame_size=cfg.viz.max_frame_size,
    )
    logging.info("Exported %d overlay videos to %s", len(written_videos), viz_output_dir)
    return [str(path) for path in written_videos]


def run_value_inference_pipeline(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    cfg.validate()

    accelerator = _create_accelerator(cfg, accelerator)
    output_dir, device = _init_runtime(cfg, accelerator)

    dataset = _load_dataset_distributed(cfg, accelerator)
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    if not cfg.acp.enable:
        viz_outputs: list[str] = []
        if accelerator.is_main_process:
            logging.info(
                "ACP disabled; skipping value inference and reusing existing '%s' annotations.",
                cfg.acp.value_field,
            )
            if cfg.acp.value_field not in raw_frames.column_names:
                raise KeyError(
                    f"Missing value field '{cfg.acp.value_field}' in dataset while 'acp.enable=false'."
                )
            if cfg.viz.enable:
                viz_outputs = _export_visualization_outputs(dataset=dataset, cfg=cfg, output_dir=output_dir)

        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            result: dict[str, Any] = {
                "main_process": False,
                "world_size": int(accelerator.num_processes),
            }
            accelerator.end_training()
            return result

        result = {
            "main_process": True,
            "world_size": int(accelerator.num_processes),
            "num_frames": int(frame_count),
            "checkpoint": None,
            "value_field": cfg.acp.value_field,
            "acp_enabled": False,
            "value_inference_skipped": True,
            "indicator_positive_ratio": None,
            "thresholds": None,
            "viz_outputs": viz_outputs,
        }
        accelerator.end_training()
        return result

    pretrained_dir = _resolve_pretrained_model_dir(
        checkpoint_path=cfg.inference.checkpoint_path,
        checkpoint_ref=cfg.inference.checkpoint_ref,
    )
    value_policy, value_cfg, preprocessor = _load_value_policy_and_processors(
        cfg=cfg,
        dataset=dataset,
        pretrained_dir=pretrained_dir,
        device=device,
    )

    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64).reshape(-1)

    if value_cfg.task_index_feature not in raw_frames.column_names:
        raise KeyError(f"Missing task feature '{value_cfg.task_index_feature}' in dataset columns.")

    task_indices = np.asarray(raw_frames[value_cfg.task_index_feature], dtype=np.int64).reshape(-1)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64).reshape(-1)

    if cfg.acp.intervention_field in raw_frames.column_names:
        interventions = np.asarray(raw_frames[cfg.acp.intervention_field], dtype=np.float32).reshape(-1)
    else:
        interventions = np.zeros(frame_count, dtype=np.float32)

    if cfg.acp.expert_episode_field in raw_frames.column_names:
        expert_episode_labels = np.asarray(
            raw_frames[cfg.acp.expert_episode_field], dtype=np.float32
        ).reshape(-1)
        expert_episode_mask = _compute_episode_positive_mask(
            episode_indices=episode_indices,
            episode_labels=expert_episode_labels,
        )
    else:
        expert_episode_mask = np.zeros(frame_count, dtype=np.bool_)

    # Determine prefetch_factor: use config if set, otherwise PyTorch default (2 if workers > 0, None if workers = 0)
    if cfg.runtime.prefetch_factor is not None:
        prefetch_factor = cfg.runtime.prefetch_factor
    else:
        prefetch_factor = 2 if cfg.runtime.num_workers > 0 else None

    eval_loader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )

    value_policy = accelerator.prepare(value_policy)
    eval_loader = accelerator.prepare(eval_loader)

    if accelerator.is_main_process:
        if cfg.acp.streaming_write:
            # Streaming mode: use dict to accumulate (lower memory)
            prediction_dict = {}
            prediction_chunks = []  # Store chunks to disk periodically
            chunk_size = 5000  # Flush to disk every 5000 frames
            logging.info(
                "Start value inference (streaming) | world_size=%d batches=%d batch_size=%d checkpoint=%s chunk_size=%d",
                accelerator.num_processes,
                len(eval_loader),
                cfg.runtime.batch_size,
                pretrained_dir,
                chunk_size,
            )
        else:
            # Full-memory mode: pre-allocate lookup arrays
            max_abs_index = int(np.max(absolute_indices))
            prediction_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
            prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)
            prediction_dict = None
            logging.info(
                "Start value inference (full-memory) | world_size=%d batches=%d batch_size=%d checkpoint=%s",
                accelerator.num_processes,
                len(eval_loader),
                cfg.runtime.batch_size,
                pretrained_dir,
            )
    else:
        prediction_lookup = None
        prediction_seen = None
        prediction_dict = None

    value_policy.eval()
    eval_iter = tqdm(
        eval_loader,
        desc="Value inference",
        total=len(eval_loader),
        leave=False,
        disable=(not accelerator.is_main_process) or inside_slurm(),
    )

    with torch.no_grad():
        for raw_batch in eval_iter:
            batch_indices = raw_batch["index"]
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            batch_indices = batch_indices.to(device=device, dtype=torch.long, non_blocking=True)

            processed_batch = preprocessor(raw_batch)
            with accelerator.autocast():
                predicted_value = accelerator.unwrap_model(value_policy).predict_value(processed_batch)

            gathered_idx = accelerator.gather_for_metrics(batch_indices)
            gathered_val = accelerator.gather_for_metrics(predicted_value)

            if accelerator.is_main_process:
                idx_np = gathered_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                val_np = gathered_val.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

                if cfg.acp.streaming_write:
                    # Streaming: accumulate in dict
                    for idx, val in zip(idx_np, val_np):
                        prediction_dict[int(idx)] = float(val)

                    # Periodically flush dict to temp storage to prevent WSS accumulation
                    if len(prediction_dict) >= chunk_size:
                        import psutil
                        process = psutil.Process()
                        mem_before = process.memory_info().rss / 1024 / 1024  # MB

                        prediction_chunks.append(prediction_dict.copy())
                        prediction_dict.clear()
                        logging.info(f"Flushed prediction chunk {len(prediction_chunks)}, dict cleared")
                        import gc
                        gc.collect()

                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        logging.info(f"Memory: before={mem_before:.1f}MB, after={mem_after:.1f}MB, freed={mem_before - mem_after:.1f}MB")
                else:
                    # Full-memory: accumulate in lookup array
                    prediction_lookup[idx_np] = val_np
                    prediction_seen[idx_np] = True

                # Explicitly delete tensors to free memory
                del gathered_idx, gathered_val, idx_np, val_np

            # Clear CUDA cache periodically to prevent accumulation
            if accelerator.is_main_process and len(eval_iter) > 100:
                batch_num = getattr(eval_iter, 'n', 0)
                if batch_num % 100 == 0:
                    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if cfg.acp.streaming_write:
            # Merge all chunks back into dict
            if prediction_chunks:
                logging.info(f"Merging {len(prediction_chunks)} chunks back into dict...")
                for chunk in prediction_chunks:
                    prediction_dict.update(chunk)
                prediction_chunks.clear()
                del prediction_chunks

            # Convert dict to array in dataset order
            predicted_values = np.array([prediction_dict[int(idx)] for idx in absolute_indices], dtype=np.float32)
            if len(predicted_values) != len(absolute_indices):
                raise RuntimeError(f"Inference missing predictions for some frames.")

            # Clear dict to free memory
            prediction_dict.clear()
            del prediction_dict
            import gc
            gc.collect()
        else:
            # Full-memory mode: extract from lookup
            if prediction_lookup is None or prediction_seen is None:
                raise RuntimeError("Prediction buffers unexpectedly missing on main process.")

            missing_mask = ~prediction_seen[absolute_indices]
            if bool(np.any(missing_mask)):
                missing_count = int(np.sum(missing_mask))
                raise RuntimeError(f"Inference is missing predictions for {missing_count} frames.")

            predicted_values = prediction_lookup[absolute_indices]
        logging.info(
            "Predicted value stats | min=%.6f max=%.6f mean=%.6f std=%.6f",
            float(np.min(predicted_values)),
            float(np.max(predicted_values)),
            float(np.mean(predicted_values)),
            float(np.std(predicted_values)),
        )

        columns: dict[str, np.ndarray] = {
            cfg.acp.value_field: predicted_values.astype(np.float32),
        }
        feature_infos: dict[str, dict[str, Any]] = {
            cfg.acp.value_field: {"dtype": "float32", "shape": (1,), "names": None},
        }

        indicator_positive_ratio: float | None = None
        thresholds: dict[int, float] | None = None

        if cfg.acp.enable:
            # Build episode info (needed for both streaming and full-memory modes)
            episode_info, task_max_lengths = _build_episode_info(
                dataset=dataset,
                success_field=cfg.dataset.success_field,
                default_success=cfg.dataset.default_success,
            )

            # Use streaming write (chunked processing) or full-memory mode
            if cfg.acp.streaming_write and cfg.acp.write_mode == "sidecar":
                logging.info("Using streaming write mode (chunked processing)")

                # Derive sidecar subdir from value_field if not explicitly set
                sidecar_subdir = cfg.acp.sidecar_subdir
                if sidecar_subdir is None:
                    prefix = "complementary_info.value_"
                    if cfg.acp.value_field.startswith(prefix):
                        sidecar_subdir = cfg.acp.value_field[len(prefix) :]
                    else:
                        sidecar_subdir = "default"

                # Streaming write: processes in episode chunks, reduces memory peak
                thresholds = _write_columns_sidecar_streaming(
                    dataset_root=Path(dataset.root),
                    sidecar_subdir=sidecar_subdir,
                    absolute_indices=absolute_indices,
                    episode_indices=episode_indices,
                    frame_indices=frame_indices,
                    task_indices=task_indices,
                    predicted_values=predicted_values,
                    interventions=interventions,
                    expert_episode_mask=expert_episode_mask,
                    episode_info=episode_info,
                    task_max_lengths=task_max_lengths,
                    value_cfg=value_cfg,
                    cfg=cfg,
                )

                # For viz and stats, read back from sidecar
                sidecar_path = Path(dataset.root) / "advantage" / sidecar_subdir / "frames.parquet"
                sidecar_table = pq.read_table(sidecar_path)

                # Build lookup to extract values in dataset order
                sidecar_indices = sidecar_table["index"].to_numpy()
                max_idx = int(np.max(absolute_indices))

                value_lookup = np.full(max_idx + 1, np.nan, dtype=np.float32)
                advantage_lookup = np.full(max_idx + 1, np.nan, dtype=np.float32)
                indicator_lookup = np.zeros(max_idx + 1, dtype=np.int64)

                value_lookup[sidecar_indices] = sidecar_table[cfg.acp.value_field].to_numpy().astype(np.float32)
                advantage_lookup[sidecar_indices] = sidecar_table[cfg.acp.advantage_field].to_numpy().astype(np.float32)
                indicator_lookup[sidecar_indices] = sidecar_table[cfg.acp.indicator_field].to_numpy().astype(np.int64)

                predicted_values = value_lookup[absolute_indices]
                advantages = advantage_lookup[absolute_indices]
                indicators = indicator_lookup[absolute_indices]

                indicator_positive_ratio = float(np.mean(indicators.astype(np.float32)))

                columns[cfg.acp.value_field] = predicted_values.astype(np.float32)
                columns[cfg.acp.advantage_field] = advantages.astype(np.float32)
                columns[cfg.acp.indicator_field] = indicators.astype(np.int64)

                logging.info(
                    "ACP stats (streaming) | n_step=%d positive_ratio_target=%.4f positive_ratio_observed=%.4f",
                    cfg.acp.n_step,
                    cfg.acp.positive_ratio,
                    indicator_positive_ratio,
                )
            else:
                # Original full-memory mode
                logging.info("Using full-memory mode (compute all at once)")

                value_targets = _compute_value_targets(
                    value_cfg=value_cfg,
                    episode_indices=episode_indices,
                    frame_indices=frame_indices,
                    episode_info=episode_info,
                    task_max_lengths=task_max_lengths,
                    c_fail_coef=cfg.acp.c_fail_coef,
                )
                rewards = _compute_rewards(
                    value_cfg=value_cfg,
                    targets=value_targets,
                    episode_indices=episode_indices,
                    frame_indices=frame_indices,
                    episode_info=episode_info,
                )
                advantages = _compute_advantages(
                    value_cfg=value_cfg,
                    rewards=rewards,
                    values=predicted_values,
                    episode_indices=episode_indices,
                    frame_indices=frame_indices,
                    n_step=cfg.acp.n_step,
                    episode_info=episode_info,
                )
                thresholds = _compute_task_thresholds(
                    task_indices=task_indices,
                    advantages=advantages,
                    positive_ratio=cfg.acp.positive_ratio,
                )
                indicators = _binarize_advantages(
                    task_indices=task_indices,
                    advantages=advantages,
                    thresholds=thresholds,
                    interventions=interventions,
                    force_intervention_positive=cfg.acp.force_intervention_positive,
                    expert_episode_mask=expert_episode_mask,
                    force_expert_episode_positive=cfg.acp.force_expert_episode_positive,
                )

                indicator_positive_ratio = float(np.mean(indicators.astype(np.float32)))
                logging.info(
                    "ACP stats | n_step=%d positive_ratio_target=%.4f positive_ratio_observed=%.4f",
                    cfg.acp.n_step,
                    cfg.acp.positive_ratio,
                    indicator_positive_ratio,
                )
                logging.info(
                    "ACP overrides | intervention_positive=%s intervention_frames=%d expert_episode_positive=%s "
                    "expert_episode_field=%s expert_episodes=%d expert_episode_frames=%d",
                    cfg.acp.force_intervention_positive,
                    int(np.sum(interventions.astype(np.float32) > 0.5)),
                    cfg.acp.force_expert_episode_positive,
                    cfg.acp.expert_episode_field,
                    int(np.unique(episode_indices[expert_episode_mask]).shape[0])
                    if bool(np.any(expert_episode_mask))
                    else 0,
                    int(np.sum(expert_episode_mask)),
                )

                columns[cfg.acp.advantage_field] = advantages.astype(np.float32)
                columns[cfg.acp.indicator_field] = indicators.astype(np.int64)
                feature_infos[cfg.acp.advantage_field] = {"dtype": "float32", "shape": (1,), "names": None}
                feature_infos[cfg.acp.indicator_field] = {"dtype": "int64", "shape": (1,), "names": None}

        # Write columns: sidecar mode or in-place mode (only if not already written by streaming)
        if not (cfg.acp.enable and cfg.acp.streaming_write and cfg.acp.write_mode == "sidecar"):
            if cfg.acp.write_mode == "sidecar":
                # Derive sidecar subdir from value_field if not explicitly set
                sidecar_subdir = cfg.acp.sidecar_subdir
                if sidecar_subdir is None:
                    # Extract tag from value_field: "complementary_info.value_<tag>" -> "<tag>"
                    prefix = "complementary_info.value_"
                    if cfg.acp.value_field.startswith(prefix):
                        sidecar_subdir = cfg.acp.value_field[len(prefix) :]
                    else:
                        sidecar_subdir = "default"
                _write_columns_sidecar(
                    dataset_root=Path(dataset.root),
                    sidecar_subdir=sidecar_subdir,
                    absolute_indices=absolute_indices,
                    columns=columns,
                    feature_infos=feature_infos,
                )
            elif cfg.acp.write_mode == "in_place":
                _write_columns_in_place(
                    dataset_root=Path(dataset.root),
                    absolute_indices=absolute_indices,
                    columns=columns,
                    feature_infos=feature_infos,
                )
                logging.info("Wrote value annotations to dataset root (in_place): %s", dataset.root)
            else:
                raise ValueError(f"Invalid write_mode '{cfg.acp.write_mode}'")

        # Sync computed columns into the in-memory hf_dataset so viz can read them
        for field, values in columns.items():
            if field in dataset.hf_dataset.column_names:
                dataset.hf_dataset = dataset.hf_dataset.remove_columns([field])
            dataset.hf_dataset = dataset.hf_dataset.add_column(field, values.tolist())

        viz_outputs: list[str] = []
        if cfg.viz.enable:
            viz_outputs = _export_visualization_outputs(dataset=dataset, cfg=cfg, output_dir=output_dir)

        result = {
            "main_process": True,
            "world_size": int(accelerator.num_processes),
            "num_frames": int(frame_count),
            "checkpoint": str(pretrained_dir),
            "value_field": cfg.acp.value_field,
            "acp_enabled": bool(cfg.acp.enable),
            "value_inference_skipped": False,
            "indicator_positive_ratio": indicator_positive_ratio,
            "thresholds": thresholds,
            "viz_outputs": viz_outputs,
        }
    else:
        result = {
            "main_process": False,
            "world_size": int(accelerator.num_processes),
        }

    # Keep non-main ranks alive until main rank finishes optional slow visualization export.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result


@parser.wrap()
def value_infer(cfg: ValueInferencePipelineConfig):
    return run_value_inference_pipeline(cfg)


def main():
    register_third_party_plugins()
    value_infer()


if __name__ == "__main__":
    main()
