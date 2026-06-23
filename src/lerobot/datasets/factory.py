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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _get_visual_normalization_mode(cfg: PreTrainedConfig) -> NormalizationMode | None:
    norm_map = getattr(cfg, "normalization_mapping", None)
    if norm_map is None:
        return None

    return norm_map.get("VISUAL") or norm_map.get(FeatureType.VISUAL)


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    available_observation_keys = {key for key in ds_meta.features if key.startswith(OBS_PREFIX)}
    available_visual_keys = {key for key in available_observation_keys if key.startswith(f"{OBS_IMAGES}.")}
    available_non_visual_keys = available_observation_keys - available_visual_keys

    declared_observation_keys = {key for key in (cfg.input_features or {}) if key.startswith(OBS_PREFIX)}
    declared_visual_keys = {key for key in declared_observation_keys if key.startswith(f"{OBS_IMAGES}.")}
    declared_non_visual_keys = declared_observation_keys - declared_visual_keys

    if declared_visual_keys:
        matched_visual_keys = declared_visual_keys & available_visual_keys
        selected_visual_keys = matched_visual_keys or available_visual_keys
    else:
        selected_visual_keys = available_visual_keys

    if declared_non_visual_keys:
        matched_non_visual_keys = declared_non_visual_keys & available_non_visual_keys
        selected_non_visual_keys = matched_non_visual_keys or available_non_visual_keys
    else:
        selected_non_visual_keys = available_non_visual_keys

    observation_keys = selected_visual_keys | selected_non_visual_keys

    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key in observation_keys and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    # Generic hook: a policy config may declare additional non-standard columns
    # to window (e.g. ARM's per-frame `tri_state` labels). `extra_delta_timestamps_keys`
    # maps a dataset column name to a list of integer delta indices.
    extra_keys = getattr(cfg, "extra_delta_timestamps_keys", None)
    if extra_keys:
        for key, indices in extra_keys.items():
            if key in ds_meta.features and indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # Use cfg.value for value training, cfg.policy for policy training
        model_cfg = getattr(cfg, 'value', None) or cfg.policy
        delta_timestamps = resolve_delta_timestamps(model_cfg, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                image_center_crop=cfg.dataset.image_center_crop,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                image_center_crop=cfg.dataset.image_center_crop,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    # Use cfg.value for value training, cfg.policy for policy training
    model_cfg = getattr(cfg, 'value', None) or cfg.policy
    visual_norm_mode = _get_visual_normalization_mode(model_cfg)
    if cfg.dataset.use_imagenet_stats and visual_norm_mode != NormalizationMode.IDENTITY:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
