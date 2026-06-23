#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""ARM: Advantage Reward Modeling for Long-Horizon Manipulation — Configuration."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("arm")
@dataclass
class ARMConfig(PreTrainedConfig):
    """Configuration for ARM (Advantage Reward Modeling).

    Architecture: single Transformer encoder with additive input fusion and two
    jointly-trained output heads (multi-frame advantage + task completion).

    The model uses a causal window of `window_size` frames sampled at 1Hz
    (frame_gap=30 at 30fps). Each frame's representation is formed by additive
    fusion of visual, state, and language embeddings projected to d_model.
    """

    # Window and sampling
    window_size: int = 5
    frame_gap: int = 30

    # Architecture
    image_dim: int = 512
    text_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    max_state_dim: int = 32
    dropout: float = 0.1
    batch_size: int = 64
    clip_batch_size: int = 64

    # Advantage head
    num_advantage_classes: int = 3

    # Loss weights
    lambda_int: float = 1.0
    lambda_succ: float = 1.0

    # Focal loss params (task completion head). Per ARM paper Table 6: alpha=2.0,
    # gamma=2.0 is a flat scaling factor in -alpha*(1-p)^gamma*log(p).
    focal_alpha: float = 2.0
    focal_gamma: float = 2.0

    # Target generation
    # completion_threshold: paper uses ε=1e-3 (only true terminal frames count as positive)
    completion_threshold: float = 0.001
    progress_delta_threshold: float = 0.01

    # Data
    pretrained_model_path: str | None = None
    device: str | None = None
    image_key: str = OBS_IMAGES + ".top"
    # Per-frame tri-state advantage labels from dataset ({-1,0,+1} → mapped to {0,1,2} internally)
    tri_state_key: str = "tri_state"
    image_downsample_size: tuple[int, int] | None = None
    image_downsample_mode: str = "bilinear"
    image_downsample_antialias: bool = True
    state_key: str = OBS_STATE

    input_features: dict = field(default_factory=lambda: {})
    output_features: dict = field(default_factory=lambda: {})

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {self.window_size}")
        if self.num_advantage_classes < 2:
            raise ValueError(f"num_advantage_classes must be at least 2, got {self.num_advantage_classes}")

        if self.image_downsample_size is not None:
            if len(self.image_downsample_size) != 2:
                raise ValueError("image_downsample_size must be a tuple of (height, width)")
            height, width = self.image_downsample_size
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"image_downsample_size must contain positive integers, got {self.image_downsample_size}"
                )

        valid_resize_modes = {"nearest", "nearest-exact", "bilinear", "bicubic", "area"}
        if self.image_downsample_mode not in valid_resize_modes:
            raise ValueError(
                f"image_downsample_mode must be one of {sorted(valid_resize_modes)}, "
                f"got {self.image_downsample_mode}"
            )

        self.input_features = {}
        self.output_features = {}

        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(shape=(480, 640, 3), type=FeatureType.VISUAL)

        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),
            type=FeatureType.STATE,
        )

        # Advantage is per-transition (window_size - 1); completion is a single
        # prediction for the current (rightmost) frame t, per the ARM paper.
        self.output_features["advantage"] = PolicyFeature(
            shape=(self.window_size - 1, self.num_advantage_classes), type=FeatureType.REWARD
        )
        self.output_features["completion"] = PolicyFeature(
            shape=(1,), type=FeatureType.REWARD
        )

        # n_obs_steps is an inherited dataclass field; derive it from window_size.
        self.n_obs_steps = self.window_size - 1

    @property
    def num_frames(self) -> int:
        return self.window_size

    @property
    def max_length(self) -> int:
        return self.window_size

    @property
    def observation_delta_indices(self) -> list[int]:
        """Causal frame indices: [-(W-1)*gap, ..., -gap, 0]."""
        return [-self.frame_gap * i for i in range(self.window_size - 1, 0, -1)] + [0]

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def extra_delta_timestamps_keys(self) -> dict[str, list[int]]:
        """Non-standard dataset columns to window alongside observations.

        The per-frame ``tri_state`` advantage labels are windowed with the same
        causal indices as observations so the dataloader returns one label per
        frame in the window. The processor then derives (W-1) transition labels.
        """
        return {self.tri_state_key: self.observation_delta_indices}

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=5e-5,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=5e-5,
            decay_lr=5e-6,
            num_warmup_steps=1000,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        pass
