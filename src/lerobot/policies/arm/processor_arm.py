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

"""ARM Processor: encodes images/text with CLIP and generates per-frame
tri-state advantage labels and binary task-completion labels."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import CLIPModel, CLIPProcessor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.arm.arm_utils import (
    compute_advantage_labels,
    compute_causal_indices,
    compute_completion_labels,
    pad_state_to_max_dim,
)
from lerobot.policies.arm.configuration_arm import ARMConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    from_tensor_to_numpy,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

TRI_STATE_KEY = "tri_state"


def arm_batch_to_transition(batch: dict) -> EnvTransition:
    """`batch_to_transition` that also preserves the windowed ``tri_state`` column.

    The standard converter only keeps `observation.*` keys plus a fixed
    complementary-data whitelist, dropping everything else. ARM's per-frame
    advantage labels arrive under the top-level ``tri_state`` column, so we
    stash them in COMPLEMENTARY_DATA for the encoding step to consume.
    """
    transition = batch_to_transition(batch)
    if TRI_STATE_KEY in batch:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp[TRI_STATE_KEY] = batch[TRI_STATE_KEY]
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
    return transition


class ARMEncodingProcessorStep(ProcessorStep):
    """Encodes images and text with CLIP and generates ARM training targets.

    For each frame in the causal window a linear progress value is computed
    (frame_position / episode_length), from which targets are derived:
      - advantage_targets: (B, T-1) tri-state {0=neg, 1=neutral, 2=pos} per
        transition. Overridden by the dataset's per-frame ``tri_state`` column
        when present (mapped {-1,0,+1} -> {0,1,2}).
      - completion_targets: (B,) binary {0, 1} for the current (rightmost)
        frame only, from a progress threshold.
    """

    def __init__(
        self,
        config: ARMConfig,
        image_key: str | None = None,
        dataset_meta=None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats

        self.device = torch.device(
            self.config.device if self.config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()

    def _find_episode_for_frame(self, frame_idx: int) -> int:
        """Find the episode index for a given frame index."""
        for ep_idx in range(len(self.dataset_meta.episodes)):
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            if ep_start <= frame_idx < ep_end:
                return ep_idx
        return 0

    def _get_episode_indices(self, frame_indices: np.ndarray, episode_index) -> np.ndarray:
        """Get episode indices for each frame index."""
        if episode_index is None:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        episode_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(episode_index)))

        if len(episode_indices) == 1 and len(frame_indices) > 1:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        return episode_indices

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Encode images/text with CLIP and attach ARM targets to the observation."""
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)

        image = observation.get(self.image_key)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # (T, C, H, W) -> (1, T, C, H, W); (C, H, W) -> (1, 1, C, H, W)
        if image.ndim == 4:
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]

        batch_size = image.shape[0]
        total_frames = image.shape[1]  # window_size

        # Causal window: all frames valid (boundary frames are clamped/duplicated).
        lengths = torch.full((batch_size,), total_frames, dtype=torch.int32)

        # Encode images with CLIP
        observation["video_features"] = self._encode_images_batch(image)

        # State
        state_key = self.config.state_key
        state_data = observation.get(state_key)
        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)  # (T, D) -> (1, T, D)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (D,) -> (1, 1, D)

        observation["state_features"] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        # Text
        task = comp_data.get("task")
        if isinstance(task, list):
            task = task[0] if task else ""
        observation["text_features"] = self._encode_text_clip(task, batch_size)

        observation["lengths"] = lengths

        # Targets
        # Priority 1: use per-frame tri-state labels from the dataset when available.
        # The dataloader windows them to (T,) per sample; here batch_size=1 so shape
        # is (T,) or (B, T). Values are {-1, 0, +1} → map to {0, 1, 2} via label+1.
        # advantage_targets shape must be (B, T-1): discard index-0 (no prior frame).
        tri_state_raw = comp_data.get(TRI_STATE_KEY)
        if tri_state_raw is not None:
            if not isinstance(tri_state_raw, torch.Tensor):
                tri_state_raw = torch.tensor(tri_state_raw, dtype=torch.long)
            else:
                tri_state_raw = tri_state_raw.long()
            if tri_state_raw.dim() == 1:
                tri_state_raw = tri_state_raw.unsqueeze(0)  # (T,) -> (1, T)
            # Map {-1,0,+1} -> {0,1,2} and drop first frame (no preceding transition)
            observation["advantage_targets"] = (tri_state_raw + 1)[:, 1:]  # (B, T-1)
            # completion_targets still come from the linear-progress heuristic until
            # a dedicated label column is added.
            if self.dataset_meta is not None:
                _, completion_targets = self._compute_batch_targets(
                    frame_indices, episode_indices, total_frames
                )
                observation["completion_targets"] = completion_targets
        elif self.dataset_meta is not None:
            # Fallback: derive both targets from linear progress (placeholder).
            advantage_targets, completion_targets = self._compute_batch_targets(
                frame_indices, episode_indices, total_frames
            )
            observation["advantage_targets"] = advantage_targets
            observation["completion_targets"] = completion_targets

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def _compute_batch_targets(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        total_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frame advantage + completion targets for a batch.

        Progress is linear within the episode: rel_frame / (ep_length - 1).
        """
        batch_size = len(frame_indices)
        window_size = self.config.window_size
        frame_gap = self.config.frame_gap

        progress = torch.zeros(batch_size, total_frames, dtype=torch.float32)

        for b_idx in range(batch_size):
            ep_idx = int(episode_indices[b_idx])
            frame_idx = int(frame_indices[b_idx])

            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            ep_length = ep_end - ep_start

            causal_indices, _ = compute_causal_indices(
                frame_idx, ep_start, ep_end, window_size, frame_gap=frame_gap
            )

            for t_idx, abs_idx in enumerate(causal_indices.tolist()):
                rel_frame = abs_idx - ep_start
                progress[b_idx, t_idx] = float(np.clip(rel_frame / max(ep_length - 1, 1), 0.0, 1.0))

        advantage_targets = compute_advantage_labels(progress, self.config.progress_delta_threshold)
        # Completion is a single label for the current (rightmost) frame only.
        completion_per_frame = compute_completion_labels(progress, 1.0 - self.config.completion_threshold)
        completion_targets = completion_per_frame[:, -1]  # (B,)
        return advantage_targets, completion_targets

    @property
    def training(self) -> bool:
        return getattr(self, "_training_mode", True)

    def train(self, mode: bool = True):
        self._training_mode = mode
        return self

    def eval(self):
        return self.train(False)

    @torch.no_grad()
    def _encode_images_batch(self, images: np.ndarray) -> torch.Tensor:
        """Encode a batch of images using CLIP.

        Args:
            images: (B, T, C, H, W)

        Returns:
            (B, T, 512) image embeddings
        """
        batch_size, seq_length = images.shape[0], images.shape[1]
        images = images.reshape(batch_size * seq_length, *images.shape[2:])
        images = self._maybe_downsample_images(images)

        num_frames = images.shape[0]
        images_list = []
        for i in range(num_frames):
            img = images[i]
            if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            images_list.append(img)

        all_embeddings = []
        for i in range(0, num_frames, self.config.clip_batch_size):
            batch_imgs = images_list[i : i + self.config.clip_batch_size]
            inputs = self.clip_processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.clip_model.get_image_features(**inputs).detach().cpu()
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings)  # (B*T, 512)
        return all_embeddings.reshape(batch_size, seq_length, -1)  # (B, T, 512)

    def _maybe_downsample_images(self, images: np.ndarray) -> np.ndarray:
        """Resize frames before CLIP preprocessing when configured."""
        target_size = self.config.image_downsample_size
        if target_size is None:
            return images

        target_height, target_width = target_size
        current_height, current_width = images.shape[-2:]
        if (current_height, current_width) == (target_height, target_width):
            return images

        image_tensor = torch.from_numpy(images).float()
        align_corners = False if self.config.image_downsample_mode in {"bilinear", "bicubic"} else None
        antialias = (
            self.config.image_downsample_antialias
            if self.config.image_downsample_mode in {"bilinear", "bicubic"}
            else False
        )
        resized = F.interpolate(
            image_tensor,
            size=(target_height, target_width),
            mode=self.config.image_downsample_mode,
            align_corners=align_corners,
            antialias=antialias,
        )
        return resized.numpy()

    @torch.no_grad()
    def _encode_text_clip(self, text: str, batch_size: int) -> torch.Tensor:
        """Encode task text using the CLIP text encoder. Returns (B, 512)."""
        inputs = self.clip_processor.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_embedding = self.clip_model.get_text_features(**inputs).detach().cpu()
        return text_embedding.expand(batch_size, -1)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add encoded features to the observation features."""
        features[PipelineFeatureType.OBSERVATION]["video_features"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.config.num_frames, self.config.image_dim)
        )
        features[PipelineFeatureType.OBSERVATION]["text_features"] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(self.config.text_dim,)
        )
        features[PipelineFeatureType.OBSERVATION]["state_features"] = PolicyFeature(
            type=FeatureType.STATE, shape=(self.config.num_frames, self.config.max_state_dim)
        )
        return features


def make_arm_pre_post_processors(
    config: ARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create pre-processor and post-processor pipelines for ARM."""
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            to_transition=arm_batch_to_transition,
            steps=[
                AddBatchDimensionProcessorStep(),
                RenameObservationsProcessorStep(rename_map={}),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                ARMEncodingProcessorStep(
                    config=config, dataset_meta=dataset_meta, dataset_stats=dataset_stats
                ),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
