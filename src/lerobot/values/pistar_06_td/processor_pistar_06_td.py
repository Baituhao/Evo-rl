#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.values.pistar_06_td.configuration_pistar_06_td import Pistar_06_tdConfig

PISTAR_06_TD_IMAGES_KEY = "observation.pistar_06_td.images"
PISTAR_06_TD_IMAGE_MASK_KEY = "observation.pistar_06_td.image_attention_mask"


def _pad_last_dim(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return functional.pad(vector, (0, new_dim - vector.shape[-1]))


@ProcessorStepRegistry.register(name="pistar_06_td_prepare_task_prompt")
@dataclass
class Pistar_06_tdPrepareTaskPromptProcessorStep(ProcessorStep):
    task_key: str = "task"
    include_state_in_prompt: bool = True
    state_feature: str = OBS_STATE
    max_state_dim: int = 32
    state_discretization_bins: int = 256

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "include_state_in_prompt": self.include_state_in_prompt,
            "state_feature": self.state_feature,
            "max_state_dim": self.max_state_dim,
            "state_discretization_bins": self.state_discretization_bins,
        }

    @staticmethod
    def _clean_prompt(task: str) -> str:
        return str(task).strip().replace("_", " ").replace("\n", " ").strip()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.task_key not in complementary_data:
            raise KeyError(f"Missing task field '{self.task_key}' in complementary data.")
        tasks_raw = complementary_data[self.task_key]
        if isinstance(tasks_raw, str):
            tasks = [tasks_raw]
        elif isinstance(tasks_raw, Sequence) and all(isinstance(task, str) for task in tasks_raw):
            tasks = list(tasks_raw)
        else:
            raise TypeError(
                f"Expected task field '{self.task_key}' as sequence of strings, got {type(tasks_raw)}."
            )

        prompts: list[str] = []
        if self.include_state_in_prompt:
            if self.state_feature not in observation:
                raise KeyError(
                    f"Missing state feature '{self.state_feature}' while include_state_in_prompt=True."
                )
            state = observation[self.state_feature]
            if not isinstance(state, Tensor):
                state = torch.as_tensor(state)

            if state.ndim == 1:
                state = state.unsqueeze(0)

            # Support dual-frame state [B, N_delta, D] from delta_indices (for online TD bootstrap).
            # Single-frame [B, D] falls back to standard processing.
            if state.ndim == 3:
                # Dual-frame: construct prompts for t=0 (current) and t=1 (next) separately,
                # then concatenate into a 2B batch so downstream tokenizer processes them as one.
                # Model's forward will split them back into current/next for V(s) and V(s').
                B, N_delta, D = state.shape
                if N_delta < 2:
                    raise ValueError(
                        f"Dual-frame state expected N_delta>=2, got {N_delta}. "
                        "Ensure observation_delta_indices=[0,1] is set in config."
                    )

                state_current = state[:, 0, :]  # [B, D]
                state_next = state[:, 1, :]  # [B, D]

                # Discretize both frames
                def discretize(s: Tensor) -> np.ndarray:
                    s = s.detach().to(dtype=torch.float32, device="cpu")
                    s = _pad_last_dim(s, self.max_state_dim)
                    s_np = s.numpy()
                    bins = np.linspace(-1.0, 1.0, self.state_discretization_bins + 1, dtype=np.float32)[:-1]
                    return np.digitize(s_np, bins=bins) - 1

                disc_current = discretize(state_current)  # [B, D_padded]
                disc_next = discretize(state_next)  # [B, D_padded]

                if len(tasks) != B:
                    raise ValueError(f"Task count ({len(tasks)}) does not match state batch size ({B}).")

                # Build 2B prompts: [current_0, ..., current_B-1, next_0, ..., next_B-1]
                prompts_current = []
                prompts_next = []
                for i, task in enumerate(tasks):
                    cleaned_task = self._clean_prompt(task)
                    state_str_cur = " ".join(map(str, disc_current[i].tolist()))
                    state_str_nxt = " ".join(map(str, disc_next[i].tolist()))
                    prompts_current.append(f"Task: {cleaned_task}, State: {state_str_cur}\nValue: ")
                    prompts_next.append(f"Task: {cleaned_task}, State: {state_str_nxt}\nValue: ")
                prompts = prompts_current + prompts_next

            elif state.ndim == 2:
                # Single-frame: standard processing [B, D]
                state = state.detach().to(dtype=torch.float32, device="cpu")
                state = _pad_last_dim(state, self.max_state_dim)
                state_np = state.numpy()
                bins = np.linspace(-1.0, 1.0, self.state_discretization_bins + 1, dtype=np.float32)[:-1]
                discretized_state = np.digitize(state_np, bins=bins) - 1

                if discretized_state.shape[0] != len(tasks):
                    raise ValueError(
                        f"Task count ({len(tasks)}) does not match state batch size ({discretized_state.shape[0]})."
                    )

                for i, task in enumerate(tasks):
                    cleaned_task = self._clean_prompt(task)
                    state_str = " ".join(map(str, discretized_state[i].tolist()))
                    prompts.append(f"Task: {cleaned_task}, State: {state_str}\nValue: ")
            else:
                raise ValueError(
                    f"Expected state tensor with shape [B, D] or [B, N_delta, D], got {tuple(state.shape)} "
                    f"for feature '{self.state_feature}'."
                )
        else:
            prompts = [f"Task: {self._clean_prompt(task)}\nValue: " for task in tasks]

        complementary_data[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pistar_06_td_prepare_images")
@dataclass
class Pistar_06_tdPrepareImagesProcessorStep(ProcessorStep):
    camera_features: list[str]

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_features": self.camera_features,
        }

    @staticmethod
    def _to_bchw(img_batch: Tensor) -> Tensor:
        """Convert to [B, C, H, W] or [B, N_delta, C, H, W] if temporal dimension present."""
        if img_batch.ndim == 4:
            # Standard case: [B, C, H, W] or [B, H, W, C]
            if img_batch.shape[1] in {1, 3}:  # [B,C,H,W]
                return img_batch
            if img_batch.shape[-1] in {1, 3}:  # [B,H,W,C]
                return img_batch.permute(0, 3, 1, 2)
            raise ValueError(
                "Camera tensor must be channels-first or channels-last. "
                f"Got camera batch with shape={tuple(img_batch.shape)}."
            )
        elif img_batch.ndim == 5:
            # Delta indices case: [B, N_delta, C, H, W] or [B, N_delta, H, W, C]
            if img_batch.shape[2] in {1, 3}:  # [B, N_delta, C, H, W]
                return img_batch
            if img_batch.shape[-1] in {1, 3}:  # [B, N_delta, H, W, C]
                return img_batch.permute(0, 1, 4, 2, 3)
            raise ValueError(
                "Camera tensor with temporal dim must be channels-first or channels-last. "
                f"Got shape={tuple(img_batch.shape)}."
            )
        raise ValueError(f"Expected image batch rank 4 or 5, got shape {tuple(img_batch.shape)}.")

    def _process_camera_batch(self, img_batch: Tensor) -> Tensor:
        return self._to_bchw(img_batch).detach().to(dtype=torch.float32)

    @staticmethod
    def _resize_spatial(img_batch: Tensor, target_hw: tuple[int, int]) -> Tensor:
        """Resize spatial dimensions. Handles both [B,C,H,W] and [B,N_delta,C,H,W]."""
        if img_batch.shape[-2:] == target_hw:
            return img_batch

        if img_batch.ndim == 4:
            # Standard case: [B, C, H, W]
            return functional.interpolate(
                img_batch,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        elif img_batch.ndim == 5:
            # Delta indices case: [B, N_delta, C, H, W]
            # Flatten B and N_delta, resize, then unflatten
            B, N_delta, C, H, W = img_batch.shape
            flat = img_batch.view(B * N_delta, C, H, W)
            resized = functional.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
            return resized.view(B, N_delta, C, target_hw[0], target_hw[1])
        raise ValueError(f"Unsupported tensor rank {img_batch.ndim} for spatial resize.")

    def _prepare_images(self, observation: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Prepare camera images, unifying all to dual-frame [B, 2, N_cam, C, H, W] for TD bootstrap.

        When delta_indices is enabled, some cameras may have temporal dimension [B, 2, C, H, W]
        while others remain single-frame [B, C, H, W] (if not in input_features). This method
        unifies all cameras to dual-frame by repeating single-frame cameras along dim=1.
        """
        present_img_keys = [key for key in self.camera_features if key in observation]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(observation.keys())}"
            )

        reference_img = self._process_camera_batch(torch.as_tensor(observation[present_img_keys[0]]))
        bsize = reference_img.shape[0]

        # Detect temporal dimension: [B, N_delta, C, H, W] vs [B, C, H, W]
        has_temporal_dim = reference_img.ndim == 5
        if has_temporal_dim:
            n_delta = reference_img.shape[1]
            reference_shape = reference_img.shape[2:]  # (C, H, W)
            reference_hw = reference_img.shape[-2:]
        else:
            n_delta = 1  # Will broadcast single-frame cameras to dual-frame
            reference_shape = reference_img.shape[1:]  # (C, H, W)
            reference_hw = reference_img.shape[-2:]

        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in observation:
                img = self._process_camera_batch(torch.as_tensor(observation[key]))
                if img.shape[0] != bsize:
                    raise ValueError(
                        f"Mismatched batch size across cameras. Camera '{key}' has {img.shape[0]}, expected {bsize}."
                    )

                # Unify to dual-frame: if img is [B, C, H, W], repeat to [B, 2, C, H, W]
                if img.ndim == 4 and has_temporal_dim:
                    # Single-frame camera, but others have temporal → broadcast by repeating
                    img = img.unsqueeze(1).repeat(1, n_delta, 1, 1, 1)  # [B, N_delta, C, H, W]
                elif img.ndim == 5:
                    # Already has temporal dimension
                    if img.shape[1] != n_delta:
                        raise ValueError(f"Camera '{key}' has N_delta={img.shape[1]}, expected {n_delta}.")
                elif img.ndim == 4 and not has_temporal_dim:
                    # All cameras are single-frame → no unification needed yet
                    pass
                else:
                    raise ValueError(f"Unexpected camera tensor rank {img.ndim} for key '{key}'.")

                # Check channel consistency
                channel_dim = 2 if img.ndim == 5 else 1
                if img.shape[channel_dim] != reference_shape[0]:
                    raise ValueError(
                        f"Camera '{key}' has {img.shape[channel_dim]} channels, expected {reference_shape[0]}."
                    )

                # Resize spatial dimensions
                img = self._resize_spatial(img, reference_hw)
                image_tensors.append(img)
                image_masks.append(torch.ones(bsize, dtype=torch.bool))
            else:
                # Missing camera → pad with zeros matching reference shape
                if has_temporal_dim or any(t.ndim == 5 for t in image_tensors):
                    # Use dual-frame shape
                    pad_shape = (bsize, n_delta if n_delta > 1 else 2, *reference_shape)
                else:
                    pad_shape = (bsize, *reference_shape)
                image_tensors.append(torch.zeros(pad_shape))
                image_masks.append(torch.zeros(bsize, dtype=torch.bool))

        # Final unification pass: if any camera is 5D but some are 4D, broadcast all 4D to 5D
        final_ndim = max(t.ndim for t in image_tensors)
        if final_ndim == 5:
            unified_tensors = []
            for img in image_tensors:
                if img.ndim == 4:
                    # Broadcast [B, C, H, W] → [B, 2, C, H, W]
                    img = img.unsqueeze(1).repeat(1, 2, 1, 1, 1)
                unified_tensors.append(img)
            image_tensors = unified_tensors

        # Stack: [B, N_cam, N_delta, C, H, W] or [B, N_cam, C, H, W]
        if image_tensors[0].ndim == 5:
            images = torch.stack(image_tensors, dim=1)  # [B, N_cam, N_delta, C, H, W]
            images = images.transpose(1, 2)  # [B, N_delta, N_cam, C, H, W]
        else:
            images = torch.stack(image_tensors, dim=1)  # [B, N_cam, C, H, W]

        masks = torch.stack(image_masks, dim=1)
        return images, masks

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        images, image_attention_mask = self._prepare_images(observation)
        observation[PISTAR_06_TD_IMAGES_KEY] = images.to(dtype=torch.float32)
        observation[PISTAR_06_TD_IMAGE_MASK_KEY] = image_attention_mask.to(dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pistar_06_td_pre_post_processors(
    config: Pistar_06_tdConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,  # noqa: ARG001
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [k for k in (config.input_features or {}) if k.startswith(OBS_IMAGES)]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        Pistar_06_tdPrepareTaskPromptProcessorStep(
            task_key=config.task_field,
            include_state_in_prompt=config.include_state_in_prompt,
            state_feature=config.state_feature,
            max_state_dim=config.max_state_dim,
            state_discretization_bins=config.state_discretization_bins,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.language_repo_id,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        Pistar_06_tdPrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
