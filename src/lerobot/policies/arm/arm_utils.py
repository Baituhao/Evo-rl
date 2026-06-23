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

"""ARM utility functions: focal loss, label generation, frame indexing."""

import torch
import torch.nn.functional as F  # noqa: N812


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss: -alpha * (1 - p_t)^gamma * log(p_t).

    Here ``p_t`` is the predicted probability of the ground-truth class and
    ``alpha`` is a flat scaling factor (the ARM paper uses alpha=2.0), NOT a
    per-class balancing term. ``-log(p_t)`` is computed via BCE-with-logits.

    Args:
        logits: (B, T) raw logits before sigmoid
        targets: (B, T) binary targets {0, 1}
        alpha: flat loss-scaling factor
        gamma: focusing parameter

    Returns:
        Scalar mean focal loss
    """
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")  # -log(p_t)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()


def compute_advantage_labels(
    progress_values: torch.Tensor,
    delta_threshold: float = 0.01,
) -> torch.Tensor:
    """Derive tri-state transition labels between consecutive frames.

    The advantage head predicts the transition between each consecutive pair of
    hidden states (h_i, h_{i+1}), so a window of T frames yields T-1 labels.
    Each label encodes the progress-based advantage y in {-1, 0, +1} mapped to:
      0 = regressing  (-1): progress decreased
      1 = stagnant     (0): no significant change
      2 = progressing (+1): progress increased

    NOTE: This heuristic (from a linear progress prior) is a placeholder. At
    training time, ground-truth advantage labels of shape (B, T-1) from the
    dataset should be passed in directly.

    Args:
        progress_values: (B, T) normalized progress in [0, 1]
        delta_threshold: minimum delta to count as progressing/regressing

    Returns:
        labels: (B, T-1) long tensor with values in {0, 1, 2}
    """
    B, T = progress_values.shape
    if T < 2:
        return torch.ones(B, 0, dtype=torch.long, device=progress_values.device)

    delta = progress_values[:, 1:] - progress_values[:, :-1]  # (B, T-1)
    labels = torch.ones(B, T - 1, dtype=torch.long, device=progress_values.device)
    labels[delta > delta_threshold] = 2
    labels[delta < -delta_threshold] = 0
    return labels


def compute_completion_labels(
    progress_values: torch.Tensor,
    threshold: float = 0.95,
) -> torch.Tensor:
    """Derive binary task completion labels.

    Args:
        progress_values: (B, T) normalized progress in [0, 1]
        threshold: P_t >= threshold means task is complete (1 - epsilon)

    Returns:
        labels: (B, T) float tensor with values in {0.0, 1.0}
    """
    return (progress_values >= threshold).float()


def compute_causal_indices(
    frame_idx: int,
    ep_start: int,
    ep_end: int,
    window_size: int,
    frame_gap: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal frame indices: past frames + current.

    Window: [t - (window_size-1)*gap, ..., t - gap, t]
    Out-of-bounds frames are clamped to episode boundaries.

    Args:
        frame_idx: target frame index (rightmost in causal window)
        ep_start: episode start index
        ep_end: episode end index (exclusive)
        window_size: number of frames in the causal window
        frame_gap: gap between frames (at 30fps, gap=30 gives 1Hz)

    Returns:
        Tuple of (indices, out_of_bounds_flags)
    """
    deltas = [-frame_gap * i for i in range(window_size - 1, 0, -1)] + [0]

    frames = []
    out_of_bounds = []
    for delta in deltas:
        target = frame_idx + delta
        clamped = max(ep_start, min(ep_end - 1, target))
        frames.append(clamped)
        out_of_bounds.append(1 if target != clamped else 0)

    return torch.tensor(frames), torch.tensor(out_of_bounds)


def pad_state_to_max_dim(state: torch.Tensor, max_state_dim: int) -> torch.Tensor:
    """Pad the state tensor's last dimension to max_state_dim with zeros."""
    current_dim = state.shape[-1]
    if current_dim >= max_state_dim:
        return state[..., :max_state_dim]
    padding = (0, max_state_dim - current_dim)
    return F.pad(state, padding, mode="constant", value=0)
