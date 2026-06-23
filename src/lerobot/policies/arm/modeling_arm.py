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

"""
ARM: Advantage Reward Modeling for Long-Horizon Manipulation.

Architecture:
  - Additive input fusion: x_i = MLP(v_i) + MLP(s_i) + MLP(g)
  - Single 8-layer causal Transformer encoder
  - Multi-frame Advantage Head: tri-state classification (neg/neutral/pos)
  - Task Completion Head: binary classification with focal loss
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.arm.arm_utils import focal_loss, pad_state_to_max_dim
from lerobot.policies.arm.configuration_arm import ARMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STR


class ARMEncoder(nn.Module):
    """Single Transformer encoder with additive input fusion.

    x_i = MLP_v(v_i) + MLP_s(s_i) + MLP_g(g)
    {h} = TransformerEncoder({x_i}, causal_mask)
    """

    def __init__(
        self,
        d_model: int = 512,
        image_dim: int = 512,
        text_dim: int = 512,
        state_dim: int = 32,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        self.visual_mlp = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.ln = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        visual_features: torch.Tensor,
        state_features: torch.Tensor,
        text_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, image_dim) CLIP image features
            state_features: (B, T, state_dim) proprioceptive state
            text_features: (B, text_dim) CLIP text features (goal)
            lengths: (B,) valid sequence lengths

        Returns:
            h: (B, T, d_model) transformer hidden states
        """
        B, T, _ = visual_features.shape

        v = self.visual_mlp(visual_features)
        s = self.state_mlp(state_features)
        g = self.goal_mlp(text_features).unsqueeze(1).expand(-1, T, -1)

        x = v + s + g
        x = x + self.pos_embedding[:, :T, :]

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        pad_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)

        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
        return self.ln(h)


class AdvantageHead(nn.Module):
    """Multi-frame advantage head: tri-state classification of transitions.

    The advantage is defined on the transition between consecutive hidden
    states (h_i, h_{i+1}), per the ARM paper. The two states are concatenated
    and classified, so a window of T frames yields T-1 transition predictions.
    """

    def __init__(self, d_model: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Linear(2 * d_model, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> transition logits (B, T-1, num_classes).

        Returns an empty (B, 0, num_classes) tensor when T < 2.
        """
        if h.shape[1] < 2:
            return h.new_zeros(h.shape[0], 0, self.classifier.out_features)
        pair = torch.cat([h[:, :-1, :], h[:, 1:, :]], dim=-1)  # (B, T-1, 2D)
        return self.classifier(pair)


class TaskCompletionHead(nn.Module):
    """Task completion head: binary classification of the current frame.

    Per the ARM paper, the completion head C predicts the probability that the
    *current* observation s_t (the rightmost / most recent frame in the causal
    window) is a successful terminal state — a single prediction, not per-frame.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> logits for the current frame: (B, 1)"""
        return self.classifier(h[:, -1, :])


class ARMRewardModel(PreTrainedPolicy):
    """ARM Reward Model.

    Uses a single Transformer encoder with additive fusion and two output heads:
    - Advantage head: tri-state per-frame classification (negative/neutral/positive)
    - Completion head: binary per-frame classification (task done or not)

    Joint loss: L = lambda_int * CE(advantage) + lambda_succ * FocalLoss(completion)
    """

    name = "arm"
    config_class = ARMConfig

    def __init__(self, config: ARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.device_setting = torch.device(
            config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = ARMEncoder(
            d_model=config.hidden_dim,
            image_dim=config.image_dim,
            text_dim=config.text_dim,
            state_dim=config.max_state_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            window_size=config.window_size,
        )

        self.advantage_head = AdvantageHead(
            d_model=config.hidden_dim,
            num_classes=config.num_advantage_classes,
        )

        self.completion_head = TaskCompletionHead(d_model=config.hidden_dim)

        self.encoder.to(self.device_setting)
        self.advantage_head.to(self.device_setting)
        self.completion_head.to(self.device_setting)

        logging.info(
            f"ARM initialized: d_model={config.hidden_dim}, layers={config.num_layers}, "
            f"heads={config.num_heads}, window={config.window_size}, device={self.device_setting}"
        )

    def to(self, device):
        super().to(device)
        self.device_setting = device if isinstance(device, torch.device) else torch.device(device)
        self.encoder.to(device)
        self.advantage_head.to(device)
        self.completion_head.to(device)
        return self

    def forward(self, batch):
        """Training forward pass.

        Args:
            batch: Dictionary with 'observation' containing:
                - 'video_features': (B, T, 512)
                - 'text_features': (B, 512)
                - 'state_features': (B, T, state_dim)
                - 'lengths': (B,)
                - 'advantage_targets': (B, T-1) long {0, 1, 2}  (per transition)
                - 'completion_targets': (B, T) float {0.0, 1.0}  (per frame)

        Returns:
            Tuple of (total_loss, output_dict)
        """
        observation = batch.get(OBS_STR, batch)

        video_features = observation["video_features"].to(self.device_setting)
        text_features = observation["text_features"].to(self.device_setting)
        state_features = observation.get("state_features")

        batch_size = video_features.shape[0]
        seq_len = video_features.shape[1]

        if state_features is not None:
            state_features = state_features.to(self.device_setting)
        else:
            state_features = torch.zeros(
                batch_size, seq_len, self.config.max_state_dim, device=self.device_setting
            )

        state_features = pad_state_to_max_dim(state_features, self.config.max_state_dim)

        lengths = observation.get("lengths")
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=self.device_setting)
        else:
            lengths = lengths.to(self.device_setting)

        # Encoder
        h = self.encoder(video_features, state_features, text_features, lengths)

        # Per-frame valid mask (frame index < length)
        frame_valid = torch.arange(seq_len, device=self.device_setting).expand(
            batch_size, seq_len
        ) < lengths.unsqueeze(1)  # (B, T)

        # Advantage head: tri-state per transition (h_i, h_{i+1}) -> (B, T-1, C)
        adv_logits = self.advantage_head(h)
        advantage_targets = observation["advantage_targets"].to(self.device_setting).long()
        # A transition i->i+1 is valid only if both endpoint frames are valid.
        trans_valid = frame_valid[:, :-1] & frame_valid[:, 1:]  # (B, T-1)
        num_classes = self.config.num_advantage_classes
        if trans_valid.any():
            adv_loss = F.cross_entropy(
                adv_logits.reshape(-1, num_classes)[trans_valid.reshape(-1)],
                advantage_targets.reshape(-1)[trans_valid.reshape(-1)],
                reduction="mean",
            )
        else:
            adv_loss = adv_logits.sum() * 0.0

        # Completion head: binary for current frame only -> (B, 1)
        comp_logits = self.completion_head(h).squeeze(-1)  # (B,)
        completion_targets = observation["completion_targets"].to(self.device_setting).float()
        # completion_targets shape: (B,) — label for the rightmost frame
        if completion_targets.dim() > 1:
            completion_targets = completion_targets[:, -1]  # take current frame
        comp_loss = focal_loss(
            comp_logits,
            completion_targets,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )

        total_loss = self.config.lambda_int * adv_loss + self.config.lambda_succ * comp_loss

        output_dict = {
            "advantage_loss": adv_loss.item(),
            "completion_loss": comp_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, output_dict

    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        return_confidence: bool = False,
        head_mode: str | None = "sparse",
        frame_index: int | None = None,
    ) -> np.ndarray | tuple:
        """Calculate rewards (task completion probability) for inference.

        Args:
            text_embeddings: (B, 512) or (512,) CLIP text features
            video_embeddings: (B, T, 512) or (T, 512) CLIP image features
            state_features: (B, T, state_dim) or None
            lengths: (B,) valid sequence lengths or None
            return_all_frames: if True, return per-frame values
            return_stages: if True, also return advantage probabilities
            return_confidence: if True, also return completion confidence
            head_mode: accepted for compatibility, ignored
            frame_index: target frame index (default: last frame in window)

        Returns:
            Rewards as numpy array, optionally with advantage probs / confidence
        """
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)

        single_sample = False
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            if state_features is not None:
                state_features = state_features.unsqueeze(0)
            single_sample = True

        batch_size = video_embeddings.shape[0]
        seq_len = video_embeddings.shape[1]

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        elif isinstance(lengths, np.ndarray):
            lengths = torch.tensor(lengths, dtype=torch.int32)

        video_emb = video_embeddings.to(self.device_setting)
        text_emb = text_embeddings.to(self.device_setting)
        state = (
            state_features.to(self.device_setting)
            if state_features is not None
            else torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device_setting)
        )
        state = pad_state_to_max_dim(state, self.config.max_state_dim)
        lens = lengths.to(self.device_setting)

        h = self.encoder(video_emb, state, text_emb, lens)

        # Completion probability for the current (rightmost) frame -> (B,)
        comp_logits = self.completion_head(h).squeeze(-1)
        completion_prob = torch.sigmoid(comp_logits)  # (B,)

        # Advantage transition probabilities: (B, T-1, num_classes)
        adv_logits = self.advantage_head(h)
        adv_probs = F.softmax(adv_logits, dim=-1)

        # Rewards: use completion probability as progress signal (placeholder)
        rewards = completion_prob.cpu().numpy()  # (B,)
        if single_sample:
            rewards = rewards[0]

        outputs = [rewards]

        if return_stages:
            # Advantage is per-transition (T-1). For all-frames return the full
            # (B, T-1, C); otherwise return the transition ending at frame_index.
            if return_all_frames:
                probs = adv_probs.cpu().numpy()
                labels = adv_probs.argmax(dim=-1).cpu().numpy() - 1  # {0,1,2} -> {-1,0,+1}
            else:
                # Default to the last transition (most recent frame pair) in the window.
                trans_idx = (
                    adv_probs.shape[1] - 1
                    if frame_index is None
                    else min(max(frame_index - 1, 0), adv_probs.shape[1] - 1)
                )
                probs = adv_probs[:, trans_idx].cpu().numpy()
                labels = adv_probs[:, trans_idx].argmax(dim=-1).cpu().numpy() - 1
            if single_sample:
                probs = probs[0]
                labels = labels[0]
            outputs.append((labels, probs))

        if return_confidence:
            conf = completion_prob.cpu().numpy()
            if single_sample:
                conf = conf[0]
            outputs.append(conf)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)
        self.advantage_head.train(mode)
        self.completion_head.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        from itertools import chain

        return chain(
            self.encoder.parameters(),
            self.advantage_head.parameters(),
            self.completion_head.parameters(),
        )

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ARM is a reward model, not an action policy")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ARM is a reward model, not an action policy")
