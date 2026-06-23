#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("pistar_06_td")
@dataclass
class Pistar_06_tdConfig(PreTrainedConfig):
    """Value model config using the Pistar_06_td stack (SigLIP + Gemma)."""

    # Backbone components
    vision_repo_id: str = "google/siglip-so400m-patch14-384"
    language_repo_id: str = "google/gemma-3-270m"
    vision_revision: str | None = None
    language_revision: str | None = None

    # Input fields
    task_field: str = "task"
    camera_features: list[str] = field(default_factory=list)
    state_feature: str = OBS_STATE
    include_state_in_prompt: bool = True
    max_state_dim: int = 32
    state_discretization_bins: int = 256
    target_key: str = "observation.value_target"
    loss_weight_key: str = "observation.value_loss_weight"
    task_index_feature: str = "task_index"

    # Tokenizer / model shape
    tokenizer_max_length: int = 200
    state_proj_dim: int = 512
    fusion_hidden_dim: int = 512
    fusion_num_layers: int = 2
    fusion_num_heads: int = 8

    # Value head (distributional)
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0

    # TD loss (RISE-style online bootstrap with target network).
    # Uses sparse rewards at episode termination and bootstraps from an EMA target model.
    td_loss_weight: float = 1.0  # set to 0 to disable the TD term (pure cross-entropy)
    td_gamma: float = 0.99  # discount factor
    td_terminal_window: int = 10  # frames within this distance from episode end get terminal reward
    td_success_reward: float = 1.0  # reward at successful episode termination
    td_failure_reward: float = -1.0  # reward at failed episode termination
    target_model_ema_decay: float = 0.995  # EMA decay rate for target model updates

    # Runtime
    dropout: float = 0.1
    dtype: str = "float32"
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    use_gradient_checkpointing: bool = False
    push_to_hub: bool = False

    # Training presets
    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 8_000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.vision_repo_id:
            raise ValueError("'value.vision_repo_id' must be non-empty.")
        if not self.language_repo_id:
            raise ValueError("'value.language_repo_id' must be non-empty.")
        if not self.task_field:
            raise ValueError("'value.task_field' must be non-empty.")
        if not self.state_feature:
            raise ValueError("'value.state_feature' must be non-empty.")
        if not self.state_feature.startswith("observation."):
            raise ValueError("'value.state_feature' must start with 'observation.'.")
        if not self.target_key:
            raise ValueError("'value.target_key' must be non-empty.")
        if not self.loss_weight_key:
            raise ValueError("'value.loss_weight_key' must be non-empty.")
        if not self.loss_weight_key.startswith("observation."):
            raise ValueError("'value.loss_weight_key' must start with 'observation.'.")
        if self.max_state_dim <= 0:
            raise ValueError("'value.max_state_dim' must be > 0.")
        if self.state_discretization_bins < 2:
            raise ValueError("'value.state_discretization_bins' must be >= 2.")

        if self.tokenizer_max_length <= 0:
            raise ValueError("'value.tokenizer_max_length' must be > 0.")
        if self.state_proj_dim <= 0:
            raise ValueError("'value.state_proj_dim' must be > 0.")
        if self.fusion_hidden_dim <= 0:
            raise ValueError("'value.fusion_hidden_dim' must be > 0.")
        if self.fusion_num_layers <= 0:
            raise ValueError("'value.fusion_num_layers' must be > 0.")
        if self.fusion_num_heads <= 0:
            raise ValueError("'value.fusion_num_heads' must be > 0.")
        if self.fusion_hidden_dim % self.fusion_num_heads != 0:
            raise ValueError("'value.fusion_hidden_dim' must be divisible by 'value.fusion_num_heads'.")

        if self.num_bins < 2:
            raise ValueError("'value.num_bins' must be >= 2.")
        if self.bin_min >= self.bin_max:
            raise ValueError("'value.bin_min' must be < 'value.bin_max'.")
        if self.td_loss_weight < 0:
            raise ValueError("'value.td_loss_weight' must be >= 0.")
        if not 0.0 < self.td_gamma <= 1.0:
            raise ValueError("'value.td_gamma' must be within (0, 1].")
        if self.td_terminal_window < 0:
            raise ValueError("'value.td_terminal_window' must be >= 0.")
        if not 0.0 < self.target_model_ema_decay < 1.0:
            raise ValueError("'value.target_model_ema_decay' must be within (0, 1).")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("'value.dtype' must be one of {'float32', 'bfloat16'}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("'value.dropout' must be within [0, 1).")

        if self.optimizer_lr <= 0:
            raise ValueError("'value.optimizer_lr' must be > 0.")
        if self.optimizer_weight_decay < 0:
            raise ValueError("'value.optimizer_weight_decay' must be >= 0.")
        if self.optimizer_grad_clip_norm < 0:
            raise ValueError("'value.optimizer_grad_clip_norm' must be >= 0.")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("'value.scheduler_warmup_steps' must be >= 0.")
        if self.scheduler_decay_steps <= 0:
            raise ValueError("'value.scheduler_decay_steps' must be > 0.")
        if self.scheduler_decay_lr < 0:
            raise ValueError("'value.scheduler_decay_lr' must be >= 0.")

    def validate_features(self) -> None:
        # Value model consumes observation + task text, and supervises with a scalar target key.
        # The training loop injects target tensors into `target_key`.
        return

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Return current frame (0) and next frame (1) for RISE-style online TD bootstrap.

        The training factory (`resolve_delta_timestamps`) iterates over this list and applies
        it to every observation key, so it must be a flat list of integer deltas (not a dict).
        """
        if self.td_loss_weight > 0:
            return [0, 1]
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
