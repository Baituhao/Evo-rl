"""
Checkpoint management for resumable inference.

Supports incremental inference with episode-level granularity,
allowing recovery from crashes without re-running completed episodes.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class InferenceCheckpoint:
    """Checkpoint state for resumable inference.

    Attributes:
        total_episodes: Total number of episodes to process
        completed_episodes: List of episode IDs that have been completed
        last_update: ISO timestamp of last checkpoint update
        status: Current inference status (inference/merging/completed)
        config_hash: Hash of inference config to detect config changes
        total_frames_processed: Total number of frames processed so far
        start_time: ISO timestamp when inference started
        estimated_completion: ISO timestamp of estimated completion (optional)
        current_episode: Current episode being processed (for progress tracking)
    """
    total_episodes: int
    completed_episodes: list[int]
    last_update: str
    status: str  # "inference" | "merging" | "completed"
    config_hash: str
    total_frames_processed: int
    start_time: str
    estimated_completion: Optional[str] = None
    current_episode: Optional[int] = None

    def progress_ratio(self) -> float:
        """Calculate completion progress as ratio [0, 1]."""
        if self.total_episodes == 0:
            return 1.0
        return len(self.completed_episodes) / self.total_episodes

    def is_episode_completed(self, ep_idx: int) -> bool:
        """Check if an episode has been completed."""
        return ep_idx in self.completed_episodes

    def mark_episode_completed(self, ep_idx: int, frames_count: int) -> None:
        """Mark an episode as completed and update statistics."""
        if ep_idx not in self.completed_episodes:
            self.completed_episodes.append(ep_idx)
            self.total_frames_processed += frames_count
            self.last_update = datetime.now().isoformat()
            self.current_episode = None

    def set_current_episode(self, ep_idx: int) -> None:
        """Set the episode currently being processed."""
        self.current_episode = ep_idx
        self.last_update = datetime.now().isoformat()


def compute_config_hash(cfg: dict) -> str:
    """Compute a hash of the configuration for change detection.

    Args:
        cfg: Configuration dictionary

    Returns:
        SHA256 hash of the configuration (first 16 characters)
    """
    # Serialize config to JSON with sorted keys for deterministic hash
    config_str = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def save_checkpoint(checkpoint: InferenceCheckpoint, checkpoint_path: Path) -> None:
    """Save checkpoint to disk (atomic write).

    Args:
        checkpoint: Checkpoint state to save
        checkpoint_path: Path to checkpoint file
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first (atomic operation)
    tmp_path = checkpoint_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(asdict(checkpoint), f, indent=2)

    # Atomic rename
    tmp_path.replace(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> InferenceCheckpoint:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded checkpoint state

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint file is corrupted
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        return InferenceCheckpoint(**data)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Corrupted checkpoint file: {checkpoint_path}") from e


def create_checkpoint(
    total_episodes: int,
    config_hash: str,
) -> InferenceCheckpoint:
    """Create a new checkpoint for a fresh inference run.

    Args:
        total_episodes: Total number of episodes to process
        config_hash: Hash of the inference configuration

    Returns:
        New checkpoint instance
    """
    now = datetime.now().isoformat()
    return InferenceCheckpoint(
        total_episodes=total_episodes,
        completed_episodes=[],
        last_update=now,
        status="inference",
        config_hash=config_hash,
        total_frames_processed=0,
        start_time=now,
        estimated_completion=None,
        current_episode=None,
    )
