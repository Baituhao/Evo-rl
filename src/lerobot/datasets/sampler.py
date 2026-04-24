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
from collections.abc import Iterator

import pandas as pd
import torch

from lerobot.utils.recording_annotations import EPISODE_FAILURE, normalize_episode_success_label


def _is_missing_episode_success(label: object) -> bool:
    if label is None:
        return True
    if hasattr(label, "item") and not isinstance(label, (str, bytes)):
        label = label.item()
    if pd.isna(label):
        return True
    return isinstance(label, str) and label.strip() == ""


def resolve_episode_indices_to_use(
    episodes_metadata,
    requested_episodes: list[int] | None = None,
    omit_failed: bool = False,
    repo_id: str | None = None,
) -> list[int] | None:
    if not omit_failed:
        return requested_episodes

    episodes_meta = episodes_metadata.with_format(None)
    repo_label = repo_id if repo_id is not None else "<unknown>"
    if "episode_success" not in episodes_meta.column_names:
        logging.debug(
            "dataset.omit_failed=true but dataset '%s' has no 'episode_success' column; keeping all requested episodes.",
            repo_label,
        )
        return requested_episodes

    requested_episode_set = set(requested_episodes) if requested_episodes is not None else None
    kept_episodes: list[int] = []
    dropped_failure_episodes: list[int] = []
    unlabeled_kept = 0
    kept_frames = 0
    dropped_frames = 0

    for ep_idx, ep_success, from_idx, to_idx in zip(
        episodes_meta["episode_index"],
        episodes_meta["episode_success"],
        episodes_meta["dataset_from_index"],
        episodes_meta["dataset_to_index"],
        strict=False,
    ):
        episode_index = int(ep_idx)
        if requested_episode_set is not None and episode_index not in requested_episode_set:
            continue

        num_frames = int(to_idx) - int(from_idx)
        if _is_missing_episode_success(ep_success):
            kept_episodes.append(episode_index)
            kept_frames += num_frames
            unlabeled_kept += 1
            continue

        normalized_success = normalize_episode_success_label(ep_success)
        if normalized_success == EPISODE_FAILURE:
            dropped_failure_episodes.append(episode_index)
            dropped_frames += num_frames
            continue

        kept_episodes.append(episode_index)
        kept_frames += num_frames

    logging.info(
        (
            "Episode filtering for dataset '%s': omit_failed=%s requested_episodes=%s "
            "kept_episodes=%d dropped_failure_episodes=%d kept_frames=%d dropped_frames=%d unlabeled_kept=%d"
        ),
        repo_label,
        omit_failed,
        "all" if requested_episodes is None else len(requested_episodes),
        len(kept_episodes),
        len(dropped_failure_episodes),
        kept_frames,
        dropped_frames,
        unlabeled_kept,
    )
    if dropped_failure_episodes:
        logging.info(
            "Dropped failure episodes from dataset '%s' (first 20): %s",
            repo_label,
            dropped_failure_episodes[:20],
        )

    return kept_episodes


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        index_mapping: dict[int, int] | None = None,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            index_mapping: Optional mapping from absolute frame indices to relative dataset indices.
        """
        allowed_episode_indices = set(episode_indices_to_use) if episode_indices_to_use is not None else None
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if allowed_episode_indices is None or episode_idx in allowed_episode_indices:
                episode_indices = range(start_index + drop_n_first_frames, end_index - drop_n_last_frames)
                if index_mapping is None:
                    indices.extend(episode_indices)
                else:
                    indices.extend(index_mapping[idx] for idx in episode_indices if idx in index_mapping)

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
