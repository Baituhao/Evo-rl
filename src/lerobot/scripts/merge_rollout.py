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
Merge multiple local LeRobot datasets while aligning schemas and tagging sources.

This script is intended for expert + rollout style merges where sources may not
share identical feature sets. It will:

- merge multiple local datasets
- add `is_expert` and `source_dataset_index` frame-level labels
- ignore `complementary_info.*` features
- align non-visual schemas using the union of compatible features
- fill missing non-visual features with zeros
- force output fps metadata to a configured value (default: 10)
- regenerate `meta/stats.json` during the merge

Notes:
- Missing image/video features are treated as an error and are not synthesized.
- Feature dtype/shape mismatches on the same key are treated as an error.
- Existing output directories are moved to `<output>_old`.

Example:

    python -m lerobot.scripts.merge_rollout \
        --repo_id merged/libero_with_rollout \
        --output_dir /mnt/data/syk/Evo-RL/outputs/libero_with_rollout \
        --sources "[
            {
                path: datasets_oss/HuggingFaceVLA/libero,
                repo_id: HuggingFaceVLA/libero,
                is_expert: true
            },
            {
                path: outputs/policy_rollout_0_20260322_103353/dataset,
                repo_id: rollout_0,
                is_expert: false
            }
        ]"
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import draccus
import numpy as np
import pandas as pd

from lerobot.configs import parser
from lerobot.datasets.aggregate import append_or_create_parquet_file
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURES,
    get_hf_features_from_features,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS
from lerobot.utils.utils import init_logging

IGNORED_FEATURE_PREFIXES = ("complementary_info.",)
DROPPED_FRAME_FEATURES = {"next.done", "next.success", "next.reward"}
EXPERT_LABEL_FEATURE = "is_expert"
SOURCE_INDEX_FEATURE = "source_dataset_index"


def _should_compute_quantile_stats(feature_name: str, feature_info: dict[str, Any]) -> bool:
    if feature_info["dtype"] in {"image", "video", "string"}:
        return False
    if feature_name == "action":
        return True
    if feature_name.startswith("observation."):
        return True
    return False


@dataclass
class SourceConfig:
    path: str
    is_expert: bool
    repo_id: str | None = None


@dataclass
class MergeRolloutConfig:
    repo_id: str
    sources: list[SourceConfig]
    output_dir: str | None = None
    push_to_hub: bool = False
    fps: int = 10
    ignore_feature_prefixes: list[str] = field(default_factory=lambda: list(IGNORED_FEATURE_PREFIXES))


def _should_ignore_feature(feature_name: str, ignore_prefixes: tuple[str, ...]) -> bool:
    return any(feature_name.startswith(prefix) for prefix in ignore_prefixes)


def _normalize_names(names: Any) -> Any:
    if names is None:
        return None
    if isinstance(names, tuple):
        names = list(names)
    if isinstance(names, list):
        return ["channel" if name == "channels" else name for name in names]
    return names


def _canonicalize_feature_info(feature_name: str, feature_info: dict[str, Any], target_fps: int) -> dict[str, Any]:
    canonical = {
        "dtype": feature_info["dtype"],
        "shape": tuple(feature_info["shape"]),
        "names": _normalize_names(feature_info.get("names")),
    }
    if feature_info["dtype"] in {"image", "video"}:
        canonical["fps"] = float(target_fps)
    elif "fps" in feature_info:
        canonical["fps"] = feature_info["fps"]
    return canonical


def _feature_signature(feature_info: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return feature_info["dtype"], tuple(feature_info["shape"])


def _resolve_source_repo_id(source: SourceConfig) -> str:
    if source.repo_id:
        return source.repo_id
    return Path(source.path).name


def _prepare_output_dir(cfg: MergeRolloutConfig) -> Path:
    output_dir = Path(cfg.output_dir) if cfg.output_dir else HF_LEROBOT_HOME / cfg.repo_id
    backup_dir = Path(str(output_dir) + "_old")

    if output_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(output_dir), str(backup_dir))

    return output_dir


def _load_dataset_for_source(source: SourceConfig) -> LeRobotDataset:
    source_path = Path(source.path)
    repo_id = _resolve_source_repo_id(source)
    return LeRobotDataset(repo_id=repo_id, root=source_path)


def _collect_tasks(datasets_by_source: list[tuple[SourceConfig, LeRobotDataset]]) -> pd.DataFrame:
    tasks: list[str] = []
    seen: set[str] = set()
    for _, dataset in datasets_by_source:
        for task in dataset.meta.tasks.index:
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    return pd.DataFrame({"task_index": range(len(tasks))}, index=tasks)


def _choose_robot_type(datasets_by_source: list[tuple[SourceConfig, LeRobotDataset]]) -> str | None:
    robot_types = [dataset.meta.robot_type for _, dataset in datasets_by_source if dataset.meta.robot_type is not None]
    if not robot_types:
        return None

    chosen = robot_types[0]
    for robot_type in robot_types[1:]:
        if robot_type != chosen:
            logging.warning(
                "Conflicting robot_type values detected during merge. Using '%s' and ignoring '%s'.",
                chosen,
                robot_type,
            )
    return chosen


def _build_target_features(
    datasets_by_source: list[tuple[SourceConfig, LeRobotDataset]],
    ignore_prefixes: tuple[str, ...],
    target_fps: int,
) -> dict[str, dict[str, Any]]:
    target_features: dict[str, dict[str, Any]] = {}

    for _, dataset in datasets_by_source:
        for feature_name, feature_info in dataset.meta.features.items():
            if feature_name in DEFAULT_FEATURES:
                continue
            if feature_name in DROPPED_FRAME_FEATURES:
                continue
            if _should_ignore_feature(feature_name, ignore_prefixes):
                continue

            canonical = _canonicalize_feature_info(feature_name, feature_info, target_fps)
            if feature_name not in target_features:
                target_features[feature_name] = canonical
                continue

            if _feature_signature(target_features[feature_name]) != _feature_signature(canonical):
                raise ValueError(
                    f"Incompatible feature '{feature_name}' detected while aligning schemas: "
                    f"{target_features[feature_name]} vs {canonical}"
                )

    for label_name in (EXPERT_LABEL_FEATURE, SOURCE_INDEX_FEATURE):
        if label_name in target_features:
            raise ValueError(f"Target feature '{label_name}' already exists in one of the source datasets.")
        target_features[label_name] = {"dtype": "int64", "shape": (1,), "names": None}

    return target_features


def _default_scalar_value(dtype: str, value: int | float) -> Any:
    np_dtype = np.dtype(dtype)
    return np_dtype.type(value).item()


def _default_feature_value(feature_info: dict[str, Any], fill_value: int | float = 0) -> Any:
    shape = tuple(feature_info["shape"])
    dtype = feature_info["dtype"]

    if shape == (1,):
        return _default_scalar_value(dtype, fill_value)

    return np.full(shape, fill_value, dtype=np.dtype(dtype))


def _load_source_data_frame(dataset: LeRobotDataset, parquet_path: Path) -> pd.DataFrame:
    if len(dataset.meta.image_keys) > 0:
        return datasets.Dataset.from_parquet(str(parquet_path)).to_pandas()
    return pd.read_parquet(parquet_path)


def _series_to_numpy(series: pd.Series, feature_info: dict[str, Any]) -> np.ndarray:
    shape = tuple(feature_info["shape"])
    dtype = np.dtype(feature_info["dtype"])

    if shape == (1,):
        first_valid = next((value for value in series.tolist() if value is not None), None)
        if isinstance(first_valid, (np.ndarray, list, tuple)):
            return np.stack([np.asarray(value, dtype=dtype) for value in series.tolist()]).astype(dtype, copy=False)
        return series.to_numpy(dtype=dtype)

    return np.stack([np.asarray(value, dtype=dtype) for value in series.tolist()]).astype(dtype, copy=False)


def _build_episode_stats(
    aligned_episode_df: pd.DataFrame,
    target_features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    numeric_features = {
        name: info
        for name, info in target_features.items()
        if _should_compute_quantile_stats(name, info)
    }
    episode_numeric_data = {
        name: _series_to_numpy(aligned_episode_df[name], feature_info)
        for name, feature_info in numeric_features.items()
    }
    episode_stats = compute_episode_stats(episode_numeric_data, numeric_features)

    return episode_stats


def _collect_source_episode_rows(dataset: LeRobotDataset) -> dict[int, dict[str, Any]]:
    episode_rows: dict[int, dict[str, Any]] = {}
    episodes_dir = dataset.root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for row in df.to_dict(orient="records"):
            episode_rows[int(row["episode_index"])] = row
    return episode_rows


def _list_source_data_files(dataset: LeRobotDataset) -> list[Path]:
    data_dir = dataset.root / "data"
    data_files = sorted(data_dir.glob("*/*.parquet"))
    if not data_files:
        raise ValueError(f"No data parquet files found under {data_dir}")

    episode_to_file: dict[int, Path] = {}
    file_ranges: list[tuple[int, int, Path]] = []
    for parquet_path in data_files:
        episode_df = pd.read_parquet(parquet_path, columns=["episode_index"])
        unique_episode_indices = sorted(map(int, pd.unique(episode_df["episode_index"])))
        if not unique_episode_indices:
            continue

        for episode_index in unique_episode_indices:
            previous_path = episode_to_file.get(episode_index)
            if previous_path is not None and previous_path != parquet_path:
                raise ValueError(
                    "Episode indices spanning multiple source data parquet files are not supported. "
                    f"Episode {episode_index} appears in both {previous_path} and {parquet_path}."
                )
            episode_to_file[episode_index] = parquet_path

        file_ranges.append((unique_episode_indices[0], unique_episode_indices[-1], parquet_path))

    return [path for _, _, path in sorted(file_ranges)]


def _infer_episode_success_label(
    episode_metadata: dict[str, Any],
    source_episode_df: pd.DataFrame,
) -> str:
    if "episode_success" in episode_metadata:
        return episode_metadata["episode_success"]

    if "next.success" in source_episode_df.columns:
        next_success = source_episode_df["next.success"].to_numpy()
        if next_success.size > 0:
            return EPISODE_SUCCESS if bool(np.max(next_success)) else EPISODE_FAILURE

    return EPISODE_SUCCESS


def _extract_optional_episode_metadata(episode_row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in episode_row.items():
        if key in {"episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"}:
            continue
        if key.startswith("stats/"):
            continue
        if key.startswith("meta/episodes/"):
            continue
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        metadata[key] = value
    return metadata


def _align_data_frame(
    df: pd.DataFrame,
    target_features: dict[str, dict[str, Any]],
    source_index: int,
    is_expert: bool,
) -> pd.DataFrame:
    aligned = df.copy()

    for feature_name in list(aligned.columns):
        if feature_name not in target_features and feature_name not in DEFAULT_FEATURES:
            aligned = aligned.drop(columns=[feature_name])

    aligned[EXPERT_LABEL_FEATURE] = _default_scalar_value("int64", int(is_expert))
    aligned[SOURCE_INDEX_FEATURE] = _default_scalar_value("int64", source_index)

    full_target_features = {**target_features, **DEFAULT_FEATURES}
    for feature_name, feature_info in full_target_features.items():
        if feature_name in aligned.columns:
            continue

        if feature_info["dtype"] in {"image", "video"}:
            raise ValueError(
                f"Feature '{feature_name}' is required by the aligned schema but is missing from one source dataset. "
                "Missing visual features cannot be synthesized."
            )

        default_value = _default_feature_value(feature_info, fill_value=0)
        if tuple(feature_info["shape"]) == (1,):
            aligned[feature_name] = [_default_scalar_value(feature_info["dtype"], 0)] * len(aligned)
        else:
            aligned[feature_name] = [default_value.copy() for _ in range(len(aligned))]

    ordered_columns = [name for name in full_target_features if name in aligned.columns]
    return aligned[ordered_columns]


def _remap_indices_and_tasks(
    aligned_df: pd.DataFrame,
    dataset: LeRobotDataset,
    global_tasks: pd.DataFrame,
    episode_offset: int,
    frame_offset: int,
) -> pd.DataFrame:
    remapped = aligned_df.copy()
    remapped["episode_index"] = remapped["episode_index"] + episode_offset
    remapped["index"] = remapped["index"] + frame_offset

    task_names = dataset.meta.tasks.index.take(remapped["task_index"].to_numpy())
    remapped["task_index"] = global_tasks.loc[task_names, "task_index"].to_numpy()
    return remapped


def merge_rollout_datasets(cfg: MergeRolloutConfig) -> Path:
    if not cfg.sources:
        raise ValueError("At least one source dataset must be provided.")

    ignore_prefixes = tuple(cfg.ignore_feature_prefixes)
    datasets_by_source = [(source, _load_dataset_for_source(source)) for source in cfg.sources]
    output_dir = _prepare_output_dir(cfg)

    target_features = _build_target_features(datasets_by_source, ignore_prefixes, cfg.fps)
    global_tasks = _collect_tasks(datasets_by_source)
    robot_type = _choose_robot_type(datasets_by_source)
    data_files_size_in_mb = max(
        dataset.meta.data_files_size_in_mb for _, dataset in datasets_by_source
    )
    video_files_size_in_mb = max(
        dataset.meta.video_files_size_in_mb for _, dataset in datasets_by_source
    )

    if any(dataset.meta.video_keys for _, dataset in datasets_by_source):
        raise ValueError("merge_rollout.py currently supports image/non-video datasets only.")

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        features=target_features,
        robot_type=robot_type,
        root=output_dir,
        use_videos=False,
        chunks_size=DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )
    dst_meta.tasks = global_tasks
    write_tasks(global_tasks, output_dir)

    contains_images = len(dst_meta.image_keys) > 0
    hf_features = get_hf_features_from_features(dst_meta.features) if contains_images else None
    data_idx = {"chunk": 0, "file": 0}
    total_episodes = 0
    total_frames = 0

    for source_index, (source_cfg, dataset) in enumerate(datasets_by_source):
        logging.info(
            "Merging source %d/%d: path=%s repo_id=%s is_expert=%s",
            source_index + 1,
            len(datasets_by_source),
            source_cfg.path,
            dataset.repo_id,
            source_cfg.is_expert,
        )
        episode_rows = _collect_source_episode_rows(dataset)
        source_data_files = _list_source_data_files(dataset)

        for src_path in source_data_files:
            src_df = _load_source_data_frame(dataset, src_path)
            aligned_df = _align_data_frame(src_df, target_features, source_index, source_cfg.is_expert)
            remapped_df = _remap_indices_and_tasks(aligned_df, dataset, global_tasks, total_episodes, total_frames)

            data_idx, (dst_chunk, dst_file) = append_or_create_parquet_file(
                remapped_df,
                src_path,
                data_idx,
                data_files_size_in_mb,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_DATA_PATH,
                contains_images=contains_images,
                aggr_root=output_dir,
                hf_features=hf_features,
            )

            for new_episode_index, episode_df in remapped_df.groupby("episode_index", sort=True):
                old_episode_index = int(new_episode_index - total_episodes)
                source_episode_row = episode_rows[old_episode_index]
                source_episode_df = src_df[src_df["episode_index"] == old_episode_index]
                episode_stats = _build_episode_stats(episode_df, dst_meta.features)
                episode_metadata = _extract_optional_episode_metadata(source_episode_row)
                episode_metadata["episode_success"] = _infer_episode_success_label(
                    episode_metadata, source_episode_df
                )
                episode_metadata.update(
                    {
                        "data/chunk_index": dst_chunk,
                        "data/file_index": dst_file,
                        "source_episode_index": old_episode_index,
                        "source_repo_id": dataset.repo_id,
                        EXPERT_LABEL_FEATURE: int(source_cfg.is_expert),
                        SOURCE_INDEX_FEATURE: source_index,
                    }
                )
                dst_meta.save_episode(
                    episode_index=int(new_episode_index),
                    episode_length=int(len(episode_df)),
                    episode_tasks=list(source_episode_row["tasks"]),
                    episode_stats=episode_stats,
                    episode_metadata=episode_metadata,
                )

        total_episodes += dataset.meta.total_episodes
        total_frames += dataset.meta.total_frames

    dst_meta._close_writer()
    logging.info("Merged dataset saved to %s", output_dir)
    logging.info("Episodes: %s", dst_meta.info["total_episodes"])
    logging.info("Frames: %s", dst_meta.info["total_frames"])
    logging.info("Features: %s", list(dst_meta.features.keys()))

    if cfg.push_to_hub:
        logging.info("Pushing merged dataset to hub as %s", cfg.repo_id)
        LeRobotDataset(repo_id=cfg.repo_id, root=output_dir).push_to_hub()

    return output_dir


@parser.wrap()
def main(cfg: MergeRolloutConfig) -> None:
    init_logging()
    merge_rollout_datasets(cfg)


if __name__ == "__main__":
    main()
