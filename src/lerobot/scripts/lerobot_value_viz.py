#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Visualize value/advantage/indicator annotations already written to a dataset.

Skips inference entirely — the dataset must already contain the annotated columns.

Usage:
    python -m lerobot.scripts.lerobot_value_viz \
        --dataset-root /path/to/dataset \
        --repo-id my_org/my_dataset \
        --output-dir /path/to/output_videos \
        [--episodes all] \
        [--video-key observation.images.front] \
        [--value-field complementary_info.value] \
        [--advantage-field complementary_info.advantage] \
        [--enable-advantage-threshold-marker] \
        [--advantage-threshold 0.0] \
        [--indicator-field complementary_info.acp_indicator] \
        [--vcodec libsvtav1] \
        [--smooth-window 1] \
        [--overwrite]
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.value_infer_viz import _export_overlay_videos
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize value/advantage/indicator annotations in a dataset (no inference)."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local root directory of the dataset.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g. 'my_org/my_dataset').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output MP4 videos will be saved.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episodes to visualize: 'all', comma-separated indices, or ranges like '0-5,10'. Default: all.",
    )
    parser.add_argument(
        "--video-key",
        default=None,
        help="Single camera key to use for visualization. Auto-selected if not specified.",
    )
    parser.add_argument(
        "--video-keys",
        default=None,
        help="Comma-separated camera keys for multi-view visualization.",
    )
    parser.add_argument(
        "--value-field",
        default="complementary_info.value",
        help="Dataset column name for value scores. Default: complementary_info.value",
    )
    parser.add_argument(
        "--advantage-field",
        default="complementary_info.advantage",
        help=(
            "Dataset column name for advantage scores. Also used by the threshold marker when enabled. "
            "Default: complementary_info.advantage"
        ),
    )
    parser.add_argument(
        "--enable-advantage-threshold-marker",
        action="store_true",
        help=(
            "Draw a red vertical marker line for frames whose advantage is greater than "
            "--advantage-threshold."
        ),
    )
    parser.add_argument(
        "--advantage-threshold",
        type=float,
        default=None,
        help="Threshold used by --enable-advantage-threshold-marker. Required when the marker is enabled.",
    )
    parser.add_argument(
        "--indicator-field",
        default="complementary_info.acp_indicator",
        help="Dataset column name for binary ACP indicators. Default: complementary_info.acp_indicator",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec for encoding output videos. Default: libsvtav1",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Savitzky-Golay smoothing window for value curve. Use 1 to disable. Default: 1",
    )
    parser.add_argument(
        "--frame-storage-mode",
        default="memory",
        choices=["memory", "disk"],
        help="Frame storage mode during encoding. Default: memory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.enable_advantage_threshold_marker and args.advantage_threshold is None:
        raise ValueError(
            "--advantage-threshold is required when --enable-advantage-threshold-marker is set."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_logging(log_file=None)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    logging.info("Loading dataset from root=%s repo_id=%s", args.dataset_root, args.repo_id)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        download_videos=False,
    )

    raw_frames = dataset.hf_dataset.with_format(None)
    required_fields = [args.value_field]
    if args.enable_advantage_threshold_marker:
        required_fields.append(args.advantage_field)

    for field in required_fields:
        if field not in raw_frames.column_names:
            raise KeyError(
                f"Required field '{field}' not found in dataset columns: {raw_frames.column_names}\n"
                "Make sure you have already run value inference on this dataset."
            )

    logging.info(
        "Starting visualization | episodes=%s value=%s advantage=%s indicator=%s",
        args.episodes,
        args.value_field,
        args.advantage_field,
        args.indicator_field,
    )

    written = _export_overlay_videos(
        dataset=dataset,
        value_field=args.value_field,
        advantage_field=args.advantage_field,
        indicator_field=args.indicator_field,
        viz_episodes=args.episodes,
        video_key=args.video_key,
        video_keys=args.video_keys,
        output_dir=output_dir,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
        frame_storage_mode=args.frame_storage_mode,
        smooth_window=args.smooth_window,
        draw_advantage_threshold_marker=args.enable_advantage_threshold_marker,
        advantage_threshold=args.advantage_threshold,
    )

    logging.info("Done. Wrote %d video(s) to %s", len(written), output_dir)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
