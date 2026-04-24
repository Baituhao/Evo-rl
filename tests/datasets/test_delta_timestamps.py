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
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
)
from tests.fixtures.constants import DUMMY_MOTOR_FEATURES


@pytest.fixture(scope="module")
def valid_delta_timestamps_factory():
    def _create_valid_delta_timestamps(
        fps: int = 30, keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)
    ) -> dict:
        delta_timestamps = {key: [i * (1 / fps) for i in range(*min_max_range)] for key in keys}
        return delta_timestamps

    return _create_valid_delta_timestamps


@pytest.fixture(scope="module")
def invalid_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_invalid_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just outside tolerance
        for key in keys:
            delta_timestamps[key][3] += tolerance_s * 1.1
        return delta_timestamps

    return _create_invalid_delta_timestamps


@pytest.fixture(scope="module")
def slightly_off_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_slightly_off_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just inside tolerance
        for key in delta_timestamps:
            delta_timestamps[key][3] += tolerance_s * 0.9
            delta_timestamps[key][-3] += tolerance_s * 0.9
        return delta_timestamps

    return _create_slightly_off_delta_timestamps


@pytest.fixture(scope="module")
def delta_indices_factory():
    def _delta_indices(keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)) -> dict:
        return {key: list(range(*min_max_range)) for key in keys}

    return _delta_indices


def test_check_delta_timestamps_valid(valid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    valid_delta_timestamps = valid_delta_timestamps_factory(fps)
    result = check_delta_timestamps(
        delta_timestamps=valid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_slightly_off(slightly_off_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    slightly_off_delta_timestamps = slightly_off_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=slightly_off_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_invalid(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    with pytest.raises(ValueError):
        check_delta_timestamps(
            delta_timestamps=invalid_delta_timestamps,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_delta_timestamps_invalid_no_exception(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=invalid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_delta_timestamps_empty():
    delta_timestamps = {}
    fps = 30
    tolerance_s = 1e-4
    result = check_delta_timestamps(
        delta_timestamps=delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_delta_indices(valid_delta_timestamps_factory, delta_indices_factory):
    fps = 50
    min_max_range = (-100, 100)
    delta_timestamps = valid_delta_timestamps_factory(fps, min_max_range=min_max_range)
    expected_delta_indices = delta_indices_factory(min_max_range=min_max_range)
    actual_delta_indices = get_delta_indices(delta_timestamps, fps)
    assert expected_delta_indices == actual_delta_indices


def test_resolve_delta_timestamps_uses_only_declared_observation_features():
    cfg = SimpleNamespace(
        input_features={
            "observation.images.head": object(),
            "observation.state": object(),
        },
        observation_delta_indices=[-1, 0, 1],
        action_delta_indices=None,
        reward_delta_indices=None,
    )
    ds_meta = SimpleNamespace(
        fps=30,
        features={
            "observation.images.head": {},
            "observation.images.left": {},
            "observation.images.right": {},
            "observation.state": {},
            "action": {},
        },
    )

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    assert delta_timestamps == {
        "observation.images.head": [-1 / 30, 0.0, 1 / 30],
        "observation.state": [-1 / 30, 0.0, 1 / 30],
    }


def test_resolve_delta_timestamps_falls_back_to_all_observation_features_when_names_do_not_match():
    cfg = SimpleNamespace(
        input_features={
            "observation.images.camera1": object(),
            "observation.state": object(),
        },
        observation_delta_indices=[0],
        action_delta_indices=None,
        reward_delta_indices=None,
    )
    ds_meta = SimpleNamespace(
        fps=30,
        features={
            "observation.images.top": {},
            "observation.images.side": {},
            "observation.state": {},
            "action": {},
        },
    )

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    assert delta_timestamps == {
        "observation.images.top": [0.0],
        "observation.images.side": [0.0],
        "observation.state": [0.0],
    }


def test_lerobot_dataset_query_timestamps_defaults_missing_video_keys_to_current_frame():
    class MockHFDataset:
        def __init__(self, timestamps):
            self.timestamps = timestamps

        def __getitem__(self, key):
            if isinstance(key, list):
                return {"timestamp": [self.timestamps[idx] for idx in key]}
            raise TypeError(f"Unsupported key type: {type(key)}")

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = SimpleNamespace(video_keys=["observation.images.head", "observation.images.left"])
    dataset._absolute_to_relative_idx = None
    dataset.hf_dataset = MockHFDataset([torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)])

    query_timestamps = dataset._get_query_timestamps(
        current_ts=0.0,
        query_indices={"observation.images.head": [0, 1], "observation.state": [0, 1]},
    )

    assert list(query_timestamps) == ["observation.images.head", "observation.images.left"]
    assert query_timestamps["observation.images.head"] == pytest.approx([0.1, 0.2])
    assert query_timestamps["observation.images.left"] == pytest.approx([0.0])
