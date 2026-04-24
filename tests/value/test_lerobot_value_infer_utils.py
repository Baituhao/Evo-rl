#!/usr/bin/env python

import numpy as np

from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_advantages,
    _compute_rewards,
    _compute_task_thresholds,
    _compute_value_targets,
)
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import EpisodeTargetInfo
from lerobot.values.value01.configuration_value01 import Value01Config


def test_compute_rewards_pistar06_terminal_handling():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    targets = np.array([-0.6, -0.4, -0.2, -0.8, -0.5], dtype=np.float32)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=True),
        1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
    }

    rewards = _compute_rewards(
        Pistar06Config(),
        targets,
        episode_indices,
        frame_indices,
        episode_info,
    )
    expected = np.array([-0.2, -0.2, -0.2, -0.3, -0.5], dtype=np.float32)
    assert np.allclose(rewards, expected)


def test_compute_advantages_pistar06_simple_case():
    rewards = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1], dtype=np.float32)
    episode_indices = np.array([0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=True),
    }

    advantages = _compute_advantages(
        Pistar06Config(),
        rewards,
        values,
        episode_indices,
        frame_indices,
        2,
        episode_info,
    )

    expected = np.array([0.0, -0.1, -0.1], dtype=np.float32)
    assert np.allclose(advantages, expected)


def test_compute_task_thresholds_and_binarize_with_interventions():
    task_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    advantages = np.array([-0.4, -0.1, 0.3, -0.2, 0.2], dtype=np.float32)
    interventions = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    expert_episode_mask = np.array([False, False, False, False, False], dtype=bool)

    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio=0.5)

    indicators = _binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=True,
        expert_episode_mask=expert_episode_mask,
        force_expert_episode_positive=True,
    )

    assert indicators.tolist() == [0, 1, 1, 0, 1]


def test_compute_value_targets_dispatches_by_value_type():
    episode_indices = np.array([0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=2, success=True),
        1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
    }
    task_max_lengths = {0: 4}

    pistar06_targets = _compute_value_targets(
        value_cfg=Pistar06Config(),
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=1.0,
    )
    value01_targets = _compute_value_targets(
        value_cfg=Value01Config(),
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=1.0,
    )

    expected_pistar06 = np.array([-1 / 8, 0.0, -5 / 8, -4 / 8], dtype=np.float32)
    expected_value01 = np.array([-0.5, 0.0, -0.5, -1.0], dtype=np.float32)

    assert np.allclose(pistar06_targets, expected_pistar06)
    assert np.allclose(value01_targets, expected_value01)


def test_compute_rewards_dispatches_by_value_type():
    targets = np.array([-0.6, -0.4, -0.8, -0.5], dtype=np.float32)
    episode_indices = np.array([0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=2, success=True),
        1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
    }

    pistar06_expected = np.array([-0.2, -0.4, -0.3, -0.5], dtype=np.float32)
    value01_expected = np.array([-0.2, -0.4, -0.3, 0.5], dtype=np.float32)

    pistar06_rewards = _compute_rewards(
        Pistar06Config(), targets, episode_indices, frame_indices, episode_info
    )
    value01_rewards = _compute_rewards(
        Value01Config(), targets, episode_indices, frame_indices, episode_info
    )

    assert np.allclose(pistar06_rewards, pistar06_expected)
    assert np.allclose(value01_rewards, value01_expected)


def test_compute_advantages_dispatches_by_value_type():
    rewards = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1], dtype=np.float32)
    episode_indices = np.array([0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=False),
    }
    pistar06_expected = np.array([0.0, -0.1, -0.1], dtype=np.float32)
    value01_expected = np.array([0.0, -1.1, -1.1], dtype=np.float32)

    pistar06_advantages = _compute_advantages(
        Pistar06Config(),
        rewards,
        values,
        episode_indices,
        frame_indices,
        2,
        episode_info,
    )
    value01_advantages = _compute_advantages(
        Value01Config(),
        rewards,
        values,
        episode_indices,
        frame_indices,
        2,
        episode_info,
    )

    assert np.allclose(pistar06_advantages, pistar06_expected)
    assert np.allclose(value01_advantages, value01_expected)
