#!/usr/bin/env python

import torch

from lerobot.values.value01.processor_value01 import Value01PrepareImagesProcessorStep


def test_value01_prepare_images_resizes_mixed_camera_resolutions():
    step = Value01PrepareImagesProcessorStep(
        camera_features=[
            "observation.images.head",
            "observation.images.left",
            "observation.images.right",
        ]
    )

    images, image_attention_mask = step._prepare_images(
        {
            "observation.images.head": torch.rand(2, 3, 96, 128),
            "observation.images.left": torch.rand(2, 3, 48, 64),
            "observation.images.right": torch.rand(2, 48, 64, 3),
        }
    )

    assert images.shape == (2, 3, 3, 96, 128)
    assert images.dtype == torch.float32
    assert torch.equal(image_attention_mask, torch.ones(2, 3, dtype=torch.bool))
