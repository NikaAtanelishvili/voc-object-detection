"""Augmentation for the Faster R-CNN dataset: a hand-written horizontal flip.

Resizing and normalization are not done here: torchvision's Faster R-CNN does both inside the model
(GeneralizedRCNNTransform, min_size/max_size from the config), and rescales boxes back afterwards.
"""

import torch


def hflip(image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror a CHW image and its 0-based xyxy boxes left-right.

    A box spanning [x1, x2] in an image of width W spans [W - x2, W - x1] after the flip. In
    continuous 0-based coordinates there is no "- 1": the pixel column [0, 1] maps to [W - 1, W].
    """
    width = image.shape[-1]
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return image.flip(-1), flipped


def random_hflip(
    image: torch.Tensor, boxes: torch.Tensor, p: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip with probability p, using torch's RNG so `seed_everything` makes it reproducible."""
    if torch.rand(1).item() < p:
        return hflip(image, boxes)
    return image, boxes
