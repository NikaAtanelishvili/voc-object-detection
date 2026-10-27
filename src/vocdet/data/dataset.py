"""torch Dataset that feeds Faster R-CNN from the canonical annotation table.

Targets follow torchvision's detection API: boxes as float32 [N, 4] 0-based xyxy, labels as int64.
torchvision reserves label 0 for background, so labels are VOC class_id + LABEL_OFFSET (1..20);
predictions are shifted back by the same constant (P3.3). Difficult objects are excluded from
training targets (PLAN.md §3).
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from vocdet.data.transforms import random_hflip

LABEL_OFFSET = 1  # torchvision detection models: 0 = background, so VOC class k becomes label k + 1


class VOCDetectionDataset(Dataset):
    """One item = (image float CHW in [0, 1], {"boxes", "labels", "image_id"})."""

    def __init__(
        self, annotations: pd.DataFrame, image_ids: list[str], image_dir: Path, train: bool
    ) -> None:
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.train = train
        # Only this split's non-difficult objects: a train dataset never holds val/test targets.
        in_split = annotations["image_id"].isin(set(self.image_ids))
        kept = annotations[in_split & ~annotations["difficult"]]
        # Precompute per-image tensors once; __getitem__ then only decodes the JPEG.
        self.targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for image_id, group in kept.groupby("image_id"):
            boxes = torch.tensor(group[["x1", "y1", "x2", "y2"]].to_numpy(), dtype=torch.float32)
            labels = torch.tensor(group["class_id"].to_numpy() + LABEL_OFFSET, dtype=torch.int64)
            self.targets[image_id] = (boxes, labels)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        image_id = self.image_ids[index]
        path = self.image_dir / f"{image_id}.jpg"
        image = decode_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
        # An image whose objects are all difficult has no targets: an empty [0, 4] box tensor is
        # valid input for torchvision detection models (it is treated as pure background).
        boxes, labels = self.targets.get(
            image_id, (torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64))
        )
        if self.train:
            image, boxes = random_hflip(image, boxes)
        return image, {"boxes": boxes, "labels": labels, "image_id": image_id}


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    """Detection models take lists, not stacked tensors, because images differ in size."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
