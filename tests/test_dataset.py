"""dataset.py and transforms.py: flip correctness, label offset, difficult exclusion, collate."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from vocdet.constants import VOC_CLASSES
from vocdet.data.dataset import LABEL_OFFSET, VOCDetectionDataset, collate_fn
from vocdet.data.parse_voc import parse_dataset
from vocdet.data.transforms import hflip

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
IDS = ["2008_000001", "2008_000002"]


@pytest.fixture
def dataset_parts(tmp_path: Path):
    """Real JPEGs with the fixture XML sizes (500x375 and 400x300)."""
    for image_id, size in zip(IDS, [(500, 375), (400, 300)], strict=True):
        Image.new("RGB", size, color=(255, 0, 0)).save(tmp_path / f"{image_id}.jpg")
    return parse_dataset(VOC_FIXTURE, IDS), tmp_path


def test_hflip_maps_box_edges() -> None:
    image = torch.zeros(3, 10, 100)
    boxes = torch.tensor([[0.0, 1.0, 30.0, 5.0], [90.0, 0.0, 100.0, 10.0]])
    _, flipped = hflip(image, boxes)
    assert flipped.tolist() == [[70.0, 1.0, 100.0, 5.0], [0.0, 0.0, 10.0, 10.0]]


def test_hflip_moves_pixels() -> None:
    image = torch.zeros(1, 1, 4)
    image[0, 0, 0] = 1.0  # left-most column
    flipped, _ = hflip(image, torch.zeros((0, 4)))
    assert flipped[0, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_hflip_twice_is_identity() -> None:
    image = torch.rand(3, 20, 37)
    boxes = torch.tensor([[1.5, 2.0, 30.25, 19.0], [0.0, 0.0, 37.0, 20.0]])
    image2, boxes2 = hflip(*hflip(image, boxes))
    assert torch.equal(image2, image)
    assert torch.allclose(boxes2, boxes)


def test_hflip_keeps_boxes_valid() -> None:
    image = torch.zeros(3, 50, 80)
    boxes = torch.tensor([[0.0, 0.0, 80.0, 50.0], [10.0, 5.0, 11.0, 6.0]])
    _, flipped = hflip(image, boxes)
    assert (flipped[:, 0] >= 0).all() and (flipped[:, 2] <= 80).all()
    assert (flipped[:, 0] < flipped[:, 2]).all()
    assert torch.allclose(flipped[:, 2] - flipped[:, 0], boxes[:, 2] - boxes[:, 0])


def test_labels_are_offset_and_difficult_excluded(dataset_parts) -> None:
    annotations, image_dir = dataset_parts
    ds = VOCDetectionDataset(annotations, IDS, image_dir, train=False)
    image, target = ds[0]
    assert image.shape == (3, 375, 500) and image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    # 2008_000001: dog + difficult bottle -> only the dog, with label class_id + 1.
    assert target["labels"].tolist() == [VOC_CLASSES.index("dog") + LABEL_OFFSET]
    assert target["boxes"].tolist() == [[0.0, 9.0, 500.0, 375.0]]
    assert target["image_id"] == "2008_000001"
    assert LABEL_OFFSET == 1
    assert (target["labels"] >= 1).all()  # 0 is background in torchvision


def test_image_with_only_difficult_objects_has_empty_targets(dataset_parts) -> None:
    annotations, image_dir = dataset_parts
    only_difficult = annotations[annotations["image_id"] == "2008_000001"].copy()
    only_difficult["difficult"] = True
    ds = VOCDetectionDataset(only_difficult, ["2008_000001"], image_dir, train=False)
    _, target = ds[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_train_mode_flips_with_seeded_rng(dataset_parts) -> None:
    annotations, image_dir = dataset_parts
    ds = VOCDetectionDataset(annotations, IDS, image_dir, train=True)
    seen = set()
    for seed in range(20):
        torch.manual_seed(seed)
        _, target = ds[1]
        seen.add(tuple(target["boxes"][0].tolist()))
    # Person box (99, 49, 200.5, 290) in a 400-wide image flips to (199.5, 49, 301, 290).
    assert seen == {(99.0, 49.0, 200.5, 290.0), (199.5, 49.0, 301.0, 290.0)}


def test_collate_returns_lists(dataset_parts) -> None:
    annotations, image_dir = dataset_parts
    ds = VOCDetectionDataset(annotations, IDS, image_dir, train=False)
    images, targets = collate_fn([ds[0], ds[1]])
    assert isinstance(images, list) and isinstance(targets, list)
    assert [t["image_id"] for t in targets] == IDS
    assert images[0].shape != images[1].shape  # different sizes are fine as a list


def test_targets_only_for_own_split(dataset_parts) -> None:
    annotations, image_dir = dataset_parts
    ds = VOCDetectionDataset(annotations, ["2008_000002"], image_dir, train=False)
    assert set(ds.targets) == {"2008_000002"}
