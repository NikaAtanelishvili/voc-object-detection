"""visualize.py: boxes are drawn where expected; galleries contain the right panels in order."""

from pathlib import Path

from PIL import Image

from vocdet.constants import VOC_CLASSES
from vocdet.eval.error_analysis import analyze
from vocdet.eval.schema import GroundTruth, Prediction
from vocdet.eval.visualize import (
    FP_COLOR,
    THUMB_SIZE,
    detection_texts,
    draw_boxes,
    failure_gallery,
)

CAT, DOG = VOC_CLASSES.index("cat"), VOC_CLASSES.index("dog")


def test_draw_boxes_outlines_box_and_keeps_original() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    out = draw_boxes(image, [[10, 30, 50, 80]], [""], (255, 0, 0), width=1)
    assert out.getpixel((10, 50)) == (255, 0, 0)  # left edge
    assert out.getpixel((30, 50)) == (255, 255, 255)  # inside stays untouched
    assert image.getpixel((10, 50)) == (255, 255, 255)  # input image not modified


def test_detection_texts() -> None:
    assert detection_texts([CAT], [0.876]) == ["cat 0.88"]
    assert detection_texts([DOG]) == ["dog"]


def test_failure_gallery(tmp_path: Path) -> None:
    for image_id in ("a", "b"):
        Image.new("RGB", (400, 300), (200, 200, 200)).save(tmp_path / f"{image_id}.jpg")
    gts = [
        GroundTruth("a", [[0, 0, 100, 100]], [CAT], [False]),
        GroundTruth("b", [[0, 0, 200, 200], [300, 0, 320, 20]], [DOG, DOG], [False, False]),
    ]
    preds = [
        # Two background false positives predicted as cat, with different scores.
        Prediction("a", [[0, 0, 100, 100], [300, 200, 350, 250]], [0.9, 0.6], [CAT, CAT]),
        Prediction("b", [[300, 200, 390, 290]], [0.8], [CAT]),
    ]
    result = analyze(preds, gts, score_threshold=0.5)
    written = failure_gallery(result, gts, tmp_path, tmp_path / "gallery", k=1)
    names = sorted(p.name for p in written)
    assert names == ["fp_cat.jpg", "miss_dog.jpg"]  # the cat TP produces no gallery

    strip = Image.open(tmp_path / "gallery" / "fp_cat.jpg")
    assert strip.width == THUMB_SIZE  # k=1: a single panel, the 0.8 FP from image b
    # The 400x300 image is scaled to 320x240, so the FP box (300..390) is drawn near x = 240..312.
    r, g, b = strip.getpixel((241, 200))
    assert r > 150 and g < 100 and b < 100, FP_COLOR

    misses = Image.open(tmp_path / "gallery" / "miss_dog.jpg")
    assert misses.width == THUMB_SIZE  # k=1 keeps only the largest missed dog
