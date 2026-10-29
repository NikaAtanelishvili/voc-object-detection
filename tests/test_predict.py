"""predict.py: class mapping, VOC conversion, and the YOLO adapter with a fake model."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vocdet.constants import COCO_TO_VOC, VOC_CLASSES
from vocdet.eval.predict import build_class_map, predict_yolo, to_voc_prediction

# Model-index layout of a COCO model, reduced: the 20 VOC-relevant names plus two others.
COCO_NAMES = {i: name for i, name in enumerate(["truck", *COCO_TO_VOC, "banana"])}


def test_class_map_for_coco_and_voc_models() -> None:
    coco_map = build_class_map(COCO_NAMES)
    assert 0 not in coco_map and 21 not in coco_map  # truck, banana dropped
    airplane = next(i for i, n in COCO_NAMES.items() if n == "airplane")
    assert coco_map[airplane] == VOC_CLASSES.index("aeroplane")
    voc_map = build_class_map(dict(enumerate(VOC_CLASSES)))
    assert voc_map == {i: i for i in range(20)}


def test_class_map_must_cover_all_voc_classes() -> None:
    with pytest.raises(ValueError, match="cover all 20"):
        build_class_map({0: "cat", 1: "dog"})


def test_to_voc_prediction_drops_maps_and_keeps_top_k() -> None:
    class_map = {5: VOC_CLASSES.index("cat"), 6: VOC_CLASSES.index("dog")}
    boxes = np.array([[0, 0, 10, 10], [1, 1, 5, 5], [2, 2, 8, 8], [3, 3, 9, 9]], dtype=np.float32)
    scores = np.array([0.2, 0.9, 0.95, 0.5], dtype=np.float32)
    labels = np.array([5, 99, 6, 5])  # 99 has no VOC class
    p = to_voc_prediction("img", boxes, scores, labels, class_map, max_detections=2)
    assert p.scores.tolist() == pytest.approx([0.95, 0.5])  # 0.9 was class 99, dropped first
    assert [VOC_CLASSES[label] for label in p.labels] == ["dog", "cat"]
    np.testing.assert_array_equal(p.boxes, boxes[[2, 3]])


class FakeYOLO:
    """Mimics the parts of ultralytics.YOLO that predict_yolo uses."""

    names = dict(enumerate(VOC_CLASSES))

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, sources: list[str], **kwargs) -> list[SimpleNamespace]:
        self.calls.append({"n": len(sources), **kwargs})
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
            conf=torch.tensor([0.8]),
            cls=torch.tensor([7.0]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in sources]


def test_predict_yolo_batches_and_converts() -> None:
    model = FakeYOLO()
    paths = [Path(f"/x/2008_{i:06d}.jpg") for i in range(5)]
    preds = predict_yolo(model, paths, imgsz=512, batch=2)
    assert [c["n"] for c in model.calls] == [2, 2, 1]
    assert model.calls[0]["conf"] == 0.001 and model.calls[0]["imgsz"] == 512
    assert [p.image_id for p in preds] == [p.stem for p in paths]
    assert preds[0].labels.tolist() == [7] and preds[0].boxes.tolist() == [[1.0, 2.0, 30.0, 40.0]]


def test_frcnn_label_offset_round_trip() -> None:
    """Dataset adds LABEL_OFFSET for torchvision; the adapter must subtract exactly that."""
    from vocdet.data.dataset import LABEL_OFFSET
    from vocdet.eval.predict import frcnn_output_to_prediction

    voc_labels = np.array([0, VOC_CLASSES.index("person"), 19])
    torchvision_labels = torch.from_numpy(voc_labels + LABEL_OFFSET)  # what the model is trained on
    output = {
        "boxes": torch.tensor([[0.0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 7, 7]]),
        "scores": torch.tensor([0.9, 0.5, 0.01]),
        "labels": torchvision_labels,
    }
    p = frcnn_output_to_prediction("img", output)
    assert p.labels.tolist() == voc_labels.tolist()  # 1..20 -> 0..19
    assert p.scores.tolist() == pytest.approx([0.9, 0.5, 0.01])


def test_predict_frcnn_with_random_model(tmp_path: Path) -> None:
    from PIL import Image

    from vocdet.eval.predict import predict_frcnn
    from vocdet.models.frcnn import build_frcnn

    for i in ("a", "b", "c"):
        Image.new("RGB", (160, 120), (90, 120, 150)).save(tmp_path / f"{i}.jpg")
    model = build_frcnn(pretrained=False, min_size=128, max_size=160)
    preds = predict_frcnn(model, ["a", "b", "c"], tmp_path, torch.device("cpu"), batch_size=2)
    assert [p.image_id for p in preds] == ["a", "b", "c"]
    for p in preds:  # random weights: maybe no boxes, but any box obeys the schema
        assert len(p.boxes) <= 100
        assert ((p.labels >= 0) & (p.labels <= 19)).all()
