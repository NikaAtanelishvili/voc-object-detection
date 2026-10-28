"""models/frcnn.py: the replaced head, inference settings, and a forward pass (no downloads)."""

import pytest
import torch

from vocdet.models.frcnn import NUM_CLASSES, build_frcnn, count_parameters


@pytest.fixture(scope="module")
def model():
    return build_frcnn(pretrained=False, min_size=320, max_size=480)


def test_head_has_21_classes(model) -> None:
    predictor = model.roi_heads.box_predictor
    assert NUM_CLASSES == 21
    assert predictor.cls_score.out_features == 21
    assert predictor.bbox_pred.out_features == 21 * 4  # one box regression per class


def test_inference_settings(model) -> None:
    assert model.transform.min_size == (320,) and model.transform.max_size == 480
    assert model.roi_heads.score_thresh == 0.001
    assert model.roi_heads.detections_per_img == 100


def test_default_sizes_follow_plan() -> None:
    m = build_frcnn(pretrained=False)
    assert m.transform.min_size == (600,) and m.transform.max_size == 1000


def test_train_forward_returns_four_losses(model) -> None:
    model.train()
    images = [torch.rand(3, 120, 160)]
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 80.0, 100.0]]), "labels": torch.tensor([12])}]
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_eval_forward_returns_detections_in_input_pixels(model) -> None:
    model.eval()
    with torch.no_grad():
        (out,) = model([torch.rand(3, 120, 160)])
    assert set(out) == {"boxes", "labels", "scores"}
    if len(
        out["boxes"]
    ):  # random weights may detect nothing; when they do, boxes are in input pixels
        assert out["boxes"][:, 2].max() <= 160 and out["boxes"][:, 3].max() <= 120


def test_count_parameters(model) -> None:
    total, trainable = count_parameters(model)
    assert 0 < trainable <= total
