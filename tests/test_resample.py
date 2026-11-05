"""resample.py: fast resample scoring equals evaluate() on the explicitly resampled set.

evaluate() returns torchmetrics' float32 values and the fast path pycocotools' float64 ones, so
they are compared to float32 precision.
"""

import numpy as np
import pytest

from vocdet.eval.evaluator import evaluate
from vocdet.eval.resample import MatchedImages
from vocdet.eval.schema import GroundTruth, Prediction

FLOAT32_TOL = 1e-6


def random_dataset(seed: int, num_images: int = 12) -> tuple[list[GroundTruth], list[Prediction]]:
    """Images with 0-3 objects of 3 classes (some difficult) and noisy, partly wrong detections."""
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    for i in range(num_images):
        n = int(rng.integers(0, 4))
        xy = rng.uniform(0, 300, size=(n, 2))
        boxes = np.hstack([xy, xy + rng.uniform(20, 120, size=(n, 2))]).astype(np.float32)
        labels = rng.integers(0, 3, size=n)
        gts.append(GroundTruth(f"img{i}", boxes, labels, rng.random(n) < 0.2))
        jitter = rng.normal(0, 8, size=boxes.shape).astype(np.float32)
        extra = int(rng.integers(0, 3))
        exy = rng.uniform(0, 300, size=(extra, 2))
        fp = np.hstack([exy, exy + 40]).astype(np.float32)
        flipped = np.where(rng.random(n) < 0.2, (labels + 1) % 3, labels)
        preds.append(
            Prediction(
                f"img{i}",
                np.vstack([boxes + jitter, fp]),
                rng.uniform(0.01, 1, size=n + extra).astype(np.float32),
                np.concatenate([flipped, rng.integers(0, 3, size=extra)]),
            )
        )
    return gts, preds


def resampled(gts, preds, picks):
    """The slow way: an explicit resample, duplicates renamed, as the evaluator requires."""
    g = [
        GroundTruth(f"{gts[i].image_id}#{k}", gts[i].boxes, gts[i].labels, gts[i].difficult)
        for k, i in enumerate(picks)
    ]
    p = [
        Prediction(f"{preds[i].image_id}#{k}", preds[i].boxes, preds[i].scores, preds[i].labels)
        for k, i in enumerate(picks)
    ]
    return p, g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_scores_equal_the_evaluator_on_explicit_resamples(seed: int) -> None:
    gts, preds = random_dataset(seed)
    matched = MatchedImages(preds, gts)
    rng = np.random.default_rng(100 + seed)
    for _ in range(4):
        picks = rng.integers(0, len(gts), size=len(gts))
        slow = evaluate(*resampled(gts, preds, picks))
        fast = matched.score(picks)
        assert fast["map50"] == pytest.approx(slow["map50"], abs=FLOAT32_TOL)
        assert fast["map50_95"] == pytest.approx(slow["map50_95"], abs=FLOAT32_TOL)


def test_identity_resample_equals_the_full_evaluation() -> None:
    gts, preds = random_dataset(5)
    full = evaluate(preds, gts)
    fast = MatchedImages(preds, gts).score(np.arange(len(gts)))
    assert fast["map50"] == pytest.approx(full["map50"], abs=FLOAT32_TOL)
    assert fast["map50_95"] == pytest.approx(full["map50_95"], abs=FLOAT32_TOL)
