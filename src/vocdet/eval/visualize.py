"""Draw detections, and build failure galleries from an error analysis.

failure_gallery writes one JPEG strip per class and kind, made of up to k thumbnails:
- fp_<class>.jpg: the k highest-confidence false positives predicted as that class (red box, with
  the error type), plus the image's ground truth in green for context;
- miss_<class>.jpg: the k largest missed objects of that class (yellow box). Large misses are the
  surprising ones; small misses are expected.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from vocdet.constants import VOC_CLASSES
from vocdet.eval.error_analysis import FP_TYPES, ErrorAnalysis
from vocdet.eval.schema import GroundTruth

logger = logging.getLogger(__name__)

THUMB_SIZE = 320  # longest side of each gallery panel, keeps a 5-panel strip around 100 KB
GT_COLOR, FP_COLOR, MISS_COLOR = (0, 200, 0), (230, 0, 0), (255, 200, 0)


def draw_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    texts: list[str],
    color: tuple[int, int, int],
    width: int = 3,
) -> Image.Image:
    """Return a copy of image with each box outlined and its text drawn on a filled tag above it."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2), text in zip(boxes, texts, strict=True):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if text:
            left, top, right, bottom = draw.textbbox((x1, y1), text)
            tag_top = max(0, y1 - (bottom - top) - 4)
            draw.rectangle(
                [x1, tag_top, x1 + (right - left) + 4, tag_top + (bottom - top) + 4], fill=color
            )
            draw.text((x1 + 2, tag_top + 2), text, fill=(0, 0, 0))
    return out


def detection_texts(labels: list[int], scores: list[float] | None = None) -> list[str]:
    """'dog 0.87' style captions; without scores just the class name."""
    if scores is None:
        return [VOC_CLASSES[label] for label in labels]
    return [
        f"{VOC_CLASSES[label]} {score:.2f}" for label, score in zip(labels, scores, strict=True)
    ]


def _strip(panels: list[Image.Image]) -> Image.Image:
    """Paste thumbnails side by side on a white canvas."""
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas


def _panel(
    image_path: Path, gt: GroundTruth | None, box: list[float], text: str, color
) -> Image.Image:
    image = Image.open(image_path)
    if gt is not None and len(gt.boxes):
        image = draw_boxes(image, gt.boxes.tolist(), [""] * len(gt.boxes), GT_COLOR, width=2)
    image = draw_boxes(image, [box], [text], color)
    image.thumbnail((THUMB_SIZE, THUMB_SIZE))
    return image


def failure_gallery(
    result: ErrorAnalysis,
    ground_truth: list[GroundTruth],
    image_dir: Path,
    out_dir: Path,
    k: int = 5,
) -> list[Path]:
    """Write fp_<class>.jpg and miss_<class>.jpg strips; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_id = {g.image_id: g for g in ground_truth}
    written = []
    for class_id, name in enumerate(VOC_CLASSES):
        fps = [r for r in result.detections if r.label == class_id and r.outcome in FP_TYPES]
        fps = sorted(fps, key=lambda r: -r.score)[:k]
        if fps:
            panels = [
                _panel(
                    image_dir / f"{r.image_id}.jpg",
                    gt_by_id.get(r.image_id),
                    r.box,
                    f"{name} {r.score:.2f} {r.outcome}",
                    FP_COLOR,
                )
                for r in fps
            ]
            path = out_dir / f"fp_{name}.jpg"
            _strip(panels).save(path, quality=80)
            written.append(path)

        misses = sorted((m for m in result.misses if m.label == class_id), key=lambda m: -m.area)[
            :k
        ]
        if misses:
            panels = [
                _panel(image_dir / f"{m.image_id}.jpg", None, m.box, f"missed {name}", MISS_COLOR)
                for m in misses
            ]
            path = out_dir / f"miss_{name}.jpg"
            _strip(panels).save(path, quality=80)
            written.append(path)
    logger.info("Wrote %d gallery strips to %s", len(written), out_dir)
    return written
