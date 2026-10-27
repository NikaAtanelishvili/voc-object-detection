"""Project-wide constants: the VOC class list (index = class_id everywhere in the project)."""

# Alphabetical, index 0-19 (PLAN.md §3). Every label in the project is an index into this tuple.
VOC_CLASSES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# COCO class name (as in Ultralytics' coco.yaml, checked 2026-10-28) -> VOC class name. Keyed by
# name, not index, so the zero-shot adapter (E0b) can map whatever `model.names` the COCO checkpoint
# reports. The other 60 COCO classes have no VOC counterpart; their detections are dropped.
COCO_TO_VOC: dict[str, str] = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "person",
    "potted plant": "pottedplant",
    "sheep": "sheep",
    "couch": "sofa",
    "train": "train",
    "tv": "tvmonitor",
}
