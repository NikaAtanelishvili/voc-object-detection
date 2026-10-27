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
