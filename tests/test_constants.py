"""constants.py: the class list and the COCO -> VOC mapping."""

from vocdet.constants import COCO_TO_VOC, VOC_CLASSES

# The 80 names from Ultralytics' coco.yaml (copied 2026-10-28), so the test needs no ultralytics.
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def test_twenty_unique_alphabetical_classes() -> None:
    assert len(VOC_CLASSES) == 20
    assert len(set(VOC_CLASSES)) == 20
    assert list(VOC_CLASSES) == sorted(VOC_CLASSES)


def test_mapping_covers_every_voc_class_exactly_once() -> None:
    assert sorted(COCO_TO_VOC.values()) == sorted(VOC_CLASSES)


def test_mapping_keys_are_real_coco_names() -> None:
    assert len(COCO_NAMES) == 80
    assert set(COCO_TO_VOC) <= set(COCO_NAMES)


def test_renamed_classes() -> None:
    assert COCO_TO_VOC["airplane"] == "aeroplane"
    assert COCO_TO_VOC["motorcycle"] == "motorbike"
    assert COCO_TO_VOC["couch"] == "sofa"
    assert COCO_TO_VOC["dining table"] == "diningtable"
    assert COCO_TO_VOC["potted plant"] == "pottedplant"
    assert COCO_TO_VOC["tv"] == "tvmonitor"
