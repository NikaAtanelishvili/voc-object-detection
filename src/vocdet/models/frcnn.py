"""Build the two-stage baseline: torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone.

Starts from the COCO-trained detector and swaps only the final box predictor for a 21-way one
(20 VOC classes + background). The backbone, FPN, RPN, and box head keep their COCO weights.
"""

import logging

from torch import nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vocdet.constants import VOC_CLASSES

logger = logging.getLogger(__name__)

NUM_CLASSES = len(VOC_CLASSES) + 1  # torchvision detection models reserve label 0 for background


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_frcnn(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    min_size: int = 600,
    max_size: int = 1000,
    score_threshold: float = 0.001,
    detections_per_image: int = 100,
) -> FasterRCNN:
    """Return a Faster R-CNN ready for VOC fine-tuning.

    - pretrained=True loads torchvision's COCO weights. False builds random weights and downloads
      nothing, which is what unit tests need.
    - min_size/max_size: images are resized inside the model so the short side is min_size, capped
      so the long side stays <= max_size (PLAN.md §5: 600/1000, not the COCO default 800/1333).
    - score_threshold/detections_per_image: torchvision keeps boxes with score >= 0.05 by default;
      mAP needs the low-confidence tail too (PLAN.md §4: 0.001, top 100).
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,  # the COCO weights already include the backbone
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_threshold,
        box_detections_per_img=detections_per_image,
    )
    # The COCO head predicts 91 classes; replace it with a fresh 21-way classifier + box regressor.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    total, trainable = count_parameters(model)
    logger.info(
        "Faster R-CNN MobileNetV3-L FPN (pretrained=%s): %.2fM params, %.2fM trainable",
        pretrained,
        total / 1e6,
        trainable / 1e6,
    )
    return model
