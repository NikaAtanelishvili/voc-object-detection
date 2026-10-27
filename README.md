# voc-object-detection

End-to-end 20-class object detection on **Pascal VOC 2012** with PyTorch. A two-stage detector
(torchvision Faster R-CNN, own training loop) is compared against one-stage Ultralytics YOLO26 models
through controlled ablations and one shared evaluator. The best model is optimized for **CPU inference**
(ONNX / OpenVINO FP32 / INT8) and served as a **FastAPI + Gradio** app on a free Hugging Face Space.
Training runs on Kaggle GPUs; all code is developed and tested locally.

## Live demo

TODO (Phase 7)

## Results

TODO (generated from `results/` by `scripts/build_report_tables.py`, Phase 5)

## Quick start

```bash
make setup   # uv sync + pre-commit install
make check   # format check + lint + tests
```

## Reproduce

TODO (exact commands per experiment, Phase 8)

## API usage

TODO (Phase 7)

## Project structure

TODO (Phase 8). See `PLAN.md` §2 for the target layout.

## Licenses

- Code: MIT (see `LICENSE`).
- Ultralytics is **AGPL-3.0**. That is fine for this open-source project, but a closed-source production
  service would need an Ultralytics enterprise license or a different one-stage detector.
- Pascal VOC data: subject to the VOC terms of use (Flickr image licenses). TODO: link (Phase 8).
