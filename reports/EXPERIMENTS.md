# Experiments

All runs use the frozen splits and the one evaluator (ADR-008). Metrics are on `val`. The table below is
generated from `results/experiments.csv` by `scripts/build_report_tables.py` (`make report`). Do not edit
it by hand. Rows with Exp `E0a_frcnn*` are pipeline and timing checks (64 val images, a few training
steps), not models.

<!-- BEGIN:experiments -->
| Run                                     | Exp               |   mAP50 |   mAP50-95 |   AP small |   AP medium |   AP large |   AR@100 |   Wall time (s) |
|:----------------------------------------|:------------------|--------:|-----------:|-----------:|------------:|-----------:|---------:|----------------:|
| E0a_frcnn_20261029-0548_0256862         | E0a_frcnn         |   0.050 |      0.026 |      0.000 |       0.038 |      0.032 |    0.366 |          57.100 |
| E0a_frcnn_timing_20261029-0552_2dd6526  | E0a_frcnn_timing  |   0.424 |      0.217 |      0.071 |       0.352 |      0.206 |    0.438 |         212.900 |
| E0a_yolo_20261029-1534_6a64129          | E0a_yolo          |   0.652 |      0.546 |      0.277 |       0.510 |      0.571 |    0.712 |          91.100 |
| E0a_yolo_timing_n_20261029-1539_860a6b4 | E0a_yolo_timing_n |   0.809 |      0.637 |      0.291 |       0.476 |      0.707 |    0.775 |        1505.600 |
| E0a_yolo_timing_s_20261029-1749_631ad4c | E0a_yolo_timing_s |   0.858 |      0.689 |      0.407 |       0.507 |      0.764 |    0.803 |         365.400 |
| E0b_20261028-0321_fa7911f               | E0b               |   0.827 |      0.649 |      0.292 |       0.493 |      0.719 |    0.765 |          50.500 |
| E1_20261029-0602_da4e489                | E1                |   0.795 |      0.548 |      0.131 |       0.354 |      0.638 |    0.640 |        2946.600 |
| E2_20261029-1808_ae67b2f                | E2                |   0.812 |      0.643 |      0.300 |       0.478 |      0.718 |    0.754 |        3243.700 |
<!-- END:experiments -->

## E1: Faster R-CNN MobileNetV3-L FPN baseline (Phase 3)

Sources:
- run `E1_20261029-0602_da4e489`: `results/runs/E1_20261029-0602_da4e489/`, with metrics, config, run
  metadata, and per-step `train_log.csv`;
- the val curve in the run's `val_log.csv`, kept on Kaggle and in `runs/`;
- TIDE-lite errors at score ≥ 0.5 in `error_analysis/errors.json`.

**Observations.**
- **Training curve.** Validation mAP50 is flat, and even dips slightly, during the constant-LR epochs. It
  jumps after the first LR drop at epoch 8 and improves a little more after the second. The best
  checkpoint is the last epoch. This suggests the schedule's decay points matter more than more epochs at
  the base LR. A longer run would most likely need its milestones moved, not just more epochs added.
- **Object size.** Accuracy depends strongly on size: AP on small objects is a fraction of AP on large
  ones (table). This is expected from a MobileNet backbone at a 600-pixel short side.
- **Error types.** At the error-analysis score threshold, missed objects outnumber every false-positive
  type combined. Among false positives, localisation errors are the largest group, followed by background
  detections; class confusions are rarer.
- **Weakest classes.** The weakest per-class AP50 values (in `metrics.json`) are pottedplant, diningtable,
  bottle and chair: small, cluttered, or heavily occluded objects. Animals and vehicles are the strongest.
- **Against E0b.** E1 is fine-tuned on VOC train, yet it scores *below* the zero-shot COCO YOLO26n (E0b)
  on val mAP50 and mAP50-95. The comparison is not like for like. E0b's weights saw far more images of the
  same categories in COCO, and YOLO26n is a newer one-stage design. Still, it sets the bar for Phase 4: a
  fine-tuned YOLO26 has to beat its own zero-shot score to show that fine-tuning adds value.
- **Training dynamics.** The AMP gradient scaler skipped two steps with non-finite gradients, at the very
  start and once mid-run (`grad_norm` = inf in `train_log.csv`). Training was otherwise stable, with no
  NaN losses. Data loading was never the bottleneck (`data_s`).
