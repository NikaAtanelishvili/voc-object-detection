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
| E3_20261029-1933_dc8cbeb                | E3                |   0.819 |      0.659 |      0.333 |       0.512 |      0.731 |    0.757 |        4263.500 |
| E4_20261031-2228_654138d                | E4                |   0.763 |      0.590 |      0.223 |       0.478 |      0.661 |    0.752 |        1508.100 |
| E5_20261101-1538_2c9ce61                | E5                |   0.816 |      0.642 |      0.286 |       0.505 |      0.711 |    0.756 |        2780.900 |
| E6n_20261105-2334_9f2f939               | E6n               |   0.835 |      0.670 |      0.301 |       0.501 |      0.742 |    0.781 |         927.500 |
| E6s_20261102-1659_3f5e706               | E6s               |   0.869 |      0.715 |      0.382 |       0.555 |      0.788 |    0.814 |        1346.400 |
<!-- END:experiments -->

## Comparing runs

Runs are compared only through a paired bootstrap of their difference (ADR-014): every run is re-scored on
the same 200 image resamples of val, and the table gives the 95% CI of the per-resample difference.
Overlap of two separate CIs is not used, because two models on the same images share most of their noise.
Generated from `results/paired_deltas.csv` by `make report`.

<!-- BEGIN:deltas -->
| Comparison              | Metric   |      Δ |   CI95 low |   CI95 high | CI excludes 0   |
|:------------------------|:---------|-------:|-----------:|------------:|:----------------|
| E2 − E0b                | map50    | -0.015 |     -0.032 |      +0.006 | no              |
| E2 − E0b                | map50_95 | -0.006 |     -0.025 |      +0.013 | no              |
| E3 − E0b                | map50    | -0.009 |     -0.033 |      +0.012 | no              |
| E3 − E0b                | map50_95 | +0.009 |     -0.013 |      +0.031 | no              |
| E2 − E1                 | map50    | +0.017 |     -0.005 |      +0.038 | no              |
| E2 − E1                 | map50_95 | +0.096 |     +0.075 |      +0.115 | yes             |
| E3 − E2                 | map50    | +0.007 |     -0.013 |      +0.025 | no              |
| E3 − E2                 | map50_95 | +0.015 |     -0.004 |      +0.032 | no              |
| E4 − E2                 | map50    | -0.049 |     -0.078 |      -0.021 | yes             |
| E4 − E2                 | map50_95 | -0.053 |     -0.076 |      -0.027 | yes             |
| E4 − E0b                | map50    | -0.065 |     -0.086 |      -0.038 | yes             |
| E4 − E0b                | map50_95 | -0.059 |     -0.080 |      -0.036 | yes             |
| E5 − E2                 | map50    | +0.004 |     -0.013 |      +0.022 | no              |
| E5 − E2                 | map50_95 | -0.001 |     -0.018 |      +0.014 | no              |
| E5 − E0b                | map50    | -0.011 |     -0.031 |      +0.012 | no              |
| E5 − E0b                | map50_95 | -0.007 |     -0.025 |      +0.013 | no              |
| E6s − E3                | map50    | +0.050 |     +0.035 |      +0.072 | yes             |
| E6s − E3                | map50_95 | +0.056 |     +0.040 |      +0.074 | yes             |
| E6s − E0b               | map50    | +0.042 |     +0.023 |      +0.062 | yes             |
| E6s − E0b               | map50_95 | +0.065 |     +0.049 |      +0.086 | yes             |
| E6s − E2                | map50    | +0.057 |     +0.040 |      +0.077 | yes             |
| E6s − E2                | map50_95 | +0.071 |     +0.057 |      +0.089 | yes             |
| E6n − E2                | map50    | +0.023 |     +0.005 |      +0.038 | yes             |
| E6n − E2                | map50_95 | +0.026 |     +0.012 |      +0.038 | yes             |
| E6n − E0b               | map50    | +0.007 |     -0.006 |      +0.022 | no              |
| E6n − E0b               | map50_95 | +0.020 |     +0.004 |      +0.038 | yes             |
| E6s − E6n               | map50    | +0.034 |     +0.019 |      +0.052 | yes             |
| E6s − E6n               | map50_95 | +0.045 |     +0.031 |      +0.060 | yes             |
| E0a_yolo_timing_s − E3  | map50    | +0.039 |     +0.021 |      +0.063 | yes             |
| E0a_yolo_timing_s − E3  | map50_95 | +0.030 |     +0.011 |      +0.052 | yes             |
| E0a_yolo_timing_s − E0b | map50    | +0.031 |     +0.010 |      +0.050 | yes             |
| E0a_yolo_timing_s − E0b | map50_95 | +0.039 |     +0.020 |      +0.058 | yes             |
| E0a_yolo_timing_n − E2  | map50    | -0.003 |     -0.026 |      +0.018 | no              |
| E0a_yolo_timing_n − E2  | map50_95 | -0.006 |     -0.024 |      +0.012 | no              |
<!-- END:deltas -->

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

## Phase 4: YOLO26 experiments (E2–E6)

Sources:
- runs under `results/runs/<run_id>/` (config, metadata, Ultralytics `train_log.csv`, TIDE-lite errors);
- paired differences in the table above (`results/paired_deltas.csv`, ADR-014);
- head and class-order checks in `results/diagnostics/finetune_vs_zeroshot.json` (ADR-015);
- the Ultralytics-vs-ours gap breakdown in `results/diagnostics/crosscheck_gap_*.json` (ADR-013).

Every YOLO run starts from the COCO checkpoint. Ultralytics copies the COCO classification-head rows of the
14 classes whose names match exactly; E6 copies all 20.

**Observations.**
- **Standard fine-tuning did not beat zero-shot.** E2 (YOLO26n) and E3 (YOLO26s), 50 epochs with the
  default recipe, have paired CIs against the zero-shot E0b that include zero on both metrics. E3 is not
  shown to beat E2 either. The 6 classes with fresh head rows lose more AP50 against E0b than the 14
  copied ones.
- **Fine-tuning still beats the two-stage baseline** on mAP50-95: E2 over E1 is the clearest
  difference among the planned runs. On mAP50 the CI includes zero.
- **Augmentation matters (E4).** Without mosaic, mixup, colour, scale and translate augmentation, YOLO26n
  peaked in its first epoch and then early-stopped, and it is worse than both E2 and E0b, with CIs
  excluding zero.
- **A frozen backbone costs nothing measurable (E5).** Freezing the first 10 layers gives a model not
  shown to differ from E2 or E0b, while training fewer parameters.
- **Gentle fine-tuning works (E6).** Copying all 20 COCO head rows and training 8 epochs at a low LR, ending
  without mosaic, gives the best model of the project (E6s). It beats E0b, E2, E3 and E6n with CIs
  excluding zero on both metrics. E6n beats E2 on both metrics, and E0b on mAP50-95.
  - This supports the forgetting hypothesis in ADR-015: long training at the default LR moves the weights
    away from a COCO fit that already suits these classes.
  - E6 was designed after seeing val results (the 3-epoch timing run), so its val score is optimistic as a
    model-selection number. The test-split evaluation in P5.2 decides (ADR-003).
- **Caveats.** One seed per configuration: the CIs cover image sampling, not training randomness. E6n ran
  on Kaggle's newer image (CUDA 13.0 build of the same torch version), all other runs on the older one.
