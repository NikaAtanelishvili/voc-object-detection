# Phase 3 defense notes: Faster R-CNN baseline

## 1. What was built

- `build_frcnn`: torchvision Faster R-CNN MobileNetV3-Large FPN with COCO weights. The box predictor is
  replaced by a 21-class head (20 VOC classes + background), and the resize limits come from config.
- `predict_frcnn`: the model's output becomes our `Prediction`, with label − 1, score ≥ 0.001, top 100.
- Our own training loop. SGD with linear warm-up and step decay, AMP, measured gradient norm, a CSV row
  per step (losses, LR, gradient norm, data wait, step time), validation through the shared evaluator,
  `best.pt` by val map50, and a full-state `last.pt` every epoch.
- Resume that also works across Kaggle sessions. A new kernel attaches the dead run's output, finds
  `last.pt` at any mount depth, and carries over `best.pt` and both logs, dropping the unfinished epoch's
  rows. This was tested on Kaggle by resuming the E0a run on CPU.
- A training-time estimator. On the T4, a step with a new padded batch shape costs seconds more than a
  repeat. Replaying the seeded shuffle predicts exactly which steps are slow, and that prices E1.
- E0a smoke and timing runs, and the E1 baseline, all run on Kaggle T4.

## 2. Key code paths

- `models/frcnn.py::build_frcnn`: swaps `roi_heads.box_predictor` for `FastRCNNPredictor(in_features, 21)`.
- `data/dataset.py::VOCDetectionDataset`: labels become `voc_idx + 1`, since torchvision reserves 0 for
  background.
- `eval/predict.py::predict_frcnn`: undoes the offset (`label − 1`), and drops the score floor and the
  top-100 cap.
- `train/train_frcnn.py::lr_at`: the whole LR schedule as one pure function of (step, epoch).
- `train/train_frcnn.py::train_one_epoch`: the step: set LR → forward (dict of 4 losses) → sum → scaled
  backward → `unscale_` → gradient norm → `scaler.step` → `scaler.update` → log row.
- `train/train_frcnn.py::start_log`: rebuilds a log from the source run up to the checkpoint's epoch.
- `train/train_frcnn.py::train_frcnn`: setup, resume, the epoch loop with periodic validation, and the
  final re-prediction from `best.pt`.
- `utils/kaggle_input.py::find_run_file`: the any-depth `/kaggle/input` search, shared by data and resume.
- `scripts/kaggle_run.py::kernel_slug`: a resume gets its own kernel (`-from-<timestamp>`).
- `scripts/estimate_frcnn_time.py::batch_shapes`: a real `DataLoader` over indices, the same seed, and
  torchvision's resize-and-pad rule.

## 3. Design decisions

- [ADR-012 Faster R-CNN training loop and resume](../DECISIONS.md#adr-012-faster-r-cnn-training-loop-and-resume)
- [ADR-011 Prepared-data location on Kaggle](../DECISIONS.md#adr-011-prepared-data-location-on-kaggle)
  (reused for finding `last.pt`)
- Deviations: [DEVIATIONS.md](../DEVIATIONS.md), the 2026-10-29 entries:
  - P3.3 before P3.2;
  - the explicit LR function;
  - the E1 config committed early and the CPU resume check;
  - the extra timing run.

## 4. Likely interview questions

1. **Why `num_classes=21`, and where can the off-by-one go wrong?** torchvision's detection heads treat
   label 0 as background. The dataset adds 1 to every VOC index and `predict_frcnn` subtracts 1. If either
   side forgets, every class is shifted by one, and mAP collapses while the loss still looks fine. A unit
   test runs the round trip on a fake model output.
2. **Why warm-up?** The new 21-class head starts random. At the full LR (0.01) its large early gradients
   flow back into the pretrained backbone and can wreck it. Ramping the LR from 0.001 × lr over the first
   500 steps lets the head settle first.
3. **Walk through the AMP step. Why `unscale_` before the gradient norm?** `scaler.scale(loss)` multiplies
   the loss so small float16 gradients don't underflow, which means the gradients are scaled too.
   `unscale_` divides them back, so the norm we log is the true one. `scaler.step` skips the update when
   it finds inf/NaN. That is why step 0 logs `grad_norm = inf`: the initial scale of 65536 overflowed, the
   step was skipped, and `update()` lowered the scale.
4. **Why compute the LR in a function instead of using `MultiStepLR`?** Warm-up runs per step and the decay
   runs per epoch. Chaining two schedulers means each multiplies the other's LR, in an order that matters.
   One function is the same schedule, is testable at exact points, and has no state to save for resume.
5. **What exactly does `last.pt` contain, and why each piece?**
   - Model and optimizer, including the momentum buffers. Without them the first resumed steps behave
     differently.
   - The AMP scaler's current scale.
   - The shuffle generator, so the resumed epoch sees the same order it would have.
   - Python/NumPy/torch/CUDA RNG states, for the augmentations.
   - The epoch, the global step (the LR depends on it), and the best score, so `best.pt` is not
     overwritten by a worse epoch.
6. **How do you resume after a Kaggle session dies?** A *new* kernel attaches the dead run's output as a
   kernel source. It can't be a new version of the same kernel, because a source is always the latest
   version. `--resume-run <run_id>` searches `/kaggle/input` for `<run_id>/weights/last.pt`. Logs are
   rebuilt without the unfinished epoch's rows. The untested part is whether Kaggle keeps the outputs of a
   session *it* kills; `kaggle kernels files` checks that before resuming.
7. **Your first timing showed a median of about 7 s/iter, but E1 is estimated at under an hour. Why?** Step
   times are two-valued: about 7.5 s when the padded batch shape is new, about 0.25 s when it repeats. The
   8-step smoke run was mostly new shapes. Replaying the seeded shuffle predicted every slow step exactly
   (64/64), and E1 has about 105 distinct shapes in 7,728 steps. The exact numbers are in
   `results/estimates/E1.json`.
8. **Why are the final metrics re-predicted from `best.pt` instead of reusing the validation pass?**
   After a resume, the validation pass that found the best epoch may have happened in a dead session. Re-
   predicting makes `predictions_val.json` and `metrics_val.json` always come from the saved weights and
   always reproduce each other.

## 5. Practice exercises

**Bug to find.** Someone "simplifies" the resume code to `torch.load(resume, map_location=device)` with
`device = cuda`, and resume crashes on GPU only. All tests still pass on CPU.

- *Hint 1:* what type and device do `torch.set_rng_state` and `Generator.set_state` expect?
- *Hint 2:* `map_location` moves *every* tensor in the checkpoint, not only the weights.
- *Hint 3:* why does `optimizer.load_state_dict` not care where its state was loaded?

**Feature to add.** Add `--max-steps N` for a quick debug run that stops after N iterations in total,
across epochs, and still writes `last.pt`.

- *Hint 1:* the global `step` already crosses epochs. Where would you check it, and how do you leave both
  loops cleanly?
- *Hint 2:* what should `last.pt`'s `epoch` field say if you stop mid-epoch? Think about what
  `start_log` will drop on resume.
- *Hint 3:* write the test first, with the tiny CPU fixture from `tests/test_train_frcnn.py`.
