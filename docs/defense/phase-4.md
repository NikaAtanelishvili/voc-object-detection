# Phase 4 defense notes: YOLO26 experiments

## 1. What was built

- `train_yolo`: our YAML config becomes Ultralytics `train()` arguments. Only settings the config sets are
  passed, and the optimizer stays `auto` unless E6 asks for AdamW.
- A per-run writable dataset root, because `/kaggle/input` is read-only and Ultralytics writes a label
  cache: images are symlinked, labels copied (in a thread pool, since the mount is slow per file), and the
  YAML lists train and val only.
- After training, `best.pt` predicts val through the same adapter and evaluator as every other model.
  Ultralytics' own `model.val()` runs as a cross-check, and the gap between the two was broken down into
  its causes.
- Runs: E0a-yolo smoke and two full-data timing runs, E2/E3 (planned recipe), E4 (no augmentation),
  E5 (frozen backbone), and E6s/E6n (gentle fine-tune). Each `best.pt` is backed up to the Hub with a
  matching sha256.
- A paired bootstrap of differences between runs (ADR-014), and ADR-003 picking E6s for deployment.

## 2. Key code paths

- `train/train_yolo.py::build_dataset_root`: symlinks images, copies labels, writes `voc_yolo.yaml`.
  Ultralytics finds a label by replacing `/images/` with `/labels/` in the *unresolved* path.
- `train/train_yolo.py::ultralytics_train_args`: config → Ultralytics keyword arguments, `None` = default.
- `train/train_yolo.py::ultralytics_val_crosscheck`: Ultralytics val next to our score with difficult
  objects removed, warning above `CROSSCHECK_TOLERANCE`.
- `eval/predict.py::build_class_map`: model class names → VOC indices. It handles VOC names, and COCO names
  via `COCO_TO_VOC`, which is how E6's COCO-spelled names map back.
- `constants.py::COCO_NAMES_IN_VOC_ORDER`: VOC order spelled the COCO way, so `cls_remap` copies all 20
  head rows.
- `scripts/bootstrap_ci.py::resample_indices` / `delta_row`: seeded draws shared by every run, and the
  percentile CI of the per-draw difference.
- `scripts/collect_results.py::collect`: collects only the runs listed in `results/canonical_runs.yaml`.
- `scripts/kaggle_pull.py::wait_for_kernel`: polls, tolerating up to 4 failed status calls in a row.

## 3. Design decisions

- ADR-003: E6s is the deployment candidate, with E6n as the fallback.
- ADR-013: our evaluator is authoritative; Ultralytics val is a cross-check, and the gap is measured.
- ADR-014: runs are compared only through paired bootstrap CIs of the difference.
- ADR-015: what YOLO inherits from the COCO head, the forgetting hypothesis, and the E6 design.

## 4. Likely interview questions

1. **Why does fine-tuned YOLO score below the zero-shot model (E2, E3 vs E0b)?** It doesn't, measurably:
   the paired CIs include zero. What stands out is that 50 epochs at the default LR did not improve on
   COCO weights that already cover all 20 classes, and the 6 classes with fresh head rows lost the most.
   A short, low-LR fine-tune with all 20 rows copied (E6) does beat it.
2. **What does `cls_remap` do, and how did you find it?** Ultralytics 8.4.191 copies COCO classification
   rows to the new class order by exact name. The log said `Transferred 708/708` although the head shape
   changed, which made no sense for a fresh head, so I read `nn/tasks.py::_remap_cls_by_names`.
3. **Why a paired bootstrap?** Two models scored on the same val images share most of the noise: an easy
   resample lifts both. Separate CIs overlap even when one model is better on almost every resample.
   Scoring both on the same draws and taking the CI of the difference removes the shared part.
4. **Why is your mAP different from Ultralytics' val mAP?** Same weights, two inference paths. Our
   adapter and evaluator agree with Ultralytics' own metric on the same boxes. Most of the gap is
   Ultralytics' val dataloader (rect batches with padding) scoring above its own `predict()`. We score
   `predict()`, because that is what the service runs.
5. **Why symlinks and copied labels instead of using `/kaggle/input` directly?** Ultralytics writes a
   `.cache` file next to the labels, and the input mount is read-only. Copying 2 GB of images per run
   would be slow; symlinks are free, and labels are tiny.
6. **E6 was designed after you saw val. Isn't that overfitting to val?** Yes, as a selection it is
   optimistic, and the report says so. The test split has never been used; P5.2 evaluates the chosen
   models on it once, and that decides.
7. **Why did your time estimates go wrong twice?** The first one inherited a slow serial dataset-root
   build that was later fixed; the other used a session whose input mount happened to be warm. The
   estimator now takes one-time costs from a recent cold run. E6s still came in about 7% over its
   "upper bound", so it is an estimate, not a guarantee.
8. **What did E4 and E5 tell you?** Without augmentation, YOLO26n peaks in its first epoch and gets
   worse: augmentation matters on 5k images. Freezing the backbone costs nothing measurable, which fits
   the idea that the COCO features are already good for VOC.

## 5. Practice exercises

**Bug to find.** Suppose someone "simplifies" `bootstrap_scores` by dropping the `by_id` lookup and using
`predictions` in file order directly: `ordered = predictions`. All tests still pass. For which runs could
the CIs now be wrong, and how badly? Hints: who writes `predictions_val.json`, and in what order? Is that
order guaranteed to be the order of `ground_truth_from_annotations`? Why do the test fixtures not notice?
Write a test that would.

**Feature to add.** Add a `--pairs` option to `scripts/bootstrap_ci.py` that takes extra `A:B` experiment
pairs on the command line, on top of `PAIRS`, without recomputing any run's scores. Hints: which part of
`main` reads cached scores, and which writes the CSV? What should happen if A or B has no
`bootstrap_ci.json` yet? Add a test that uses two fake cached results.
