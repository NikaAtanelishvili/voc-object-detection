# Phase 2 defense notes: evaluation harness

## 1. What was built

- One prediction schema (`Prediction`, `GroundTruth`) that every model converts to. It is validated on
  construction and round-trips through JSON.
- One evaluator (`evaluate`): torchmetrics `MeanAveragePrecision` with the pycocotools backend.
  Difficult objects are passed as `iscrowd` ignore regions, and a second IoU-0.5 instance gives per-class
  AP50.
- Our own TIDE-lite error analysis. Every detection above a score threshold becomes TP, Cls, Loc, Both,
  Dupe, Bkg, or Ignored; every unmatched object becomes a Miss. There is also a 21×21 confusion matrix,
  and the IoU is hand-written.
- Visualisation: captioned boxes, and per-class failure galleries (top false positives, largest misses).
- The run artifact contract: `run_meta.json` (git SHA, seed, versions, GPU, wall time), seeding, and
  `collect_results.py`, which builds `results/experiments.csv`.
- E0b: a COCO-pretrained YOLO26n evaluated zero-shot on VOC val through the whole chain, on a Kaggle T4.
  The Kaggle runner now guards the preinstalled torch/torchvision.

## 2. Key code paths

- `eval/schema.py::Prediction.__post_init__`: reshapes to `(N, 4)`, checks `x2 ≥ x1`, labels in 0–19,
  scores in [0, 1].
- `eval/schema.py::ground_truth_from_annotations`: builds per-image ground truth from the parquet table,
  in split order, including images without objects.
- `eval/evaluator.py::_to_torchmetrics`: pairs by `image_id`; a missing or extra prediction entry raises.
- `eval/evaluator.py::evaluate`: two `MeanAveragePrecision` instances (COCO thresholds; IoU 0.5 only with
  `class_metrics`), and `-1 → None`.
- `eval/error_analysis.py::box_iou`: broadcasting intersection over union, written by hand.
- `eval/error_analysis.py::analyze_image`: VOC argmax matching, then the Cls → Loc → Both → Bkg order.
- `eval/error_analysis.py::analyze`: confusion-matrix bookkeeping (rows true, columns predicted, index 20
  = background).
- `eval/visualize.py::failure_gallery`: sorting by score (false positives) and by area (misses),
  thumbnails pasted into a strip.
- `eval/predict.py::build_class_map`: maps by name, which works for COCO models and VOC models alike.
- `eval/predict.py::to_voc_prediction`: drop unmapped classes → sort by score → top 100.
- `eval/zero_shot.py::run_zero_shot`: seed → `start_run_meta` → predict → write predictions → evaluate →
  `finish_run_meta`.
- `utils/run_meta.py::git_sha`: the runner's environment variable on Kaggle, otherwise `git rev-parse`,
  with a `-dirty` suffix when the tree is dirty.
- `scripts/collect_results.py::collect`: copies the small artifacts and rebuilds the CSV idempotently.
- `kaggle/templates/runner.py.j2::install_without_touching_guarded`: pip constraints file plus a
  before/after version check.
- `data/prepare.py::find_prepared_dir`: `os.walk` over `/kaggle/input` for
  `<prep_run_id>/voc_prepared/annotations.parquet`, pruning `voc_prepared/`; requires exactly one match.
- `eval/schema.py::write_predictions`: stores float32 values unrounded, so a saved file reproduces the
  in-run metrics bit for bit.

## 3. Design decisions

- [ADR-008 Evaluator design](../DECISIONS.md#adr-008-evaluator-design)
- [ADR-009 TIDE-lite rules](../DECISIONS.md#adr-009-tide-lite-error-analysis-rules)
- [ADR-010 Linux-only lock and ultralytics pin](../DECISIONS.md#adr-010-linux-only-lock-resolution-and-the-ultralytics-pin)
- [ADR-011 Prepared-data location on Kaggle](../DECISIONS.md#adr-011-prepared-data-location-on-kaggle)
- Deviations: [DEVIATIONS.md](../DEVIATIONS.md) (P2.6 entry; the P4.1 and P5.1 owner decisions).

## 4. Likely interview questions

1. **Why one schema and one evaluator?** If each model used its own metric code (e.g. Ultralytics' `val`
   for YOLO, our own for Faster R-CNN), differences in interpolation, difficult handling, or max
   detections would contaminate the comparison. Converting everything to `Prediction` makes the evaluator
   the single source of truth.
2. **Why two `MeanAveragePrecision` instances?** `class_metrics=True` reports per-class AP averaged over
   IoU 0.50:0.95. We want per-class AP at 0.5, so the second instance uses `iou_thresholds=[0.5]`. A test
   checks that mAP50 equals the mean of the per-class AP50 values.
3. **How do difficult objects enter the evaluation?** As `iscrowd = 1`. pycocotools never counts a
   detection matched to a crowd region as an FP, and an unmatched crowd region is not a miss. One caveat:
   COCO's crowd IoU divides by the detection's area, so it is slightly more lenient than VOC.
4. **Why does a missing prediction entry raise instead of counting as empty?** A dropped image (a crash, a
   filtering bug) would silently remove its misses and false positives. Raising forces adapters to emit an
   explicit, possibly empty, entry.
5. **Walk through how one false positive is classified.** It is above the threshold. First the best
   same-class box (difficult included): if IoU ≥ 0.5 it is TP, Dupe, or Ignored. Otherwise: other class
   with IoU ≥ 0.5 → Cls; same class with 0.1–0.5 → Loc; other class with 0.1–0.5 → Both; else Bkg.
6. **Why a score threshold in error analysis but not in mAP?** mAP integrates over all thresholds, so
   low-score detections only add points at the tail of the PR curve. Error counts are a snapshot at one
   operating point; at 0.001 they would be dominated by near-zero-score background hits.
7. **Why ask YOLO for 300 detections and keep 100?** For the COCO model, part of the top 100 are non-VOC
   classes (e.g. "truck"). Keeping the top 100 *after* dropping them gives the zero-shot model the same
   100-detection budget as fine-tuned models.
8. **How do you know a Kaggle run used the code you think it did?** The runner checks out an exact SHA,
   exports it as `VOCDET_GIT_SHA`, and `run_meta.json` records it. Local runs from a dirty tree are marked
   `-dirty`.
9. **E0b failed twice before it worked. What happened?** Kaggle mounted the prep output at
   `/kaggle/input/<slug>/` in one kernel and at `/kaggle/input/notebooks/<owner>/<slug>/` in another.
   The first search was too shallow, and the second pinned the path from the wrong kernel. The fix
   searches for the prep *run ID* at any depth, demands exactly one match, and a CPU `--resolve-only` run
   proves the path before any GPU time is spent (ADR-011).
10. **Why did you rerun E0b after it succeeded?** The saved predictions rounded boxes to 0.01 px, so
    re-evaluating the file gave `map50_95` 2e-4 off the in-run value. Bootstrap CIs and error analysis read
    the file, so file and metrics must agree exactly. After the fix they are identical.

## 5. Practice exercises (do these without AI)

**Bug to find.** In `analyze_image`, suppose the line `matched[j] = True` were moved *outside* the
`elif not matched[j]` branch, so it runs for Dupe and Ignored outcomes too. Which counts change, and which
test catches it? *Hints:* what happens when the Ignored branch marks a difficult box as matched? Does that
change the Miss loop (`~matched & ~gt.difficult`)? Think about whether any test would still pass, and
write the case that would fail.

**Feature to add.** Add an `--iou-fg` option to `vocdet errors` so the TP/Cls threshold can be 0.75
instead of 0.5. *Hints:* `IOU_FG` is a module constant used in two places; pass it down as a parameter
instead of changing the constant; extend `test_every_outcome_is_classified` with a detection whose IoU is
0.6 and check that it flips from TP to Loc at 0.75.
