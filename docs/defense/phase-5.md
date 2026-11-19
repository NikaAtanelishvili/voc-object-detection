# Phase 5 defense notes: final test evaluation (Phase 5 core)

Phase 5 was cut to its core by the owner's reorder: P5.2 now, P5.1's remaining tables and figures and
P5.3's error analysis in Phase 8.

## 1. What was built

- `scripts/final_test_eval.py`: the only code that touches the test split. `predict` scores E0b, E1, E2,
  E3, E6s and E6n on all 5823 test images, through the same adapters and evaluator as val. It ran as a
  Kaggle CPU kernel, with weights taken from the training kernels' outputs, and records each
  checkpoint's sha256. `record` writes `results/test_metrics.csv` once; a second result for an experiment
  needs `--force-rerun-with-reason`, which is logged.
- `eval/resample.py::MatchedImages`: scores a bootstrap draw by rerunning only pycocotools' accumulation
  over per-image matching computed once. It equals `evaluate()` on the resampled set to float32
  precision, and is about 100× faster. That makes 1000 draws affordable, on test too.
- `scripts/bootstrap_ci.py --split test`: paired delta CIs on the saved test predictions, for E6s against
  E0b, E3, E6n and E1. Val comparisons were recomputed with 1000 draws.
- The E6s seed-43 rerun (val only), to measure seed-to-seed spread.

## 2. Key code paths

- `scripts/final_test_eval.py::check_once`: refuses recorded experiments unless a reason is given. It
  reads the CSV path at call time, because a default bound at definition time hid it from a test.
- `scripts/final_test_eval.py::predict_one`: chooses COCO weights (E0b), `build_frcnn` (E1) or YOLO for
  each experiment, and hashes the weights used.
- `scripts/final_test_eval.py::run_record`: CSV row, `metrics_test.json` into `results/`, predictions into
  `runs/` (too big for git).
- `eval/resample.py::MatchedImages.__init__`: torchmetrics' own COCO conversion, then `COCOeval.evaluate()`
  once.
- `eval/resample.py::MatchedImages.score`: rebuilds `evalImgs` in draw order (duplicates included), runs
  `accumulate`, and takes the mean of defined precision, as `summarize` does.

## 3. Design decisions

- ADR-003: E6s as the deployment candidate, with a revisit condition on test.
- ADR-014: paired bootstrap; 1000 draws; the fast scoring path.
- ADR-015: E6 was designed after viewing val, which is why the test split decides.

## 4. Likely interview questions

1. **How do you know the test split was not used for tuning?** Only `final_test_eval.py` reads it, and it
   ran once, after ADR-003 had chosen the model. Its rows carry the weights' sha256, matching the Hub
   backups, and a second recording is refused unless a reason is logged.
2. **Why is the fast bootstrap legitimate?** pycocotools matches detections to ground truth per image and
   per class. That matching does not change when other images are added or duplicated; only accumulating
   the precision-recall curve over images does. So I match once and accumulate per draw, with the
   library's own functions, and a test checks equality with the full evaluator.
3. **Why 1000 draws instead of 200?** A 95% interval's endpoints are the 2.5th and 97.5th percentiles.
   With 200 draws, each endpoint rests on about 5 values, so it moves noticeably between seeds.
4. **What does the seed-43 run tell you?** How much E6s moves when only the seed changes. The gap is about
   the size of some of the Phase 4 differences, so a single-seed result of that size is weak evidence.
   It also ran on Kaggle's newer image, so seed and environment are mixed in that number.
5. **Why is E6 called "post-hoc"?** It was designed after seeing a val result (the 3-epoch timing run),
   so picking it on val is optimistic. The test result is the honest estimate, which is why the reports
   lead with test numbers.
6. **Why did the test evaluation run on Kaggle CPU?** It needs over an hour of CPU (Faster R-CNN is
   slow). A local background job dies with the session; a kernel does not, and CPU kernels cost no GPU
   quota.

## 5. Practice exercises

**Bug to find.** In `MatchedImages.score`, suppose the index into `evalImgs` were written as
`self.coco_eval.evalImgs[int(i) * self.num_categories + k]`. Would any test fail? What would the
bootstrap values look like on a dataset where most images hold one class? Hints: look at the comment on
the order of `evalImgs`, and at what `tests/test_resample.py` compares.

**Feature to add.** Add a `--metrics` option to `bootstrap_ci.py` that also reports AR@100 (`mar_100`)
deltas. Hints: `MatchedImages.score` would need the recall array from `accumulate`. Which `maxDets` value
does `mar_100` use, and is it already the one restricted in `__init__`? Extend the equality test.
