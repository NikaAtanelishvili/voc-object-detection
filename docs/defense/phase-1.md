# Phase 1 defense notes: data

## 1. What was built

- `vocdet download` fetches the VOC2012 trainval tar (Oxford host first, then pjreddie.com). It verifies the
  MD5 read from torchvision at runtime, then extracts the archive. Reruns skip any step that is already done.
- `vocdet parse` turns the XML annotations into one parquet table (one row per object). This is the only
  place the 1-based → 0-based coordinate conversion happens.
- `vocdet splits` writes the frozen ID lists: train (official train minus a seeded 10%), val (that 10%),
  and test (official val). Rewriting them with different content raises an error.
- Exporters write the Ultralytics YOLO layout (difficult objects excluded) and one COCO JSON per split
  (difficult objects become `iscrowd` ignore regions). The Faster R-CNN `Dataset` has a hand-written
  horizontal flip and the +1 label offset.
- `vocdet integrity` stops on any mismatch with PLAN.md. `vocdet eda` writes generated tables, figures, and
  `reports/EDA.md`.
- `vocdet prepare` runs the whole chain. The Kaggle CPU kernel `vocdet-prepare-data` runs it, and training
  kernels mount its output.

## 2. Key code paths

- `data/download.py::voc2012_spec`: URL, filename, MD5, and base dir come from
  `torchvision.datasets.voc.DATASET_YEAR_DICT["2012"]`, never typed from memory.
- `data/download.py::download_file`: streams 1 MiB chunks to `<file>.part`, renames only when complete.
- `data/download.py::ensure_archive`: reuses the cached archive only if its MD5 matches; tries each URL and
  deletes any file whose MD5 is wrong.
- `data/download.py::extract`: `tarfile` `filter="data"` (blocks path traversal) plus a completion marker.
- `data/parse_voc.py::parse_annotation`: `(xmin - 1, ymin - 1, xmax, ymax)`, direct-child `bndbox` only
  (not `<part>`), validity check `0 ≤ x1 < x2 ≤ W`.
- `data/splits.py::make_splits`: sorts IDs, then `random.Random(42).sample`, `round(0.1 × n)` for val.
- `data/splits.py::write_splits`: identical content is a no-op; different content raises
  `SplitsFrozenError`.
- `data/export_yolo.py::to_yolo_line`: center/size normalized by image size, 8 decimals.
- `data/export_coco.py::build_coco`: `category_id = class_id + 1`, `bbox = [x, y, w, h]`, `area = w × h`,
  `iscrowd = difficult`.
- `data/transforms.py::hflip`: `[x1, x2] → [W − x2, W − x1]`.
- `data/dataset.py::VOCDetectionDataset`: labels `class_id + LABEL_OFFSET`, difficult objects excluded,
  empty `[0, 4]` targets allowed.
- `data/integrity.py::check_integrity`: collects every error, then raises once with all of them.
- `data/eda.py::render_markdown`: fills a fixed template, so every number in EDA.md is generated.
- `data/prepare.py::prepare_all`: the pipeline order; `build_splits` doubles as the frozen-splits check.

## 3. Design decisions

- [ADR-002 Split protocol](../DECISIONS.md#adr-002-split-protocol)
- [ADR-005 Python 3.11 floor](../DECISIONS.md#adr-005-python-311-floor)
- [ADR-006 Download source and integrity](../DECISIONS.md#adr-006-voc2012-download-source-and-integrity-check)
- [ADR-007 Canonical table and coordinates](../DECISIONS.md#adr-007-canonical-annotation-table-and-coordinate-convention)
- Deviations for this phase: [DEVIATIONS.md](../DEVIATIONS.md) (entries dated 2026-10-28 with task P1.x).

## 4. Likely interview questions

1. **Why `xmin − 1` but not `xmax − 1`?** VOC indices are 1-based and inclusive. Pixel `k` (1-based)
   covers the continuous interval `[k − 1, k]`. A box over pixels `xmin..xmax` therefore spans
   `[xmin − 1, xmax]`, and its width `xmax − xmin + 1` equals the number of pixels.
2. **How are difficult objects handled, and why differently in training and evaluation?** In training
   they are dropped: they are ambiguous, so we should not push the model towards them. In evaluation they
   are ignore regions (`iscrowd = 1`): a detection on one is neither rewarded nor punished, and missing one
   is not counted. That matches the official VOC protocol.
3. **Why is the test split the official val?** VOC2012 test labels are not public. Using official val as
   an untouched test set and carving val out of train is the standard way to get an honest final number.
4. **How do you guarantee the splits never change?** They are committed. `write_splits` refuses different
   content, and the Kaggle prep kernel recomputes them from its own copy of the data and compares. A
   mismatch would fail the kernel.
5. **The fallback mirror is someone's personal website. Is that safe?** We don't trust the host, we trust
   the MD5 from torchvision. A file with a different MD5 is deleted before extraction. MD5 isn't collision
   resistant against an attacker, which is acceptable for a public dataset (ADR-006).
6. **Why does the Faster R-CNN dataset add 1 to the labels?** torchvision detection models reserve class
   0 for background, so `num_classes = 21` and VOC class `k` is label `k + 1`. Predictions are shifted
   back by the same `LABEL_OFFSET` (P3.3).
7. **Why are EDA statistics only on train and val?** Looking at test labels could influence modelling
   choices, which would leak the test set into development. (The integrity check prints per-split class
   counts as a sanity check, but that is not used for any decision.)
8. **Why 8 decimals in YOLO labels?** With 6, the rounding error after a round trip can reach about
   2.5e-4 px on a 500 px image, above the 1e-4 tolerance. It has no effect on training, but it makes the
   round-trip test meaningful.

## 5. Practice exercises (do these without AI)

**Bug to find.** Suppose someone "simplifies" `hflip` to `flipped[:, 0] = width - boxes[:, 0]` and
`flipped[:, 2] = width - boxes[:, 2]`. Which existing tests catch it, and what would happen in training if
none did? *Hints:* what is the relation between `x1` and `x2` after this change? Which validity condition
from `parse_voc` would the flipped boxes violate? Look at `test_hflip_keeps_boxes_valid`.

**Feature to add.** Add an optional `--min-box-size N` filter to `vocdet export` that drops objects smaller
than N px on either side from the YOLO labels only (not from the COCO ground truth). *Hints:* think about
why the evaluator's ground truth must stay unfiltered; add the column check in `export_yolo` before
`to_yolo_line`; write one test with a tiny box in a fixture; log how many objects were dropped per split.
