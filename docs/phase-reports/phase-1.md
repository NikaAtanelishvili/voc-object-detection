# Phase 1 report: Data

Branch `phase-1-data`. Exit criteria:
- Prepared data available on Kaggle ✔ (`nikaatanelishvili/vocdet-prepare-data`)
- EDA committed ✔
- Splits frozen ✔
- Defense notes written ✔ (`docs/defense/phase-1.md`)
- Tag `phase-1`: pending your approval

CI is green on every push of the branch.

## Done

| Task | Summary | Commits |
|---|---|---|
| Phase 0 follow-ups | CI matrix on 3.11 + 3.13, runner pinned to ubuntu-24.04; owner decisions recorded (Docker on 3.13, P4.1 must leave torch untouched) | `850ad69`, `7a45f46` |
| P1.1 | Python floor raised to 3.11 (ADR-005). `download.py`: torchvision MD5, Oxford host then pjreddie.com fallback (ADR-006), `.part` files, safe extraction. Fix: root-anchored `/data/` ignore rule | `f2ded7a`, `ad8e1ab`, `fa89317` |
| P1.2 | `parse_voc.py`: XML → 0-based xyxy parquet with validity checks; hand-written fixtures; ADR-007 | `3694d28` |
| P1.3 | `COCO_TO_VOC` keyed by Ultralytics COCO names | `d7b5224` |
| P1.4 | Frozen splits with a guard against overwriting them; ADR-002 | `d56c070` |
| P1.5 | YOLO layout (difficult excluded) and COCO JSON (difficult = `iscrowd`) exporters | `0d75106` |
| P1.6 | Faster R-CNN `Dataset`, hand-written flip, `LABEL_OFFSET = 1`, `collate_fn` | `d19c746` |
| P1.7 | Integrity checks; all pass on real VOC2012 | `62b6190` |
| P1.8 | EDA tables, figures, and generated `EDA.md`. Fix: deterministic table order | `c715b7a`, `0415ff6` |
| P1.9 | `prepare_all` + `vocdet prepare`; CPU kernel run and pulled. Fix: pulls no longer hit the rate limit | `7b7605e`, `ecb45c2`, `3f917e9` |

## Deviations

From `docs/DEVIATIONS.md`, all dated 2026-10-28:

1. **P0.3:** CI runs on Python 3.11 and 3.13 (your decision).
2. **P0.3:** CI runner pinned to `ubuntu-24.04` (your decision).
3. **P7.4:** the app Docker image will use Python 3.13 (your decision).
4. **P4.1:** added acceptance criterion: installing ultralytics on Kaggle must leave torch and torchvision unchanged (your decision).
5. **P1.1:** Python floor raised from 3.10 to 3.11 (your decision, ADR-005).
6. **P1.1:** fallback mirror is pjreddie.com, not the Ultralytics zip. Reported under hard rule 8; you chose this option (ADR-006).
7. **P1.2:** `VOC_CLASSES` created before P1.3, and only the trainval XMLs are parsed (ADR-007).
8. **P1.8:** the sample grid is saved as JPEG; as PNG it was 2.67 MB, over the 500 KB limit.
9. **P1.9:** prep output goes under `<run_id>/voc_prepared`, the pipeline lives in a new `prepare.py`, and a `pull_pattern` config field controls what gets downloaded.

## Results

From `results/eda/summary.json` (train and val only; test labels not inspected):

- `"images": {"train": 5145, "val": 572, "test": 5823}`
- `"objects_non_difficult": {"train": 12277, "val": 1332}`
- `"top_class": "person"`, `"top_class_share": 0.30838152643153866`
- `"rarest_class": "sofa"`, `"imbalance_ratio": 14.731517509727626`
- `"small_share_overall": 0.06010728194577118`, `"difficult_share": 0.13725117281602636`

The Kaggle run reproduced the local outputs (`results/env/kaggle_prepare_data_run.json`:
`"returncode": 0`, `"wall_time_s": 187.3`). The parquet, all three COCO JSONs, and every EDA table and
summary are byte-identical to the local run. The only difference was the row order of `resolutions.csv`,
fixed in `0415ff6`.

Checked against PLAN.md §3: the official `train.txt` / `val.txt` have 5,717 / 5,823 images, as expected.
Tests: 85 passed.

## GPU budget

| | Hours |
|---|---|
| Used this phase (the prep kernel ran on CPU) | 0 |
| Cumulative (see the `docs/ENVIRONMENT.md` usage log) | 0.05 |
| Project budget (PLAN.md §5) | ~15.5 |
| Remaining against budget | ~15.45 |

## Manual verification checklist

1. **The coordinate conversion is right.**
   - Run `uv run pytest tests/test_parse_voc.py -v`. Expected: 9 passed.
   - Then read `parse_annotation` in `src/vocdet/data/parse_voc.py` and explain why only `xmin` and `ymin`
     get `- 1`.
   - A failure means every box in every model is shifted.
2. **Boxes line up with the objects in real images.**
   - Open `reports/figures/eda_samples.jpg`. Expected: boxes fit their objects tightly, with no constant
     offset.
   - A visible shift would mean the conversion or the drawing code is wrong.
3. **The splits are disjoint and the right size.**
   - Run `wc -l splits/*.txt`. Expected: 5145 / 572 / 5823 (11540 total).
   - Run `comm -12 <(sort splits/train.txt) <(sort splits/val.txt) | wc -l`. Expected: `0`.
4. **The frozen-splits guard works.**
   - Run `sed -i '1d' splits/val.txt && uv run vocdet splits`. Expected: `SplitsFrozenError`.
   - Then restore the file with `git checkout splits/val.txt`.
   - This uses the local copy in `data/`. If nothing is raised, the splits could change silently.
5. **Difficult objects become ignore regions.**
   - Run `uv run python -c "import json; d=json.load(open('data/prepared/coco/val.json')); print(sum(a['iscrowd'] for a in d['annotations']), len(d['annotations']))"`.
   - Expected: a non-zero first number. It is the val share of difficult objects that the EDA reports.
6. **The EDA numbers are generated, not typed.**
   - Pick one finding in `reports/EDA.md`, e.g. the person share.
   - Recompute it from `results/eda/class_counts.csv`: person `train` divided by the sum of the `train`
     column.
7. **The prepared data exists on Kaggle.**
   - Open <https://www.kaggle.com/code/nikaatanelishvili/vocdet-prepare-data>, Output tab.
   - Expected: `prepare_data_20261028-0208_7b7605e/voc_prepared/` with `yolo/images/{train,val,test}`,
     `coco/`, and `annotations.parquet`.
8. **CI is green.**
   - Run `gh run list --branch phase-1-data --limit 3`. Expected: all `completed success`.

## Open issues / risks

1. **The Kaggle mount path isn't verified yet.**
   - Expected: `/kaggle/input/vocdet-prepare-data/prepare_data_20261028-0208_7b7605e/voc_prepared/`.
   - The first GPU kernel (E0b, P2.6) logs the real path, and `configs/data/voc2012.yaml` is created from it.
2. **The YOLO yaml on Kaggle won't work as-is for training.**
   - Its `path` points at `/kaggle/working/...` in the prep kernel, and the mounted input is read-only.
   - Ultralytics also tries to write label `.cache` files next to the labels.
   - Plan: `train_yolo.py` (P4.1) writes its own yaml that points at the mount, and handles the read-only
     cache.
3. **Kaggle's versions differ from our pins** (pyarrow 23.0.1, matplotlib 3.10.0). The outputs were
   identical, so this is harmless. It's recorded in ENVIRONMENT.md.
4. **The Kaggle API rate limit** can affect pulls from kernels with many outputs. It's mitigated (page
   size 200 plus one retry); if it comes back, retry later.
5. **Val is small (572 images),** so val mAP is noisy. ADR-002 says not to over-interpret differences
   under about one point.
6. **The repo's own prep output on Kaggle predates the `resolutions.csv` ordering fix.** Training never
   reads that file, so I don't plan to rerun the kernel.

## Next phase (Phase 2: evaluation harness)

- **P2.1** `eval/schema.py`: `Prediction` / `GroundTruth` dataclasses with JSON read/write.
- **P2.2** `eval/evaluator.py`: a torchmetrics `MeanAveragePrecision` wrapper with `iscrowd` from the
  difficult flag. I'll first check that the installed torchmetrics accepts `iscrowd`. Kaggle has
  torchmetrics 1.9.0.
- **P2.3** `eval/error_analysis.py`: TIDE-lite with a hand-written IoU.
- Phase 2 has no [MANUAL] steps. P2.6 (E0b zero-shot) is the first GPU run of the plan, estimated at 0.2 h.
