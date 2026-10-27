"""Exploratory data analysis: summary tables in results/eda/, figures, and reports/EDA.md.

Statistics use only the train and val splits. Test labels are never inspected before Phase 5, so no
modelling choice can be influenced by them (hard rule 4); only the test image count is reported.
Every number in EDA.md is filled in by this script from the computed tables (hard rule 3).
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on Kaggle or CI
import matplotlib.pyplot as plt  # noqa: E402  (must come after use("Agg"))
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from vocdet.constants import VOC_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

# COCO size buckets on box area in original image pixels (PLAN.md §4).
SMALL_MAX_AREA = 32**2
LARGE_MIN_AREA = 96**2
SIZE_BUCKETS = ("small", "medium", "large")
EDA_SPLITS = ("train", "val")
FIG_DPI = 100  # with JPEG for the photo grid, keeps every figure under the 500 KB limit


def size_bucket(area: float) -> str:
    """COCO convention: small < 32^2 <= medium < 96^2 <= large."""
    if area < SMALL_MAX_AREA:
        return "small"
    if area < LARGE_MIN_AREA:
        return "medium"
    return "large"


def add_box_columns(annotations: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Annotations of the EDA splits only, with split, box size, aspect ratio, and size bucket."""
    split_of = {i: name for name in EDA_SPLITS for i in splits[name]}
    df = annotations[annotations["image_id"].isin(split_of.keys())].copy()
    df["split"] = df["image_id"].map(split_of)
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["area"] = df["box_w"] * df["box_h"]
    df["aspect"] = df["box_w"] / df["box_h"]
    df["bucket"] = df["area"].map(size_bucket)
    return df


def compute_tables(df: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """All summary tables. Objects counted here exclude difficult ones unless the name says so."""
    kept = df[~df["difficult"]]
    class_counts = (
        kept.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(VOC_CLASSES)
    )
    class_counts = class_counts.reindex(columns=list(EDA_SPLITS), fill_value=0)
    class_counts["share_train"] = class_counts["train"] / class_counts["train"].sum()

    buckets = kept.groupby(["class_name", "bucket"]).size().unstack(fill_value=0)
    buckets = buckets.reindex(index=VOC_CLASSES, columns=list(SIZE_BUCKETS), fill_value=0)
    buckets["small_share"] = buckets["small"] / buckets[list(SIZE_BUCKETS)].sum(axis=1)

    flags = df.groupby("class_name")[["difficult", "truncated"]].mean().reindex(VOC_CLASSES)
    flags.columns = ["difficult_share", "truncated_share"]

    all_ids = [i for name in EDA_SPLITS for i in splits[name]]
    per_image = kept.groupby("image_id").size().reindex(all_ids, fill_value=0)
    objects_per_image = per_image.value_counts().sort_index().rename("images").to_frame()
    objects_per_image.index.name = "objects"

    sizes = df.groupby("image_id")[["width", "height"]].first()
    resolutions = sizes.value_counts().rename("images").to_frame()
    return {
        "class_counts": class_counts,
        "size_buckets": buckets,
        "flags": flags,
        "objects_per_image": objects_per_image,
        "resolutions": resolutions,
        "per_image": per_image.rename("objects").to_frame(),
        "aspect": df[["class_name", "aspect"]],
    }


def compute_summary(df: pd.DataFrame, tables: dict, splits: dict[str, list[str]]) -> dict:
    """Scalar facts quoted in EDA.md."""
    kept = df[~df["difficult"]]
    counts = tables["class_counts"]["train"]
    present = counts[counts > 0]
    aspect_by_class = df.groupby("class_name")["aspect"].median().sort_values()
    buckets = tables["size_buckets"]
    per_image = tables["per_image"]["objects"]
    sizes = tables["resolutions"].reset_index()
    return {
        "images": {name: len(splits[name]) for name in ("train", "val", "test")},
        "objects_non_difficult": {s: int((kept["split"] == s).sum()) for s in EDA_SPLITS},
        "difficult_share": float(df["difficult"].mean()),
        "truncated_share": float(df["truncated"].mean()),
        "top_class": counts.idxmax(),
        "top_class_share": float(counts.max() / counts.sum()),
        # Ratio over classes that occur; any class with no training objects is listed explicitly.
        "rarest_class": present.idxmin(),
        "imbalance_ratio": float(present.max() / present.min()),
        "classes_without_train_objects": counts[counts == 0].index.tolist(),
        "small_share_overall": float(
            buckets["small"].sum() / buckets[list(SIZE_BUCKETS)].sum().sum()
        ),
        "smallest_classes": buckets["small_share"].sort_values(ascending=False).index[:3].tolist(),
        "objects_per_image_median": float(per_image.median()),
        "objects_per_image_max": int(per_image.max()),
        "single_object_image_share": float((per_image == 1).mean()),
        "images_only_difficult": int((per_image == 0).sum()),
        "max_width": int(sizes["width"].max()),
        "max_height": int(sizes["height"].max()),
        "distinct_resolutions": len(sizes),
        "aspect_p05": float(df["aspect"].quantile(0.05)),
        "tallest_classes": aspect_by_class.index[:3].tolist(),
        "widest_classes": aspect_by_class.index[-3:][::-1].tolist(),
        "aspect_p95": float(df["aspect"].quantile(0.95)),
    }


def save_figures(df: pd.DataFrame, tables: dict, voc_dir: Path, fig_dir: Path) -> list[str]:
    """Write all eda_*.png figures; return their file names."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(
            fig_dir / name,
            dpi=FIG_DPI,
            pil_kwargs={"quality": 80} if name.endswith(".jpg") else None,
        )
        plt.close(fig)
        written.append(name)

    fig, ax = plt.subplots(figsize=(10, 4))
    tables["class_counts"][list(EDA_SPLITS)].plot.bar(ax=ax)
    ax.set_ylabel("objects (difficult excluded)")
    ax.set_title("Instances per class and split")
    save(fig, "eda_class_counts.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    tables["objects_per_image"]["images"].plot.bar(ax=ax)
    ax.set_xlabel("non-difficult objects per image")
    ax.set_ylabel("images")
    ax.set_title("Objects per image (train + val)")
    save(fig, "eda_objects_per_image.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    b = tables["size_buckets"][list(SIZE_BUCKETS)]
    (b.div(b.sum(axis=1), axis=0)).plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("share of objects")
    ax.set_title("COCO size buckets per class (small < 32², large ≥ 96² px)")
    save(fig, "eda_size_buckets.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["aspect"], bins=[2**k for k in range(-5, 6)], log=False)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("box width / height (log2 scale)")
    ax.set_ylabel("objects")
    ax.set_title("Box aspect ratios")
    save(fig, "eda_aspect_ratio.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    res = tables["resolutions"].reset_index()
    ax.scatter(res["width"], res["height"], s=res["images"] ** 0.5 * 3, alpha=0.5)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image resolutions (marker size ~ image count)")
    save(fig, "eda_resolution.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    tables["flags"].plot.bar(ax=ax)
    ax.set_ylabel("share of objects")
    ax.set_title("Difficult and truncated share per class")
    save(fig, "eda_difficult_truncated.png")

    train_ids = df[df["split"] == "train"]["image_id"].drop_duplicates()
    sample_ids = train_ids.sample(min(12, len(train_ids)), random_state=42).tolist()
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax in axes.flat:
        ax.axis("off")  # unused cells stay blank when there are fewer than 12 images (tests)
    for ax, image_id in zip(axes.flat, sample_ids, strict=False):
        ax.imshow(Image.open(voc_dir / "JPEGImages" / f"{image_id}.jpg"))
        for row in df[df["image_id"] == image_id].itertuples():
            color = "orange" if row.difficult else "lime"
            ax.add_patch(
                Rectangle((row.x1, row.y1), row.box_w, row.box_h, fill=False, color=color, lw=2)
            )
            ax.text(row.x1, row.y1, row.class_name, fontsize=8, backgroundcolor=color)
        ax.set_title(image_id, fontsize=9)
    fig.suptitle("12 random train images (orange = difficult)")
    save(fig, "eda_samples.jpg")  # JPEG: a photo grid as PNG is several MB
    return written


REPORT_TEMPLATE = """# Exploratory data analysis

_Generated by `src/vocdet/data/eda.py` from `results/eda/`. Do not edit by hand._

Statistics cover the **train** and **val** splits only. The test split (official VOC2012 val) is
not inspected before Phase 5; only its image count is shown.

## Overview

| Split | Images | Objects (non-difficult) |
|---|---|---|
| train | {n_train} | {obj_train} |
| val | {n_val} | {obj_val} |
| test | {n_test} | not inspected |

## Findings that affect modelling

1. **Class imbalance.** `{top_class}` is {top_class_share} of training objects; the most frequent
   class has {imbalance_ratio}x the objects of the rarest (`{rarest_class}`). Expect per-class AP to
   vary widely, so report per-class AP, not only the mean.
2. **Small objects.** {small_share} of objects are COCO-small (< 32² px). The classes with the
   highest small-object share are {smallest_classes}. Expect AP_small to be the weakest bucket and
   these classes to have more misses; input resolution matters most for them.
3. **Few objects per image.** The median image has {opi_median} non-difficult objects,
   {single_share} of images contain exactly one, and the maximum is {opi_max}. The default
   `max_det=100` is far above what any image needs.
4. **Difficult and truncated.** {difficult_share} of objects are marked difficult (excluded from
   training, ignore regions in evaluation) and {truncated_share} are truncated.
   {only_difficult} train/val images contain only difficult objects and act as background-only
   images.
5. **Resolution.** Images are at most {max_w}x{max_h} px ({n_res} distinct sizes). YOLO at
   640 px and Faster R-CNN with a 600 px short side (1000 px cap) therefore only ever upsample: no
   image detail is thrown away by the input size, and lower YOLO input sizes (Phase 6 sweep) are
   the first point where detail is lost.
6. **Aspect ratios.** 90% of boxes have width/height between {aspect_p05} and {aspect_p95}. The
   tallest classes by median box shape are {tallest}; the widest are {widest}. The detector must
   cover a wide range of box shapes.

## Tables

### Instances per class (difficult excluded)

{class_counts}

### Size buckets per class

{size_buckets}

### Difficult and truncated share per class

{flags}

## Figures

{figures}
"""


def render_markdown(summary: dict, tables: dict, figures: list[str]) -> str:
    """Fill REPORT_TEMPLATE with generated numbers. Rerun eda.py to refresh EDA.md."""
    s = summary
    pct = "{:.1%}".format
    return REPORT_TEMPLATE.format(
        n_train=s["images"]["train"],
        n_val=s["images"]["val"],
        n_test=s["images"]["test"],
        obj_train=s["objects_non_difficult"]["train"],
        obj_val=s["objects_non_difficult"]["val"],
        top_class=s["top_class"],
        top_class_share=pct(s["top_class_share"]),
        imbalance_ratio=f"{s['imbalance_ratio']:.1f}",
        rarest_class=s["rarest_class"],
        small_share=pct(s["small_share_overall"]),
        smallest_classes=", ".join(f"`{c}`" for c in s["smallest_classes"]),
        opi_median=f"{s['objects_per_image_median']:.0f}",
        single_share=pct(s["single_object_image_share"]),
        opi_max=s["objects_per_image_max"],
        difficult_share=pct(s["difficult_share"]),
        truncated_share=pct(s["truncated_share"]),
        only_difficult=s["images_only_difficult"],
        max_w=s["max_width"],
        max_h=s["max_height"],
        n_res=s["distinct_resolutions"],
        aspect_p05=f"{s['aspect_p05']:.2f}",
        aspect_p95=f"{s['aspect_p95']:.2f}",
        tallest=", ".join(f"`{c}`" for c in s["tallest_classes"]),
        widest=", ".join(f"`{c}`" for c in s["widest_classes"]),
        class_counts=tables["class_counts"].to_markdown(floatfmt=".3f"),
        size_buckets=tables["size_buckets"].to_markdown(floatfmt=".3f"),
        flags=tables["flags"].to_markdown(floatfmt=".3f"),
        figures="\n".join(f"![{name}](figures/{name})" for name in figures),
    )


def run_eda(
    annotations: pd.DataFrame,
    splits: dict[str, list[str]],
    voc_dir: Path,
    results_dir: Path,
    report_path: Path,
) -> dict:
    """Compute tables and figures, write results/eda/* and reports/EDA.md; return the summary."""
    df = add_box_columns(annotations, splits)
    tables = compute_tables(df, splits)
    summary = compute_summary(df, tables, splits)

    results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_counts", "size_buckets", "flags", "objects_per_image", "resolutions"):
        tables[name].to_csv(results_dir / f"{name}.csv")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figures = save_figures(df, tables, voc_dir, report_path.parent / "figures")
    report_path.write_text(render_markdown(summary, tables, figures))
    logger.info("Wrote %s, %d figures, and tables in %s", report_path, len(figures), results_dir)
    return summary
