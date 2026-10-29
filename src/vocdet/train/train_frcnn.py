"""Our own Faster R-CNN training loop, written out step by step (ADR-012).

One iteration: set the LR, forward (the model returns its four losses in train mode), sum them,
scale and backpropagate (AMP), unscale to measure the gradient norm, optimizer step, log one row.
Every `eval_every` epochs and after the last, predict on val and evaluate with the shared evaluator;
the best checkpoint by val map50 is kept. After every epoch, last.pt holds everything needed to
resume: model, optimizer, AMP scaler, step counters, best score, and every RNG state. A resumed
run may write to a new folder (a new Kaggle session always does): best.pt and both logs are carried
over from the source run, which is the folder above last.pt.
"""

import csv
import logging
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vocdet.config import FrcnnConfig, FrcnnTrainSettings
from vocdet.data.dataset import VOCDetectionDataset, collate_fn
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate, write_metrics
from vocdet.eval.predict import predict_frcnn
from vocdet.eval.schema import GroundTruth, ground_truth_from_annotations, write_predictions
from vocdet.models.frcnn import build_frcnn
from vocdet.utils.run_meta import finish_run_meta, start_run_meta
from vocdet.utils.seed import seed_everything, worker_init_fn

logger = logging.getLogger(__name__)

LOSS_NAMES = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
TRAIN_LOG_COLUMNS = ("epoch", "step", "lr", "loss", *LOSS_NAMES, "grad_norm")
VAL_LOG_COLUMNS = ("epoch", "map50", "map50_95", "mar_100", "is_best")


def lr_at(step: int, epoch: int, cfg: FrcnnTrainSettings) -> float:
    """Learning rate for a global step in a given epoch (0-based).

    Warm-up: the LR grows linearly from lr * warmup_factor at step 0 to lr at step warmup_iters, so
    the randomly initialised head cannot wreck the pretrained features with huge early gradients.
    Then step decay: multiply by gamma once for every milestone epoch already reached (MultiStepLR).
    """
    lr = cfg.lr * cfg.gamma ** sum(1 for m in cfg.milestones if epoch >= m)
    if step < cfg.warmup_iters:
        progress = step / cfg.warmup_iters
        lr *= cfg.warmup_factor + (1 - cfg.warmup_factor) * progress
    return lr


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    step: int,
    cfg: FrcnnTrainSettings,
    log_writer: csv.writer,
) -> int:
    """Train for one pass over the loader; return the global step after the last iteration."""
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    for images, targets in loader:
        lr = lr_at(step, epoch, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets
        ]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)  # train mode: a dict of the four losses
            loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            raise FloatingPointError(
                f"Loss is {loss.item()} at epoch {epoch}, step {step}: {loss_dict}"
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()  # AMP: scale up so float16 gradients do not underflow
        scaler.unscale_(optimizer)  # back to true gradient values before measuring the norm
        # max_norm=inf: measure only, never clip. A rising norm is an early sign of divergence.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        scaler.step(optimizer)  # skips the update if AMP found inf/NaN gradients
        scaler.update()

        log_writer.writerow(
            [epoch, step, f"{lr:.6g}", f"{loss.item():.5f}"]
            + [f"{loss_dict[name].item():.5f}" for name in LOSS_NAMES]
            + [f"{grad_norm.item():.4f}"]
        )
        if step % 50 == 0:
            logger.info("epoch %d step %d lr %.5f loss %.4f", epoch, step, lr, loss.item())
        step += 1
    return step


def rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def start_log(path: Path, columns: tuple[str, ...], source: Path | None, last_epoch: int) -> None:
    """Write the log's header plus the source log's rows up to last_epoch, the checkpoint's epoch.

    For a fresh run there is no source, so the file is just the header. On resume, rows after
    last_epoch come from an epoch the dead run did not finish; that epoch is trained again, so
    keeping them would log it twice. The source may be the file itself (resume in place).
    """
    rows: list[dict] = []
    if source is not None and source.exists():
        with source.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if int(row["epoch"]) <= last_epoch]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_frcnn(
    cfg: FrcnnConfig,
    config_path: Path,
    data_dir: Path,
    out_dir: Path,
    splits_dir: Path = Path("splits"),
    resume: Path | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train, keep the best checkpoint by val map50, write the run artifacts; return its metrics.

    data_dir is a voc_prepared/ folder: annotations.parquet plus yolo/images/<split>/<id>.jpg.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg.train
    seed_everything(cfg.seed)
    meta = start_run_meta(out_dir, out_dir.name, cfg.exp_id, cfg.seed)
    shutil.copyfile(config_path, out_dir / "config.yaml")
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    annotations = pd.read_parquet(data_dir / "annotations.parquet")
    splits = load_splits(splits_dir)
    train_ids = splits["train"][: t.limit_train_images]
    val_ids = splits["val"][: t.limit_val_images]
    val_gt: list[GroundTruth] = ground_truth_from_annotations(annotations, val_ids)
    train_images = data_dir / "yolo" / "images" / "train"
    val_images = data_dir / "yolo" / "images" / "val"

    dataset = VOCDetectionDataset(annotations, train_ids, train_images, train=True)
    generator = torch.Generator().manual_seed(cfg.seed)  # drives the shuffle order
    loader = DataLoader(
        dataset,
        batch_size=t.batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    model = build_frcnn(
        pretrained=cfg.model.pretrained, min_size=cfg.model.min_size, max_size=cfg.model.max_size
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=t.amp and device.type == "cuda")

    start_epoch, step, best_map50, best_epoch = 0, 0, -1.0, -1
    source_run: Path | None = None
    if resume is not None:
        # Load to CPU: the RNG and generator states must stay CPU byte tensors, and
        # optimizer.load_state_dict moves its state to the parameters' device by itself.
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        generator.set_state(ckpt["generator"])
        restore_rng_states(ckpt["rng"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        best_map50, best_epoch = ckpt["best_map50"], ckpt["best_epoch"]
        source_run = resume.parent.parent  # <source_run>/weights/last.pt
        # best.pt sits next to the resumed last.pt; bring it along unless resuming in place.
        previous_best = resume.parent / "best.pt"
        if best_epoch >= 0 and previous_best.resolve() != (weights_dir / "best.pt").resolve():
            shutil.copyfile(previous_best, weights_dir / "best.pt")
        meta["resumed_from"] = str(resume)
        logger.info("Resumed from %s at epoch %d, step %d", resume, start_epoch, step)

    for name, columns in (("train_log.csv", TRAIN_LOG_COLUMNS), ("val_log.csv", VAL_LOG_COLUMNS)):
        source = source_run / name if source_run is not None else None
        start_log(out_dir / name, columns, source, last_epoch=start_epoch - 1)
    with (
        (out_dir / "train_log.csv").open("a", newline="") as train_f,
        (out_dir / "val_log.csv").open("a", newline="") as val_f,
    ):
        train_log = csv.writer(train_f, lineterminator="\n")
        val_log = csv.writer(val_f, lineterminator="\n")

        for epoch in range(start_epoch, t.epochs):
            step = train_one_epoch(
                model, loader, optimizer, scaler, device, epoch, step, t, train_log
            )
            train_f.flush()

            if (epoch + 1) % t.eval_every == 0 or epoch == t.epochs - 1:
                metrics = evaluate(predict_frcnn(model, val_ids, val_images, device), val_gt)
                is_best = metrics["map50"] > best_map50
                if is_best:
                    best_map50, best_epoch = metrics["map50"], epoch
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch}, weights_dir / "best.pt"
                    )
                val_log.writerow(
                    [epoch, metrics["map50"], metrics["map50_95"], metrics["mar_100"], is_best]
                )
                val_f.flush()
                logger.info(
                    "epoch %d val map50 %.4f (best %.4f)", epoch, metrics["map50"], best_map50
                )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "generator": generator.get_state(),
                    "rng": rng_states(),
                    "epoch": epoch,
                    "step": step,
                    "best_map50": best_map50,
                    "best_epoch": best_epoch,
                },
                weights_dir / "last.pt",
            )

    # Final artifacts come from best.pt, re-predicted, so they are correct after a resume too and
    # the saved predictions reproduce metrics_val.json exactly.
    model.load_state_dict(torch.load(weights_dir / "best.pt", map_location=device)["model"])
    predictions = predict_frcnn(model, val_ids, val_images, device)
    write_predictions(predictions, out_dir / "predictions_val.json")
    metrics = evaluate(predictions, val_gt)
    write_metrics(metrics, out_dir / "metrics_val.json")

    meta.update({"data_dir": str(data_dir), "best_epoch": best_epoch, "device": str(device)})
    meta.update({"num_train_images": len(train_ids), "num_val_images": len(val_ids)})
    finish_run_meta(out_dir, meta)
    return metrics
