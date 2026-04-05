"""TensorBoard helpers for training scripts (scalars + sample images)."""

from __future__ import annotations

import pathlib

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def make_writer(log_dir: str | pathlib.Path) -> SummaryWriter:
    """Create a SummaryWriter under ``log_dir`` (created if missing)."""
    p = pathlib.Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(p))


def close_writer(writer: SummaryWriter | None) -> None:
    if writer is not None:
        writer.flush()
        writer.close()


def add_scalars(writer: SummaryWriter | None, scalars: dict[str, object], step: int, prefix: str = "train") -> None:
    """Log numeric scalars; skips non-numeric values and the key ``step``."""
    if writer is None:
        return
    for key, value in scalars.items():
        if key == "step":
            continue
        try:
            v = float(np.asarray(value))
        except (TypeError, ValueError):
            continue
        writer.add_scalar(f"{prefix}/{key}", v, step)


def add_image_hwc(
    writer: SummaryWriter | None,
    image_hwc: np.ndarray,
    step: int,
    *,
    tag: str = "train/camera_views",
) -> None:
    """Log a single H×W×C image (uint8 or float); normalizes uint8-style to [0, 1]."""
    if writer is None:
        return
    x = np.asarray(image_hwc, dtype=np.float32)
    if x.size == 0:
        return
    if x.max() > 1.0:
        x = x / 255.0
    writer.add_image(tag, x, step, dataformats="HWC")
