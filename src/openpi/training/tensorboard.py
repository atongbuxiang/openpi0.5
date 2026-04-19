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


def make_camera_views_image(
    images: dict[str, np.ndarray],
    *,
    batch_index: int = 0,
    max_history_frames: int | None = None,
) -> np.ndarray:
    """Create a single HWC image for logging camera views.

    Supports camera tensors shaped either `[B, H, W, C]` or `[B, T, H, W, C]`.
    For video inputs, frames are tiled horizontally per camera, and cameras are
    then stacked vertically to preserve both the history and camera identity.
    """
    camera_rows = []
    for image in images.values():
        x = np.asarray(image[batch_index])
        if x.ndim == 3:
            camera_rows.append(x)
            continue
        if x.ndim == 4:
            if max_history_frames is not None and x.shape[0] > max_history_frames:
                x = x[-max_history_frames:]
            camera_rows.append(np.concatenate([frame for frame in x], axis=1))
            continue
        raise ValueError(f"Expected image sample to have ndim 3 or 4, got {x.shape}")

    if not camera_rows:
        raise ValueError("No images provided for visualization.")

    max_width = max(row.shape[1] for row in camera_rows)
    padded_rows = []
    for row in camera_rows:
        if row.shape[1] < max_width:
            pad_width = max_width - row.shape[1]
            row = np.pad(row, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
        padded_rows.append(row)
    return np.concatenate(padded_rows, axis=0)
