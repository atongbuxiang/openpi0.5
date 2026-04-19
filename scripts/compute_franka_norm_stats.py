"""Compute normalization statistics for a local LeRobot 2.1 Franka dataset.

This script is intended for local datasets under ``data/`` that follow the LeRobot
v2.1 directory layout, e.g.:

    data/<dataset_name>/
      data/chunk-000/episode_000000.parquet
      meta/info.json
      meta/tasks.jsonl
      videos/...

It computes normalization statistics for the fields that are actually consumed by the
Franka training pipeline:

- ``state``: copied from ``observation.state`` (shape ``[8]``)
- ``actions``: built as action chunks of shape ``[action_horizon, 8]`` using the same
  episode-boundary clamping behavior as LeRobot. If ``use_delta_joint_actions=True``,
  the first 7 joint dimensions are converted to deltas relative to the current state,
  while the last gripper dimension remains absolute.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import tqdm
import tyro

import openpi.shared.normalize as normalize

STATE_DIM = 8


@dataclasses.dataclass(frozen=True)
class Options:
    dataset_root: str = "data/pick_apple_4_18"
    output_dir: str | None = None
    action_horizon: int = 32
    use_delta_joint_actions: bool = True
    stat_keys: tuple[str, ...] = ("state", "actions")


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find dataset metadata at: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _get_episode_parquets(dataset_root: Path) -> list[Path]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files found under: {dataset_root / 'data'}")
    return parquet_paths


def _validate_dataset(info: dict) -> None:
    features = info.get("features", {})
    for key in ("action", "observation.state"):
        if key not in features:
            raise KeyError(f"Dataset is missing required feature: {key}")
        shape = tuple(features[key]["shape"])
        if shape != (STATE_DIM,):
            raise ValueError(f"Expected {key} to have shape {(STATE_DIM,)}, got {shape}")


def _make_action_chunk(actions: np.ndarray, index: int, action_horizon: int) -> np.ndarray:
    end = actions.shape[0] - 1
    indices = np.clip(np.arange(index, index + action_horizon), 0, end)
    return actions[indices]


def _to_delta_actions(state: np.ndarray, action_chunk: np.ndarray) -> np.ndarray:
    # First 7 Franka arm joints become deltas; the last gripper dimension stays absolute.
    delta_chunk = action_chunk.copy()
    delta_chunk[:, :7] -= state[:7]
    return delta_chunk


def main(opts: Options) -> None:
    dataset_path = Path(opts.dataset_root).expanduser().resolve()
    info = _load_info(dataset_path)
    _validate_dataset(info)

    if opts.action_horizon <= 0:
        raise ValueError(f"action_horizon must be > 0, got {opts.action_horizon}")

    parquet_paths = _get_episode_parquets(dataset_path)
    stats = {key: normalize.RunningStats() for key in opts.stat_keys}

    for parquet_path in tqdm.tqdm(parquet_paths, desc="Computing stats"):
        frame_table = pl.read_parquet(parquet_path, columns=["observation.state", "action"])
        states = np.asarray(frame_table["observation.state"].to_list(), dtype=np.float32)
        actions = np.asarray(frame_table["action"].to_list(), dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"{parquet_path} has invalid state shape {states.shape}, expected [T, {STATE_DIM}]")
        if actions.ndim != 2 or actions.shape[1] != STATE_DIM:
            raise ValueError(f"{parquet_path} has invalid action shape {actions.shape}, expected [T, {STATE_DIM}]")

        if "state" in stats:
            stats["state"].update(states)

        if "actions" in stats:
            action_chunks = np.stack(
                [_make_action_chunk(actions, i, opts.action_horizon) for i in range(actions.shape[0])],
                axis=0,
            )
            if opts.use_delta_joint_actions:
                action_chunks = np.stack(
                    [_to_delta_actions(state, chunk) for state, chunk in zip(states, action_chunks, strict=True)],
                    axis=0,
                )
            stats["actions"].update(action_chunks)

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}

    default_output_dir = Path("assets") / dataset_path.name
    target_dir = (
        Path(opts.output_dir).expanduser().resolve() if opts.output_dir is not None else default_output_dir.resolve()
    )
    print(f"Writing stats to: {target_dir}")
    normalize.save(target_dir, norm_stats)


if __name__ == "__main__":
    main(tyro.cli(Options))
