"""Compute normalization statistics for a local LeRobot v2.1 dataset.

This script is intended for local datasets under ``data/`` that follow the LeRobot
v2.1 directory layout, e.g.:

    data/<dataset_name>/
      data/chunk-000/episode_000000.parquet
      meta/info.json
      meta/tasks.jsonl
      videos/...

Compared to the original Franka-only version, this script works for generic
LeRobot datasets by:

- auto-detecting the state/action feature keys and dimensions from ``meta/info.json``
- letting you choose whether action chunks should stay absolute or be converted to
  delta actions with respect to the current state
- supporting a configurable delta-action mask instead of hard-coding the Franka
  convention

By default, the script computes stats for the fields consumed by OpenPI training:

- ``state``: copied from the configured state feature (default: ``observation.state``)
- ``actions``: action chunks with shape ``[action_horizon, action_dim]`` using the
  same episode-boundary clamping behavior as LeRobot
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tqdm
import tyro

import openpi.shared.normalize as normalize


def _normalize_mask(mask: SequenceLike | None) -> tuple[bool, ...] | None:
    if mask is None:
        return None
    return tuple(bool(v) for v in mask)


SequenceLike = tuple[bool, ...] | list[bool]


@dataclasses.dataclass(frozen=True)
class Options:
    dataset_root: str = "data/fold_all"
    output_dir: str | None = None
    action_horizon: int = 32
    state_key: str = "observation.state"
    action_key: str = "action"
    use_delta_actions: bool | None = None
    delta_action_mask: tuple[bool, ...] | None = None
    stat_keys: tuple[str, ...] = ("state", "actions")


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find dataset metadata at: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _get_episode_parquets(dataset_root: Path) -> list[Path]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files found under: {dataset_root / 'data'}")
    return parquet_paths


def _get_feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...]:
    features = info.get("features", {})
    if key not in features:
        available = ", ".join(sorted(features))
        raise KeyError(f"Dataset is missing required feature {key!r}. Available features: {available}")
    shape = tuple(features[key].get("shape", ()))
    if not shape:
        raise ValueError(f"Feature {key!r} must have a non-empty shape, got {shape}")
    return shape


def _get_feature_dim(info: dict[str, Any], key: str) -> int:
    shape = _get_feature_shape(info, key)
    if len(shape) != 1:
        raise ValueError(f"Feature {key!r} must be 1D, got shape {shape}")
    return shape[0]


def _infer_use_delta_actions(info: dict[str, Any], action_key: str, explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit

    features = info.get("features", {})
    action_feature = features.get(action_key, {})
    names = action_feature.get("names") or []
    robot_type = str(info.get("robot_type", "")).lower()

    # Heuristic: velocity / delta style actions should stay as-is.
    if "droid" in robot_type:
        return False
    lowered_names = [str(name).lower() for name in names if name is not None]
    if any(".vel" in name or name.endswith("_dq") or "velocity" in name for name in lowered_names):
        return False

    return True


def _build_default_delta_mask(info: dict[str, Any], action_key: str, state_dim: int, action_dim: int) -> tuple[bool, ...]:
    if state_dim != action_dim:
        raise ValueError(
            "Automatic delta-action mask inference requires matching state/action dims, "
            f"got state_dim={state_dim}, action_dim={action_dim}. Pass --delta-action-mask explicitly."
        )

    features = info.get("features", {})
    action_feature = features.get(action_key, {})
    names = list(action_feature.get("names") or [])
    robot_type = str(info.get("robot_type", "")).lower()

    if "franka" in robot_type and action_dim == 8:
        return tuple([True] * 7 + [False])
    if "aloha" in robot_type and action_dim == 14:
        return tuple([True] * 6 + [False] + [True] * 6 + [False])
    if "pnd" in robot_type and action_dim == 21:
        return tuple([True] * 19 + [False] * 2)

    if names and all(isinstance(name, str) for name in names):
        lowered_names = [name.lower() for name in names]
        return tuple(not ("gripper" in name or "hand_" in name or name.endswith(".pos") and "gripper" in name) for name in lowered_names)

    # Conservative fallback: all but the last action dim are treated as delta dims.
    if action_dim == 1:
        return (False,)
    return tuple([True] * (action_dim - 1) + [False])


def _resolve_delta_action_mask(
    info: dict[str, Any],
    action_key: str,
    state_dim: int,
    action_dim: int,
    explicit_mask: tuple[bool, ...] | None,
) -> tuple[bool, ...]:
    mask = explicit_mask if explicit_mask is not None else _build_default_delta_mask(
        info, action_key, state_dim, action_dim
    )
    if len(mask) > min(state_dim, action_dim):
        raise ValueError(
            f"delta_action_mask length ({len(mask)}) exceeds compatible dims "
            f"(min(state_dim, action_dim)={min(state_dim, action_dim)})"
        )
    return mask


def _make_action_chunk(actions: np.ndarray, index: int, action_horizon: int) -> np.ndarray:
    end = actions.shape[0] - 1
    indices = np.clip(np.arange(index, index + action_horizon), 0, end)
    return actions[indices]


def _to_delta_actions(state: np.ndarray, action_chunk: np.ndarray, mask: tuple[bool, ...]) -> np.ndarray:
    delta_chunk = action_chunk.copy()
    mask_array = np.asarray(mask, dtype=bool)
    dims = mask_array.shape[0]
    delta_chunk[:, :dims] -= np.where(mask_array, state[:dims], 0.0)
    return delta_chunk


def main(opts: Options) -> None:
    dataset_path = Path(opts.dataset_root).expanduser().resolve()
    info = _load_info(dataset_path)

    if opts.action_horizon <= 0:
        raise ValueError(f"action_horizon must be > 0, got {opts.action_horizon}")

    state_dim = _get_feature_dim(info, opts.state_key)
    action_dim = _get_feature_dim(info, opts.action_key)
    use_delta_actions = _infer_use_delta_actions(info, opts.action_key, opts.use_delta_actions)
    delta_action_mask = None
    if use_delta_actions:
        delta_action_mask = _resolve_delta_action_mask(
            info,
            opts.action_key,
            state_dim,
            action_dim,
            _normalize_mask(opts.delta_action_mask),
        )

    parquet_paths = _get_episode_parquets(dataset_path)
    stats = {key: normalize.RunningStats() for key in opts.stat_keys}

    for parquet_path in tqdm.tqdm(parquet_paths, desc="Computing stats"):
        frame_table = pl.read_parquet(parquet_path, columns=[opts.state_key, opts.action_key])
        states = np.asarray(frame_table[opts.state_key].to_list(), dtype=np.float32)
        actions = np.asarray(frame_table[opts.action_key].to_list(), dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != state_dim:
            raise ValueError(f"{parquet_path} has invalid state shape {states.shape}, expected [T, {state_dim}]")
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise ValueError(f"{parquet_path} has invalid action shape {actions.shape}, expected [T, {action_dim}]")

        if "state" in stats:
            stats["state"].update(states)

        if "actions" in stats:
            action_chunks = np.stack(
                [_make_action_chunk(actions, i, opts.action_horizon) for i in range(actions.shape[0])],
                axis=0,
            )
            if use_delta_actions and delta_action_mask is not None:
                action_chunks = np.stack(
                    [
                        _to_delta_actions(state, chunk, delta_action_mask)
                        for state, chunk in zip(states, action_chunks, strict=True)
                    ],
                    axis=0,
                )
            stats["actions"].update(action_chunks)

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}

    default_output_dir = Path("assets") / dataset_path.name
    target_dir = (
        Path(opts.output_dir).expanduser().resolve() if opts.output_dir is not None else default_output_dir.resolve()
    )
    print(f"Dataset: {dataset_path.name}")
    print(f"State key: {opts.state_key} (dim={state_dim})")
    print(f"Action key: {opts.action_key} (dim={action_dim})")
    print(f"Use delta actions: {use_delta_actions}")
    if delta_action_mask is not None:
        print(f"Delta action mask: {delta_action_mask}")
    print(f"Writing stats to: {target_dir}")
    normalize.save(target_dir, norm_stats)


if __name__ == "__main__":
    main(tyro.cli(Options))
