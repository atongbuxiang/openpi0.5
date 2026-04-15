#!/usr/bin/env python3
"""Add an `action` field to a LeRobot v2.1 dataset.

This script copies `observation.state` into a new `action` field for:
- every episode parquet file under `data/`
- `meta/info.json`
- `meta/stats.json`
- `meta/episodes_stats.jsonl`

Example:
    python scripts/add_action_from_state.py data/fold_clothes_merged
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 `pyarrow`，请先安装后再运行，例如：`pip install pyarrow`。"
    ) from exc


INFO_PATH = Path("meta/info.json")
STATS_PATH = Path("meta/stats.json")
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
DATA_DIR = Path("data")
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="LeRobot 数据集根目录，例如 data/fold_clothes_merged",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果已经存在 action，则强制覆盖为 observation.state。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def insert_or_replace_after(
    mapping: dict[str, Any],
    *,
    after_key: str,
    new_key: str,
    new_value: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    inserted = False

    for key, value in mapping.items():
        if key == new_key:
            if not inserted:
                out[new_key] = new_value
                inserted = True
            continue

        out[key] = value
        if key == after_key:
            out[new_key] = new_value
            inserted = True

    if not inserted:
        out[new_key] = new_value

    return out


def update_info(info_path: Path) -> None:
    info = load_json(info_path)
    features = info.get("features", {})
    if STATE_KEY not in features:
        raise KeyError(f"`{info_path}` 中缺少 `{STATE_KEY}`。")

    action_feature = copy.deepcopy(features[STATE_KEY])
    info["features"] = insert_or_replace_after(
        features,
        after_key=STATE_KEY,
        new_key=ACTION_KEY,
        new_value=action_feature,
    )
    write_json(info_path, info)


def update_stats(stats_path: Path) -> None:
    stats = load_json(stats_path)
    if STATE_KEY not in stats:
        raise KeyError(f"`{stats_path}` 中缺少 `{STATE_KEY}`。")

    action_stats = copy.deepcopy(stats[STATE_KEY])
    stats = insert_or_replace_after(
        stats,
        after_key=STATE_KEY,
        new_key=ACTION_KEY,
        new_value=action_stats,
    )
    write_json(stats_path, stats)


def update_episodes_stats(episodes_stats_path: Path) -> int:
    items = load_jsonl(episodes_stats_path)
    for item in items:
        episode_index = item.get("episode_index", "<unknown>")
        episode_stats = item.get("stats")
        if not isinstance(episode_stats, dict):
            raise ValueError(f"`{episodes_stats_path}` 中 episode {episode_index} 的 `stats` 格式不正确。")
        if STATE_KEY not in episode_stats:
            raise KeyError(f"`{episodes_stats_path}` 中 episode {episode_index} 缺少 `{STATE_KEY}`。")

        episode_stats = insert_or_replace_after(
            episode_stats,
            after_key=STATE_KEY,
            new_key=ACTION_KEY,
            new_value=copy.deepcopy(episode_stats[STATE_KEY]),
        )
        item["stats"] = episode_stats

    write_jsonl(episodes_stats_path, items)
    return len(items)


def update_parquet_file(parquet_path: Path, force: bool) -> bool:
    table = pq.read_table(parquet_path)
    column_names = table.column_names

    if STATE_KEY not in column_names:
        raise KeyError(f"`{parquet_path}` 中缺少 `{STATE_KEY}` 列。")

    state_column = table[STATE_KEY]
    state_index = column_names.index(STATE_KEY)

    if ACTION_KEY in column_names:
        if not force:
            return False
        action_index = column_names.index(ACTION_KEY)
        table = table.set_column(action_index, ACTION_KEY, state_column)
    else:
        table = table.add_column(state_index + 1, ACTION_KEY, state_column)

    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(parquet_path)
    return True


def validate_dataset_root(dataset_root: Path) -> None:
    required_paths = [
        dataset_root / INFO_PATH,
        dataset_root / STATS_PATH,
        dataset_root / EPISODES_STATS_PATH,
        dataset_root / DATA_DIR,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"以下路径不存在：\n{missing_text}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    validate_dataset_root(dataset_root)

    parquet_paths = sorted((dataset_root / DATA_DIR).glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"`{dataset_root / DATA_DIR}` 下没有找到任何 episode parquet 文件。")

    rewritten = 0
    skipped = 0
    for parquet_path in parquet_paths:
        changed = update_parquet_file(parquet_path, force=args.force)
        if changed:
            rewritten += 1
        else:
            skipped += 1

    update_info(dataset_root / INFO_PATH)
    update_stats(dataset_root / STATS_PATH)
    num_episode_stats = update_episodes_stats(dataset_root / EPISODES_STATS_PATH)

    print(f"数据集目录: {dataset_root}")
    print(f"处理 parquet: {len(parquet_paths)} 个")
    print(f"实际重写: {rewritten} 个")
    print(f"跳过已有 action: {skipped} 个")
    print("已同步更新: meta/info.json, meta/stats.json, meta/episodes_stats.jsonl")
    print(f"逐 episode stats 条目数: {num_episode_stats}")


if __name__ == "__main__":
    main()
