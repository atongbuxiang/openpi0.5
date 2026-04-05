#!/bin/bash

# ======== 配置部分 ========
SRC_DIR="/root/data0/openpi0.5/checkpoints/pi05_dobot_multi_data/cyt_test/49999"              # 原始 checkpoint 目录（如 ~/openpi/checkpoints/49999）
DST_DIR="/root/data0/openpi0.5/checkpoints/pi05_dobot_multi_data/cyt_test_min/49999_min"              # 目标目录（如 ~/openpi_min/49999）

# 如果未提供参数，则默认路径
if [ -z "$SRC_DIR" ]; then
    echo "Usage: $0 <source_checkpoint_dir> <target_dir>"
    echo "Example: $0 ~/openpi/checkpoints/49999 ~/openpi_min/49999"
    exit 1
fi

echo "🚀 正在从 $SRC_DIR 提取最小 checkpoint 到 $DST_DIR"

# ======== 创建目标目录结构 ========
mkdir -p "$DST_DIR"

# 1. 拷贝 norm_stats.json（如存在）
NORM_STATS_SRC=$(find "$SRC_DIR/assets" -name "norm_stats.json" 2>/dev/null)
if [ -f "$NORM_STATS_SRC" ]; then
    NORM_STATS_DST="$DST_DIR${NORM_STATS_SRC#$SRC_DIR}"
    mkdir -p "$(dirname "$NORM_STATS_DST")"
    cp "$NORM_STATS_SRC" "$NORM_STATS_DST"
    echo "✅ 已复制 norm_stats.json 到 $NORM_STATS_DST"
else
    echo "⚠️  未找到 norm_stats.json，可能不是必须文件"
fi

# 2. 拷贝 params 目录（包含权重和分片）
if [ -d "$SRC_DIR/params" ]; then
    cp -r "$SRC_DIR/params" "$DST_DIR/"
    echo "✅ 已复制 params/"
else
    echo "❌ 找不到 params/ 目录，请检查源路径"
    exit 1
fi

echo "✅ 最小 checkpoint 抽取完成：$DST_DIR"
