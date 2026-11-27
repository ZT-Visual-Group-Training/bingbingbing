#!/usr/bin/env bash
# ROI / 卡尔曼的开关对比：同一段视频跑三种模式，导出 CSV 后统计。
#
#   ./scripts/run_ab.sh data/2026-11-13_18-25-25_462305/video/video.avi
#
# 三种模式：
#   v1  全图搜索（基线）
#   v2  加 ROI（只在预测位置附近找）
#   v3  ROI + 卡尔曼（丢帧用预测续窗）
set -euo pipefail

cd "$(dirname "$0")/.."
VIDEO="${1:-data/2026-11-13_18-25-25_462305/video/video.avi}"
OUT="results"
mkdir -p "$OUT"

if [[ ! -f "$VIDEO" ]]; then
  echo "找不到视频: $VIDEO" >&2
  exit 2
fi
if [[ ! -x build/autoaim ]]; then
  echo "先编译: ./scripts/build.sh" >&2
  exit 2
fi

echo "=== v1 基线（全图，无跟踪） ==="
./build/autoaim --video "$VIDEO" --no-display --csv "$OUT/v1.csv"

echo
echo "=== v2 加 ROI ==="
./build/autoaim --video "$VIDEO" --no-display --roi --csv "$OUT/v2.csv"

echo
echo "=== v3 加 ROI + 卡尔曼 ==="
./build/autoaim --video "$VIDEO" --no-display --roi --tracker --csv "$OUT/v3.csv"

echo
echo "=== 对比 ==="
python3 scripts/ab_compare.py "$OUT/v1.csv" "$OUT/v2.csv" "$OUT/v3.csv"
