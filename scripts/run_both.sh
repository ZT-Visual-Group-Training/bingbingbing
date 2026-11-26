#!/usr/bin/env bash
# 一键跑完两段考核视频。
# 用法：bash scripts/run_both.sh [数据目录] [配置文件]
#   bash scripts/run_both.sh            # 默认 data/ 与 config/aegis.yaml
#   bash scripts/run_both.sh data config/aegis.yaml
set -euo pipefail

DATA_DIR="${1:-data}"
CFG="${2:-config/aegis.yaml}"
BIN="./build/aegis"

if [[ ! -x "$BIN" ]]; then
  echo "找不到 $BIN，请先构建：" >&2
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)" >&2
  exit 1
fi

FIRST="$DATA_DIR/2026-11-13_18-28-12_830451/video/video.avi"
SECOND="$DATA_DIR/2026-11-13_18-25-25_462305/video/video.avi"

for v in "$FIRST" "$SECOND"; do
  if [[ ! -f "$v" ]]; then
    echo "缺少视频：$v（请按 data/README.md 解压数据包）" >&2
    exit 1
  fi
done

echo "== 第一段：$FIRST"
"$BIN" --config "$CFG" --video "$FIRST" \
       --out demo/AegisVision_demo.mp4 \
       --csv results/first_run.csv

echo "== 第二段：$SECOND"
"$BIN" --config "$CFG" --video "$SECOND" \
       --out demo/AegisVision_demo_take2.mp4 \
       --csv results/second_run.csv

echo "完成：demo/ 与 results/ 已更新。"
