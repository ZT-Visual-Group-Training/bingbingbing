#!/usr/bin/env bash
# Process both assessment recordings and produce the demo videos + CSV results.
#
#   ./scripts/run_assessment.sh /path/to/data
#
# expects, under /path/to/data:
#   2026-11-13_18-25-25_462305/video/video.avi
#   2026-11-13_18-28-12_830451/video/video.avi
# i.e. the two zips unzipped in place.
set -euo pipefail

cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
OUT="results"
mkdir -p "$OUT"

declare -A CLIPS=(
  [A]="2026-11-13_18-25-25_462305"
  [B]="2026-11-13_18-28-12_830451"
)

for tag in A B; do
  dir="$DATA_ROOT/${CLIPS[$tag]}/video/video.avi"
  if [[ ! -f "$dir" ]]; then
    echo "skip $tag: $dir not found" >&2
    continue
  fi
  echo "=== clip $tag ==="
  # --out-fps 30: the source is 166 fps, so the raw clip is only ~10 s and the
  # boxes are impossible to follow. Writing at 30 fps gives a ~5.5x slow motion
  # (A: 59 s, B: 41 s) that is actually watchable in the submission video.
  ./build/autoaim \
    --video "$dir" \
    --config config/default.yaml \
    --save-video "$OUT/demo_$tag.avi" \
    --out-fps 30 \
    --csv "$OUT/result_$tag.csv" \
    --no-display \
    --verbose
done

echo
echo "results written to $OUT/"
