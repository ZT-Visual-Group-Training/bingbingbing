#!/usr/bin/env python3
"""对比两份 --csv 结果，输出 detect_ms 中位数、检出帧比例、有效位姿率等。

用法：
    python3 scripts/ab_compare.py results/v1.csv results/v2.csv [更多.csv ...]

CSV 列（由 main.cpp 写出）：
    frame,decode_ms,detect_ms,match_ms,pnp_ms,total_ms,
    n_bars,n_armors,pose_ok,x,y,z,yaw_deg,pitch_deg,reproj_err_px
"""
import csv
import statistics
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def median(values):
    return statistics.median(values) if values else float("nan")


def stats(rows):
    n = len(rows)
    detect = [float(r["detect_ms"]) for r in rows]
    total = [float(r["total_ms"]) for r in rows]
    bars = [float(r["n_bars"]) for r in rows]
    armors = [float(r["n_armors"]) for r in rows]
    err = [float(r["reproj_err_px"]) for r in rows if r["pose_ok"] == "1"]
    hit = sum(1 for r in rows if int(r["n_armors"]) > 0)
    pose = sum(1 for r in rows if r["pose_ok"] == "1")
    return {
        "frames": n,
        "hit_pct": 100.0 * hit / n,
        "pose_pct": 100.0 * pose / n,
        "detect_med": median(detect),
        "total_med": median(total),
        "bars_mean": sum(bars) / n,
        "armors_mean": sum(armors) / n,
        "err_med": median(err),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    results = []
    for path in sys.argv[1:]:
        rows = load(path)
        if not rows:
            print(f"{path}: 空文件，跳过")
            continue
        results.append((path.split("/")[-1], stats(rows)))

    if not results:
        return 1

    header = (f"{'file':<16} {'frames':>7} {'hit%':>7} {'pose%':>7} "
              f"{'detect_ms':>10} {'total_ms':>9} {'bars/f':>7} {'armors/f':>9} {'err_px':>7}")
    print(header)
    print("-" * len(header))
    for name, s in results:
        print(f"{name:<16} {s['frames']:>7d} {s['hit_pct']:>7.1f} {s['pose_pct']:>7.1f} "
              f"{s['detect_med']:>10.2f} {s['total_med']:>9.2f} "
              f"{s['bars_mean']:>7.2f} {s['armors_mean']:>9.2f} {s['err_med']:>7.2f}")

    # 和第一份比，算变化量，方便直接写成"优化前后对比"
    if len(results) >= 2:
        base = results[0][1]
        print("\n相对第一份的变化：")
        for name, s in results[1:]:
            d_detect = (s["detect_med"] - base["detect_med"]) / base["detect_med"] * 100.0
            print(f"  {name:<16} detect {d_detect:+6.1f}%   "
                  f"hit {s['hit_pct'] - base['hit_pct']:+5.1f}pt   "
                  f"pose {s['pose_pct'] - base['pose_pct']:+5.1f}pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
