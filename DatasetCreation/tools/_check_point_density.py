"""Point density distribution across matched_actor_kind categories.

Reports:
  - Global share of each kind as % of total radar points
  - Per-frame statistics (median / p5 / p95 share per kind)
  - Target check: two_wheeler 6-10%, pedestrian >18%
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent.parent / "Data" /
    "sensor_capture_20260624_021149" / "radar_data_labeled.csv"
)

# Global counters
global_counts = defaultdict(int)

# Per-frame counters  {frame_id: {kind: count}}
frame_counts: dict = defaultdict(lambda: defaultdict(int))

with csv_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        frame = row.get("frame", "").strip()
        kind  = row.get("matched_actor_kind", "").strip() or "clutter"
        if kind not in ("vehicle", "pedestrian", "two_wheeler"):
            kind = "clutter"
        global_counts[kind] += 1
        if frame:
            frame_counts[frame][kind] += 1

KINDS = ["vehicle", "two_wheeler", "pedestrian", "clutter"]
total_global = sum(global_counts.values())

print("=" * 62)
print("GLOBAL point density")
print("=" * 62)
for k in KINDS:
    n = global_counts[k]
    pct = 100.0 * n / total_global if total_global else 0
    bar = "#" * int(pct)
    if k == "two_wheeler":
        flag = "  <- target 6-10%  OK" if 6 <= pct <= 10 else "  <- target 6-10%  MISS"
    elif k == "pedestrian":
        flag = "  <- target >18%   OK" if pct >= 18 else "  <- target >18%   MISS"
    else:
        flag = ""
    print(f"  {k:<12} {n:>9,}  {pct:5.1f}%  {bar:<30}{flag}")
print(f"  {'TOTAL':<12} {total_global:>9,}  100.0%")

# Per-frame shares
frame_shares: dict = defaultdict(list)
for fd in frame_counts.values():
    ft = sum(fd.values())
    if ft == 0:
        continue
    for k in KINDS:
        frame_shares[k].append(100.0 * fd.get(k, 0) / ft)

print()
print("=" * 62)
print("PER-FRAME density  (median / p5 / p95 across all frames)")
print("=" * 62)
TARGETS = {"two_wheeler": (6, 10), "pedestrian": (18, 100)}
for k in KINDS:
    sh = sorted(frame_shares[k])
    if not sh:
        continue
    n   = len(sh)
    med = statistics.median(sh)
    p5  = sh[max(0, int(0.05 * n))]
    p95 = sh[min(n - 1, int(0.95 * n))]
    lo, hi = TARGETS.get(k, (None, None))
    if lo is not None:
        status = "OK" if lo <= med <= hi else ("above" if med > hi else "below target")
        flag = f"  target {lo}-{hi}%  {status}"
    else:
        flag = ""
    print(f"  {k:<12}  median={med:5.1f}%  p5={p5:5.1f}%  p95={p95:5.1f}%{flag}")

print()
print(f"  (over {len(frame_counts):,} frames)")
