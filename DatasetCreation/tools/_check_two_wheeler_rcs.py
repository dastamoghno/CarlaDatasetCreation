"""Quick QA: verify two_wheeler rows have RCS and Doppler assigned."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent.parent / "Data" /
    "sensor_capture_20260624_021149" / "radar_data_labeled.csv"
)

stats = defaultdict(lambda: {"rcs": [], "vel": [], "no_rcs": 0, "n": 0})

with csv_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        kind = row.get("matched_actor_kind", "").strip()
        if not kind:
            continue
        klass = row.get("matched_actor_class", "").strip()
        key = kind if kind != "two_wheeler" else f"two_wheeler/{klass}"
        s = stats[key]
        s["n"] += 1
        rcs = row.get("rcs_dBsm", "").strip()
        vel = row.get("velocity_mps", "").strip()
        if rcs:
            try:
                s["rcs"].append(float(rcs))
            except ValueError:
                pass
        else:
            s["no_rcs"] += 1
        if vel:
            try:
                s["vel"].append(float(vel))
            except ValueError:
                pass

HDR = f"{'Kind':<30} {'N':>7}  {'RCS%':>6}  {'RCS median':>11}  {'Vel median':>11}  {'Moving%':>8}"
print(HDR)
print("-" * len(HDR))

for k in sorted(stats):
    s = stats[k]
    n = s["n"]
    r = s["rcs"]
    v = s["vel"]
    rcs_pct = f"{100*len(r)//n}%" if n else "N/A"
    rcs_med = f"{statistics.median(r):+.2f} dBsm" if r else "     N/A"
    vel_med = f"{statistics.median(v):+.3f} m/s" if v else "     N/A"
    moving = sum(1 for x in v if abs(x) > 0.1)
    mov_pct = f"{100*moving//len(v)}%" if v else "N/A"
    print(f"{k:<30} {n:>7}  {rcs_pct:>6}  {rcs_med:>11}  {vel_med:>11}  {mov_pct:>8}")

print()
# Spot-check: print a few two_wheeler rows
print("--- Sample two_wheeler rows (first 5 matched) ---")
with csv_path.open(newline="", encoding="utf-8") as f:
    shown = 0
    for row in csv.DictReader(f):
        if row.get("matched_actor_kind", "").strip() == "two_wheeler":
            print(
                f"  class={row['matched_actor_class']:<12} "
                f"rcs_proxy={row.get('rcs_proxy_m2',''):<8} "
                f"rcs_dBsm={row.get('rcs_dBsm',''):<10} "
                f"velocity_mps={row.get('velocity_mps',''):<10} "
                f"snr_dB={row.get('snr_dB',''):<8} "
                f"visible={row.get('visible','')}"
            )
            shown += 1
            if shown >= 5:
                break
