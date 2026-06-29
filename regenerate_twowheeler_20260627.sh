#!/usr/bin/env bash
# Regenerate radar_data_labeled.csv for the 14 new captures with the RadarScenes-calibrated
# two-wheeler micro-Doppler (PostProcessDataset.py, sigma bicycle 3.8 / moto 4.3 = defaults).
#
# Safety:
#  - backs up the ORIGINAL csv  -> radar_data_labeled.csv.pretwfix          (cp, kept once)
#  - backs up the BEFORE-FIX npz -> *.carla_noisy1_vis1_v2.npz.pretwfix      (mv = free rename)
#    and removes the live npz so it rebuilds from the NEW csv at next training.
#  - uses each capture's ORIGINAL postprocess_seed (run_meta.json) so EVERYTHING except
#    two-wheeler velocity stays byte-identical (RCS recomputed from rcs_proxy, ped/FMCW replay).
#  - on PostProcessDataset failure, RESTORES the csv from .pretwfix (no half-written file).
#  - skips run 7 (072419, dropped).
set -uo pipefail   # NOT -e: keep going across captures
cd /home/tamoghnd/Radar_PIM/Radar-Detection-with-Deep-Learning/CarlaDatasetCreation
export PYTHONPATH="$PWD"
PY=/home/tamoghnd/miniconda3/envs/radarscenes_segmentation/bin/python
PP=DatasetCreation/capture/PostProcessDataset.py
NPZ=radar_data_labeled.carla_noisy1_vis1_v2.npz
DROP="072419"

echo "=== /scratch free space (need ~11GB for the 14 csv backups) ==="
df -h /scratch/tamoghnd 2>/dev/null | tail -1

ok=0; fail=0
for d in $(ls -d /scratch/tamoghnd/dataset_captures/sensor_capture_20260627_* | sort); do
  base=$(basename "$d")
  case "$base" in *"$DROP"*) echo "[skip] $base (dropped run 7)"; continue;; esac
  csv="$d/radar_data_labeled.csv"
  [ -f "$csv" ] || { echo "[FAIL] $base: no radar_data_labeled.csv"; fail=$((fail+1)); continue; }
  seed=$($PY -c "import json;print(json.load(open('$d/run_meta.json'))['postprocess_seed'])" 2>/dev/null)
  [ -n "${seed:-}" ] || { echo "[FAIL] $base: could not read postprocess_seed"; fail=$((fail+1)); continue; }

  # 1) back up original csv once (never overwrite an existing backup with a regenerated csv)
  [ -f "$csv.pretwfix" ] || cp -p "$csv" "$csv.pretwfix"
  # 2) back up before-fix npz once (mv = free), then ensure no live npz (rebuilds from new csv)
  if [ -f "$d/$NPZ" ] && [ ! -f "$d/$NPZ.pretwfix" ]; then mv "$d/$NPZ" "$d/$NPZ.pretwfix"; else rm -f "$d/$NPZ"; fi

  echo "[$(date '+%H:%M:%S')] regenerating $base  (seed=$seed) ..."
  if $PY "$PP" --capture-dir "$d" --seed "$seed" > "$d/.twfix_postprocess.log" 2>&1; then
    echo "[ok]   $base"
    ok=$((ok+1))
  else
    echo "[FAIL] $base — PostProcessDataset errored (see $d/.twfix_postprocess.log). Restoring csv from backup."
    cp -p "$csv.pretwfix" "$csv"
    fail=$((fail+1))
  fi
done

echo "=== REGEN DONE: ok=$ok  fail=$fail ==="
echo "backups: radar_data_labeled.csv.pretwfix (+ *.npz.pretwfix) per capture; live npz removed -> rebuilds at next training"

# 3) verification on one bike-heavy capture: bicycle velocity should now be non-zero with spread
echo "=== verify (082109): bicycle velocity_mps_noisy BEFORE(.pretwfix) vs AFTER ==="
$PY - <<'PYEOF'
import pandas as pd, numpy as np
d="/scratch/tamoghnd/dataset_captures/sensor_capture_20260627_082109"
cols=["matched_actor_class","velocity_mps_noisy"]
for tag,path in [("BEFORE",d+"/radar_data_labeled.csv.pretwfix"),("AFTER",d+"/radar_data_labeled.csv")]:
    try:
        df=pd.read_csv(path,usecols=cols)
        v=df.loc[df["matched_actor_class"]=="bicycle","velocity_mps_noisy"].to_numpy()
        print(f"  {tag}: n_bike={len(v)}  |v| p50={np.median(np.abs(v)):.3f}  std={np.std(v):.3f}")
    except Exception as e:
        print(f"  {tag}: {e}")
PYEOF
