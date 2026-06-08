# CARLA Radar & Camera Dataset Creation

Python tooling (under `DatasetCreation/`) on top of **CARLA 0.9.16** that:

- Places a fixed layout of **4 / 8 / 12 / 14 radars** plus cameras on `Town10HD_Opt`
- Spawns drive-in traffic, pedestrians, and controlled traffic lights
- Records synchronized radar detections + camera frames
- **Labels each radar return** to the actor it hit (with a data-grounded match margin)
- Exports sensor extrinsics and a radar-labeling QA report

The pipeline is driven by **`run_sim.sh`** (Linux one-shot) or **`Start.py`** (cross-platform launcher).

---

## Repository layout

```
DatasetCreation/
├── run_sim.sh                 # Linux one-shot: starts CARLA + map + Start.py
├── Start.py                   # Launcher: spawns the script stack (interactive or flags)
├── carla_connect.py           # Connection helpers (host/port/timeouts)
├── setup/                     # RadarCameraSetup{4,8,12,14}.py — sensor layouts
├── world/                     # Traffic, pedestrians, traffic lights, cleanup
│   ├── SpawnCarsAtPosition14.py
│   ├── SpawnPedestriansAcrossMap.py
│   ├── TrafficLightSetup.py / TrafficLightControl.py
│   └── Clear*.py / DespawnAllCars.py
├── capture/                   # Recording + labeling
│   ├── CaptureRadarCameraData.py   # the recorder
│   ├── LabelRadarCapture.py        # offline labeler (radar_data_labeled.csv)
│   ├── actor_frame_log.py          # per-frame actor pose log
│   └── Export{Radar,Camera}Extrinsics.py
├── testing/
│   ├── TestRadarLabeling.py        # short labeling-validation run (no full CSV)
│   └── RadarLabelingTestReport.py  # QA plots + summary + margin calibration
└── tools/
    ├── derive_match_threshold_uncensored.py  # ad-hoc margin analysis
    ├── PrintRadarLayoutExtrinsics.py
    └── FetchActorSizing.py
```

---

## Requirements

| Item | Notes |
|------|-------|
| **CARLA** | **0.9.16** — the Python `carla` wheel must match the simulator version |
| **Python** | **3.10–3.12** in a dedicated venv (this host uses `~/carla_pim_venv`, Python 3.12) |
| **Map** | **`Town10HD_Opt`** — all sensor poses and traffic logic assume this map |
| **OS** | Linux (primary; `run_sim.sh`) or Windows (run `Start.py` from an activated venv) |
| **GPU** | Discrete GPU recommended; `run_sim.sh` starts CARLA with `-RenderOffScreen` |

---

## First-time setup (Linux)

```bash
# 1. Create the venv and install deps (once)
python3.12 -m venv ~/carla_pim_venv
source ~/carla_pim_venv/bin/activate
pip install -r DatasetCreation/requirements.txt
pip install carla==0.9.16        # must match the simulator

# 2. Point run_sim.sh at your CARLA install if it isn't the default
#    (defaults: VENV=~/carla_pim_venv, CARLA_DIR=/scratch/tamoghnd/CARLA_0.9.16)
```

`run_sim.sh` activates the venv, starts the CARLA server if one isn't already
listening, loads `Town10HD_Opt`, and points captures at a scratch directory — so
on a configured host you don't manage any of that by hand.

---

## Generate your first dataset

### The fast path — `run_sim.sh`

From `DatasetCreation/`:

```bash
# Full capture, 4-radar layout (writes a sensor_capture_* folder):
./run_sim.sh --radar-count 4

# 12-radar full capture:
./run_sim.sh --radar-count 12

# Quick labeling sanity check (no full CSV, ~live stats):
./run_sim.sh --radar-count 4 --test-labeling

# No args == test labeling with 4 radars (the safe default):
./run_sim.sh
```

> **Note:** any argument you pass replaces the default `--radar-count 4 --test-labeling`,
> so `--radar-count 4` **alone** means a *full capture*, not a test run.

What it does, in order: activate venv → start CARLA (if needed) → wait for the RPC
port → load `Town10HD_Opt` → set the capture directory → run `Start.py`, which
spawns sensors, traffic, pedestrians, lights, then `CaptureRadarCameraData.py`.

**Handy `run_sim.sh` environment toggles** (prefix the command):

| Var | Default | Effect |
|-----|---------|--------|
| `KEEP_SERVER` | `0` | `1` = leave CARLA running after `Start.py` exits |
| `VENV` | `~/carla_pim_venv` | Path to the Python venv |
| `CARLA_DIR` | `/scratch/tamoghnd/CARLA_0.9.16` | CARLA install (must contain `CarlaUE4.sh`) |
| `DATASET_CAPTURE_BASE_DIR` | `/scratch/tamoghnd/dataset_captures` | Where `sensor_capture_*` folders are written |
| `RPC_PORT` / `MAP` | `2000` / `Town10HD_Opt` | CARLA RPC port / map to load |

Example: `KEEP_SERVER=1 ./run_sim.sh --radar-count 12`

### The manual path — `Start.py` (any OS)

1. Start CARLA yourself and load `Town10HD_Opt`.
2. From an activated venv, in `DatasetCreation/`:

```bash
python Start.py                       # interactive: asks radar count + mode
python Start.py --radar-count 14      # full capture, 14 radars
python Start.py --radar-count 12 --test-labeling
```

### Stopping a run

| Mode | Stop with | What happens |
|------|-----------|--------------|
| **Full capture** | **Ctrl+C** in the `Start.py` terminal | Closes CSVs, exports extrinsics, writes `radar_labeling_qa/`, runs offline labeling → `radar_data_labeled.csv`, despawns dataset vehicles |
| **Test labeling** | **Ctrl+C** / **Enter** | `TestRadarLabeling.py` finalizes plots, `summary.txt`, CSVs |

Keep CARLA alive until the extrinsic-export / labeling messages finish.

---

## Tuning the scene

Set these before launching (or `export` them; `run_sim.sh` inherits your shell env).

```bash
# More cars and people, denser radar, slower traffic:
DATASET_TARGET_CAR_COUNT=60 DATASET_PEDESTRIAN_COUNT=50 \
DATASET_RADAR_POINTS_PER_SECOND=20000 DATASET_VEHICLE_SPEED_REDUCTION_PCT=25 \
./run_sim.sh --radar-count 12
```

| Knob | Default | What it does |
|------|---------|--------------|
| `--radar-count` (CLI) | prompt | Sensor layout: `4`, `8`, `12`, or `14` |
| `DATASET_TARGET_CAR_COUNT` | `30` | Vehicles to spawn |
| `DATASET_PEDESTRIAN_COUNT` | `30` | Pedestrians to spawn |
| `DATASET_SPAWN_RADIUS_M` | `80.0` | Outer radius of the spawn annulus around the corridor |
| `DATASET_SPAWN_EXCLUSION_RADIUS_M` | `0.0` | Inner radius cars must drive *in* from (`0` = fill the zone) |
| `DATASET_VEHICLE_SPEED_REDUCTION_PCT` | `-100.0` | % slower than the speed limit. Negative = faster; `-100` ≈ 2× limit, `25` = 25% slower |
| `DATASET_FREE_VEHICLE_DRIVING` | `1` | `1` = vehicles drive (Doppler/velocity); `0` = parked |
| `DATASET_RADAR_POINTS_PER_SECOND` | `15000` | Ray density per radar (CARLA clamps at 20000). Higher = denser returns, more CPU |
| `DATASET_AUTOMATIC_TRAFFIC_LIGHTS` | `0` | `1` = all lights free-run; `0` = perimeter cycling (recommended) |

---

## Radar labeling & the match margin

After a full capture, each radar return is matched to the actor whose oriented
bounding box (OBB) it hit, producing `radar_data_labeled.csv`. A return is accepted
when its distance to the nearest actor's OBB surface (after a `0.75 m` extent
inflation) is within the **match margin**.

The defaults are **data-grounded**, not guessed:

| Param | Default | Meaning |
|-------|---------|---------|
| `DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M` | `0.5` | Primary accept margin (m). Clamped `0.5–25` |
| `DATASET_RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M` | `1.0` | Looser margin when exactly one actor is in the beam. Clamped `0.5–25` |
| `DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M` | `7.0` | Pre-filter: actors farther than this from the hit aren't candidates |
| `DATASET_LABELABLE_MIN_SPEED_MPS` | `0.0` | Skip returns slower than this when scoring (e.g. `0.5` to drop near-static clutter) |

**Why 0.5 / 1.0?** On a real capture the uncensored hit→OBB-margin distribution is a
sharp spike at ~0 (genuine on-body hits) followed by a clutter ramp with no second
lobe — so wider margins just admit road/structure clutter. On-body precision:
`0.5 m → 87%`, `1.5 m → 53%`, `2.0 m → 42%`. The trough sits at ~0.2 m; `0.5 m`
keeps a little slack.

**Every capture self-documents.** `radar_labeling_qa/radar_labeling_summary.png`
now includes a margin panel (histogram + precision-vs-threshold curve + a suggested
threshold), and the numbers are written to `summary.json` / `summary.txt`. Re-derive
per capture because the trough shifts with ray density and traffic speed.

**Let the labeler pick the margin for you** (instead of the fixed default):

| Param / flag | Default | Effect |
|--------------|---------|--------|
| `--auto-margin` / `DATASET_RADAR_AUTO_MARGIN=1` | off | Derive the margin from this capture's own distribution and apply it |
| `--auto-margin-stride` / `DATASET_RADAR_AUTO_MARGIN_STRIDE` | `20` | Sample every Nth frame during derivation (lower = more samples, slower) |

```bash
# Re-label an existing capture with the fixed defaults:
python -m capture.LabelRadarCapture --capture-dir <sensor_capture_dir>

# Re-label, deriving the margin from the capture itself:
python -m capture.LabelRadarCapture --capture-dir <sensor_capture_dir> --auto-margin

# Just analyze the margin distribution without re-labeling:
python tools/derive_match_threshold_uncensored.py <sensor_capture_dir> [frame_stride]
```

Leave `--auto-margin` off for reproducible dataset runs (a fixed, documented
threshold); turn it on for sparse-pps or high-speed captures where the trough moves.

---

## Output

Captures land in `DATASET_CAPTURE_BASE_DIR` (or `DatasetCreation/` if unset).

### Full capture — `sensor_capture_YYYYMMDD_HHMMSS/`

| File / folder | Description |
|---------------|-------------|
| `radar_data.csv` | Raw radar detections + sensor poses |
| `radar_data_labeled.csv` | Same rows + matched actor, class, bbox margin, RCS proxy, dBsm |
| `actor_frames.jsonl` | Per-frame actor poses/bboxes (drives offline labeling; no CARLA needed) |
| `camera_data.csv` / `camera_frames/` | Camera frame metadata / saved RGB images |
| `*_extrinsics.csv` / `.json` | Radar + camera poses in the world frame |
| `radar_labeling_qa/` | `radar_labeling_summary.png` (incl. margin panel), `summary.json`, `summary.txt`, per-frame/per-sensor/per-vehicle CSVs |

### Test run — `radar_labeling_test_YYYYMMDD_HHMMSS/`

`summary.txt` / `summary.json`, the QA plots, and diagnostic CSVs — but no full capture CSV.

---

## Full environment-variable reference

All optional; set before `run_sim.sh` / `Start.py` / a script. Booleans accept `1/true/yes/on`.

### Connection

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_CARLA_HOST` | `127.0.0.1` | CARLA server host |
| `DATASET_CARLA_PORT` | `2000` | CARLA RPC port |
| `DATASET_CARLA_TIMEOUT_S` | `30.0` | RPC call timeout |
| `DATASET_CARLA_READY_TIMEOUT_S` | `180.0` | How long to wait for the server to become ready |
| `DATASET_TRAFFIC_MANAGER_PORT` | CARLA default | Traffic Manager port |

### Scene & traffic

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_TARGET_CAR_COUNT` | `30` | Vehicles to spawn |
| `DATASET_PEDESTRIAN_COUNT` | `30` | Pedestrians to spawn |
| `DATASET_SPAWN_CENTER_INDEX` | `144` | Spawn-point index used as the corridor center |
| `DATASET_SPAWN_RADIUS_M` | `80.0` | Outer spawn radius |
| `DATASET_SPAWN_EXCLUSION_RADIUS_M` | `0.0` | Inner no-spawn radius (drive-in zone) |
| `DATASET_VEHICLE_SPEED_REDUCTION_PCT` | `-100.0` | % slower than limit (negative = faster) |
| `DATASET_SAFE_FOLLOWING_DISTANCE_M` | `3.0` | Traffic-manager following distance |
| `DATASET_FREE_VEHICLE_DRIVING` | `1` | `1` = drive, `0` = parked |
| `DATASET_KEEP_TRAFFIC_RUNNING` | `0` | Keep spawned vehicles after the script exits |
| `DATASET_KEEP_PEDESTRIANS_RUNNING` | `0` | Keep pedestrians after exit |
| `DATASET_KEEP_SENSORS_RUNNING` | `0` (test sets `1`) | Keep sensors alive when sharing a console |

### Traffic lights

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_AUTOMATIC_TRAFFIC_LIGHTS` | `0` | `1` = unfreeze all lights; `0` = perimeter cycling |
| `DATASET_TRAFFIC_LIGHT_GUI_AUTOCONNECT` | `0` | Auto-connect the light-control GUI |

### Radar sensor

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_EXPECTED_RADAR_COUNT` | `12` (set by `Start.py`) | Number of radar labels `R1…Rn` |
| `DATASET_RADAR_POINTS_PER_SECOND` | `15000` | Ray density per radar (clamped ≤ 20000) |
| `DATASET_RADAR_HORIZONTAL_FOV_DEG` | `120.0` | Horizontal FOV |
| `DATASET_RADAR_VERTICAL_FOV_DEG` | `60.0` | Vertical FOV |
| `DATASET_RADAR_SENSOR_TICK_S` | `0.05` | Per-sensor tick (≈20 Hz; `0` = every sim step) |
| `DATASET_RADAR_PITCH_DEG` | layout default | Per-radar downward pitch override |

### Capture pipeline

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_CAPTURE_BASE_DIR` | `DatasetCreation/` | Parent dir for `sensor_capture_*` |
| `DATASET_RADAR_CAPTURE_FAST` | `1` | Fast capture (logs `actor_frames.jsonl`, labels offline after stop) |
| `DATASET_LABEL_RADAR_AFTER_CAPTURE` | `1` | Run offline labeling automatically when capture stops |
| `DATASET_RADAR_LABEL_WORKERS` | `4` | Worker threads for in-line labeling |
| `DATASET_RADAR_PER_SENSOR_LATEST` | `0` | Keep only the latest message per sensor (drop backlog) |
| `DATASET_SYNC_MODE` | `0` | `1` = synchronous CARLA mode |
| `DATASET_SYNC_FIXED_DELTA_S` | auto | Fixed sim step when synchronous |
| `DATASET_RADAR_QUEUE_MAXSIZE` / `DATASET_RADAR_CAPTURE_DEQUE_MAX` / `DATASET_RADAR_WATCHDOG_STALE_TICKS` | tuned | Advanced queue / backpressure / stall-watchdog knobs |

### Radar labeling & match margin

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_RADAR_HIT_MATCH_MAX_MARGIN_M` | `0.5` | Primary accept margin (m), clamp `0.5–25` |
| `DATASET_RADAR_SINGLE_CANDIDATE_MAX_MARGIN_M` | `1.0` | Single-candidate fallback margin (m), clamp `0.5–25` |
| `DATASET_RADAR_CANDIDATE_HIT_MAX_BBOX_MARGIN_M` | `7.0` | Candidate pre-filter radius (m) |
| `DATASET_LABELABLE_MIN_SPEED_MPS` | `0.0` | Min `|velocity|` to score a return |
| `DATASET_RADAR_AUTO_MARGIN` | `0` | `1` = derive the margin from this capture's distribution |
| `DATASET_RADAR_AUTO_MARGIN_STRIDE` | `20` | Frame sampling stride for `--auto-margin` |

---

## Troubleshooting

**`No module named 'carla'`** — activate the venv and `pip install carla==0.9.16`; use the same Python for every script.

**`Could not connect to CARLA` / timeout** — make sure the server is up and listening on `:2000` (`ss -lntp | grep 2000`); `run_sim.sh` starts and waits for it automatically.

**`No tagged dataset sensors found`** — run a `setup/RadarCameraSetup*.py` (or full `Start.py`) before capturing; in test mode allow ~12 s for sensors to spawn.

**Wrong map / sensors in odd places** — load `Town10HD_Opt` first; the bundled extrinsics only match this map.

**Half my returns look like clutter** — open `radar_labeling_qa/radar_labeling_summary.png` and check the margin panel; lower the margin toward the trough, or re-label with `--auto-margin`.

**Extrinsics missing after stop** — stop a full run with Ctrl+C from `Start.py` (not by killing CARLA first); keep CARLA alive until export messages appear.

---

## Further reading

- CARLA docs: <https://carla.readthedocs.io>
- CARLA 0.9.16 release: <https://github.com/carla-simulator/carla/releases/tag/0.9.16>
