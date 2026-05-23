# CARLA Radar & Camera Dataset Creation

This repository is a **CARLA 0.9.16** install plus Python tools under `Scripts/DatasetCreation/` that:

- Place radar and camera sensors on a fixed layout (4, 8, 12, or 14 radars)
- Spawn traffic, pedestrians, and controlled traffic lights
- Record synchronized radar detections and camera frames
- Export sensor extrinsics and radar labeling QA reports

The main entry point is **`Start.py`**, which launches the full script stack for you.

---

## Requirements

| Item | Notes |
|------|--------|
| **OS** | Windows 10/11 (scripts are tested on Windows; `msvcrt` is used for keyboard input) |
| **GPU** | Discrete GPU recommended for CARLA |
| **CARLA** | **0.9.16** (must match the Python `carla` package version) |
| **Python** | **3.10–3.12** (this project uses a `.venv` with Python 3.12) |
| **Map** | **`Town10HD_Opt`** — sensor positions and traffic logic assume this map |

---

## First-time setup

### 1. Install CARLA (if you do not have it yet)

**Option A — Use this folder (recommended if you cloned or copied `CARLA_Latest`)**

If `CarlaUE4.exe` exists in the project root, you already have the simulator binaries. Skip to step 2.

**Option B — Download CARLA 0.9.16**

1. Download the **Windows** package for CARLA **0.9.16** from the official release page:  
   [https://github.com/carla-simulator/carla/releases/tag/0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16)
2. Extract the archive to a folder of your choice (for example `C:\CARLA_0.9.16`).
3. Copy or merge this repo’s `Scripts/DatasetCreation/` folder into that install, **or** clone this repo and replace its `CarlaUE4.exe` / `Engine` / `CarlaUE4` content with the extracted CARLA package so versions stay aligned.

> **Version match:** The Python API wheel must be the same CARLA version as the simulator. This project targets **0.9.16**.

---

### 2. Python virtual environment

Open PowerShell in the **project root** (where `CarlaUE4.exe` lives):

```powershell
cd C:\path\to\CARLA_Latest

# Create venv (once)
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Dataset scripts dependencies
pip install -r Scripts\DatasetCreation\requirements.txt

# CARLA Python API (must match simulator version)
pip install carla==0.9.16
```

If activation is blocked by execution policy, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

### 3. Verify CARLA and Python can talk

**Terminal 1 — start the simulator:**

```powershell
cd C:\path\to\CARLA_Latest
.\CarlaUE4.exe
```

Wait until the CARLA window is open and the server is listening (default port **2000**).

**Terminal 2 — test the API:**

```powershell
cd C:\path\to\CARLA_Latest
.\.venv\Scripts\Activate.ps1
python PythonAPI\util\test_connection.py
```

You should see a successful connection message. If it times out, confirm `CarlaUE4.exe` is running and no firewall is blocking port 2000.

---

### 4. Load the correct map

Before running dataset scripts, load **`Town10HD_Opt`** in the simulator.

**In the CARLA editor / spectator UI:** use the map selector to open `Town10HD_Opt`.

**Or from Python** (with CARLA running):

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(30); c.load_world('Town10HD_Opt'); print(c.get_world().get_map().name)"
```

Radar/camera transforms and traffic-light behavior are tuned for this map. Other maps may run but positions and QA results will not match the bundled extrinsic reference files.

---

## Quick start (typical workflow)

You need **two terminals**: one for the simulator, one for the dataset pipeline.

### Terminal 1 — CARLA server

```powershell
cd C:\path\to\CARLA_Latest
.\CarlaUE4.exe
```

Load **`Town10HD_Opt`**, then leave CARLA running.

### Terminal 2 — dataset pipeline

```powershell
cd C:\path\to\CARLA_Latest
.\.venv\Scripts\Activate.ps1
cd Scripts\DatasetCreation
python Start.py
```

`Start.py` will ask:

1. **Radar count** — `4`, `8`, `12`, or `14` (menu `1`–`3` map to 4, 8, 12; type `14` for fourteen radars)
2. **Run mode**
   - **Full dataset** — spawns sensors, traffic, lights, then runs `CaptureRadarCameraData.py`
   - **Test radar labeling** — shorter run with `TestRadarLabeling.py` and live stats (no full capture CSV)

Non-interactive example:

```powershell
python Start.py --radar-count 14
python Start.py --radar-count 12 --test-labeling
```

### Stop the pipeline

| Mode | How to stop | What happens |
|------|-------------|--------------|
| **Full dataset** | **Ctrl+C** in the `Start.py` terminal | Exports camera + radar extrinsics into the active `sensor_capture_*` folder, stops child scripts, despawns dataset vehicles |
| **Test labeling** | **Ctrl+C** or **Enter** (depending on which script has focus) | Requests `TestRadarLabeling.py` to finish reports (plots, `summary.txt`, CSVs) |
| **Capture only** (`CaptureRadarCameraData.py` alone) | **Enter** in that script’s window | Closes CSVs, writes extrinsics and `radar_labeling_qa/` into the capture folder |

Keep **CARLA running** until extrinsic export messages finish.

---

## What `Start.py` launches

### Full dataset mode (default)

Rough order:

1. `RadarCameraSetup{4,8,12,14}.py` — spawns tagged `dataset_radar_*` and `dataset_camera_*` sensors
2. `ClearParkedCarsAndMotorcycles.py` / `ClearTrashCansAndMailboxes.py`
3. `SpawnCarsAtPosition14.py` — free-roaming traffic near the sensor corridor
4. `SpawnPedestriansAcrossMap.py` — navmesh pedestrians (default **30**)
5. `TrafficLightSetup.py` + `TrafficLightControl.py` (GUI in a separate console on Windows)
6. `CaptureRadarCameraData.py` — records data until you stop the stack

### Test labeling mode (`--test-labeling`)

Setup + traffic + `TestRadarLabeling.py` only. Outputs go to `radar_labeling_test_YYYYMMDD_HHMMSS/` with live stats in the console every ~2 seconds.

### Not auto-started (run manually if needed)

| Script | Purpose |
|--------|---------|
| `TestRadarLabeling.py` | Standalone labeling validation |
| `PrintRadarLayoutExtrinsics.py` | Print/export radar layout poses (needs CARLA + map) |
| `ExportRadarExtrinsics.py` / `ExportCameraExtrinsics.py` | Extrinsics export utilities |
| `FetchActorSizing.py` | Build actor size catalog |
| `DespawnAllCars.py` | Remove spawned vehicles |

---

## Output folders

All capture output is written under `Scripts/DatasetCreation/` unless `DATASET_CAPTURE_BASE_DIR` is set.

### Full capture — `sensor_capture_YYYYMMDD_HHMMSS/`

| File / folder | Description |
|---------------|-------------|
| `radar_data.csv` | Radar detections, poses, labeling fields |
| `camera_data.csv` | Camera frame metadata |
| `camera_frames/` | Saved RGB images |
| `camera_extrinsics.csv` / `.json` | Camera poses in world frame |
| `radar_extrinsics.csv` / `.json` or `sensor_extrinsics.*` | Radar poses |
| `radar_labeling_qa/` | Per-frame and per-sensor labeling QA CSVs |

The active run is tracked in `.last_dataset_capture_dir`.

### Test run — `radar_labeling_test_YYYYMMDD_HHMMSS/`

Includes `summary.txt`, `summary.json`, plots (`radar_labeling_summary.png`, etc.), and diagnostic CSVs. Tracked in `.last_radar_labeling_test_dir`.

---

## Environment variables (optional)

Set these before `Start.py` or individual scripts if you need to override defaults:

| Variable | Default | Effect |
|----------|---------|--------|
| `DATASET_EXPECTED_RADAR_COUNT` | Set by `Start.py` | Number of radar labels (`R1`…`Rn`) |
| `DATASET_PEDESTRIAN_COUNT` | `30` | Pedestrians to spawn |
| `DATASET_CAPTURE_BASE_DIR` | `Scripts/DatasetCreation` | Parent directory for `sensor_capture_*` folders |
| `DATASET_KEEP_SENSORS_RUNNING` | `0` (set to `1` in test mode) | Keeps radar setup alive when sharing a console |
| `DATASET_AUTOMATIC_TRAFFIC_LIGHTS` | `0` | `1` = unfreeze all lights; `0` = perimeter cycling (recommended) |

---

## Troubleshooting

**`Could not connect to CARLA` / connection timeout**

- Start `CarlaUE4.exe` first and wait for the main window.
- Confirm nothing else is using port **2000**.
- Run `python PythonAPI\util\test_connection.py` from the activated venv.

**`No module named 'carla'`**

- Activate `.venv` and run `pip install carla==0.9.16`.
- Use the **same** `python.exe` for all scripts (the venv under this project root).

**`No tagged dataset sensors found`**

- Run a `RadarCameraSetup*.py` script (or full `Start.py`) before `CaptureRadarCameraData.py`.
- Wait at least ~12 seconds after setup in test mode for sensors to spawn.

**Wrong map / sensors in odd places**

- Load **`Town10HD_Opt`** before starting the pipeline.

**Extrinsic files missing after stop**

- In full mode, stop with **Ctrl+C** from `Start.py` (not by killing CARLA first).
- Keep CARLA and sensor scripts alive until export messages appear.

**Traffic Light GUI does not appear**

- On Windows, `TrafficLightControl.py` opens in a **second console**; check the taskbar for another terminal window.

---

## Project layout (short)

```
CARLA_Latest/
├── CarlaUE4.exe          # Start this first
├── PythonAPI/            # Upstream CARLA examples and utilities
├── Scripts/
│   └── DatasetCreation/  # Pipeline scripts (Start.py lives here)
│       ├── Start.py
│       ├── CaptureRadarCameraData.py
│       ├── RadarCameraSetup{4,8,12,14}.py
│       └── requirements.txt
└── .venv/                # Python environment (create locally)
```

---

## Further reading

- Upstream CARLA docs: [https://carla.readthedocs.io](https://carla.readthedocs.io)
- Stock simulator readme: [README](README)
- CARLA 0.9.16 release notes: [CHANGELOG](CHANGELOG)
