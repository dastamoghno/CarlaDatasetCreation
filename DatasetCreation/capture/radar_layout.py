"""Shared radar placement helpers (no imports from other capture modules)."""

import os

import carla

# Down-tilt toward road traffic. CARLA radars use positive pitch to look down
# (see PythonAPI/util/raycast_sensor_testing.py: Rotation(pitch=5) on radar mounts).
RADAR_PITCH_DEG = 8.0


def radar_pitch_deg_from_env() -> float:
    raw = os.environ.get("DATASET_RADAR_PITCH_DEG", "").strip()
    if raw:
        try:
            return max(-30.0, min(float(raw), 30.0))
        except ValueError:
            pass
    return RADAR_PITCH_DEG


# Mounting height (m) of every radar above the road. Live knob via DATASET_RIG_HEIGHT_M
# (the setup scripts apply it to each radar's z). Clamped to a sane pole range.
RADAR_HEIGHT_M = 3.0


def radar_height_m_from_env() -> float:
    raw = os.environ.get("DATASET_RIG_HEIGHT_M", "").strip()
    if raw:
        try:
            return max(0.3, min(float(raw), 12.0))
        except ValueError:
            pass
    return RADAR_HEIGHT_M


def apply_radar_pitch(radar_positions):
    """Apply shared down-tilt pitch to radar transforms (yaw and roll unchanged)."""
    pitch_deg = radar_pitch_deg_from_env()
    for name, tr in list(radar_positions.items()):
        radar_positions[name] = carla.Transform(
            tr.location,
            carla.Rotation(pitch=pitch_deg, yaw=tr.rotation.yaw, roll=tr.rotation.roll),
        )
    return radar_positions
