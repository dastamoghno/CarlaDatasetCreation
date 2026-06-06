"""
Spawn AI pedestrians across the map using CARLA's built-in navmesh logic.

Uses walker.pedestrian.* + controller.ai.walker (same flow as
PythonAPI/examples/generate_traffic.py). Pedestrians spawn on the navmesh
(sidewalks / crosswalks) and path to random nav targets — they stay on
walkable areas instead of cutting through buildings.

Run:
  python SpawnPedestriansAcrossMap.py
"""

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import os
import random
import sys
import time

import carla

from carla_connect import get_world


DEFAULT_PED_COUNT = 30
PED_RUNNING_FRACTION = 0.15
PED_CROSSING_FACTOR = 0.35
# Occasionally send a new nav target so pedestrians keep roaming the map.
RETARGET_INTERVAL_S = 45.0
# Default pedestrian spawn region: the monitored corridor (~(18.45, -63.0), 60 m).
# Keeps walkers in/around the radar coverage instead of scattered map-wide.
# Override via DATASET_PED_SPAWN_CENTER_X/Y + DATASET_PED_SPAWN_RADIUS_M; set the
# radius env to a huge value (or edit here) to restore map-wide spawning.
DEFAULT_PED_SPAWN_CENTER_X = 18.45
DEFAULT_PED_SPAWN_CENTER_Y = -63.0
DEFAULT_PED_SPAWN_RADIUS_M = 60.0


def keep_pedestrians_running() -> bool:
    """When set by Start.py, spawn once and roam until the process is stopped (no prompt)."""
    return os.environ.get("DATASET_KEEP_PEDESTRIANS_RUNNING", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pedestrian_count_from_env(default=DEFAULT_PED_COUNT) -> int:
    raw = os.environ.get("DATASET_PEDESTRIAN_COUNT", "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def ped_spawn_region_from_env():
    """Return (center_x, center_y, radius_m); defaults to the monitored corridor.

    Both initial navmesh spawn and AI retargeting reject samples outside this 2D
    circle (z ignored). Each of the three values falls back to its corridor default
    independently, so you can override just the radius (etc.) via env.
    """
    def _f(name, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default
    return (
        _f("DATASET_PED_SPAWN_CENTER_X", DEFAULT_PED_SPAWN_CENTER_X),
        _f("DATASET_PED_SPAWN_CENTER_Y", DEFAULT_PED_SPAWN_CENTER_Y),
        _f("DATASET_PED_SPAWN_RADIUS_M", DEFAULT_PED_SPAWN_RADIUS_M),
    )


_PED_REGION_MAX_NAV_ATTEMPTS = 200


def get_nav_location_in_region(world, region):
    """Rejection-sample navmesh points until one lands inside region (cx, cy, r_m)."""
    if region is None:
        return world.get_random_location_from_navigation()
    cx, cy, r = region
    r_sq = r * r
    for _ in range(_PED_REGION_MAX_NAV_ATTEMPTS):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        dx = loc.x - cx
        dy = loc.y - cy
        if dx * dx + dy * dy <= r_sq:
            return loc
    return None


def prompt_pedestrian_count(default=DEFAULT_PED_COUNT):
    while True:
        try:
            raw = input(f"How many pedestrians to spawn? [default {default}]: ").strip()
        except EOFError:
            return default

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer. Try again.")
            continue

        if value < 0:
            print("  Count must be >= 0. Try again.")
            continue

        return value


def spawn_pedestrians_with_ai(client, world, count):
    """
    CARLA official pedestrian flow (generate_traffic.py).
    """
    SpawnActor = carla.command.SpawnActor

    seed = random.randint(0, 0xFFFFFFFF)
    try:
        world.set_pedestrians_seed(seed)
    except RuntimeError:
        pass
    random.seed(seed)

    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        raise RuntimeError("No walker.pedestrian.* blueprints found.")

    controller_bp = bp_lib.find("controller.ai.walker")
    if controller_bp is None:
        raise RuntimeError("controller.ai.walker blueprint not found.")

    # 1) Collect navmesh spawn points (only valid pedestrian locations).
    region = ped_spawn_region_from_env()
    if region is not None:
        print(
            f"[ped] restricting spawn to circle (cx={region[0]:.1f}, cy={region[1]:.1f}, "
            f"r={region[2]:.1f} m)",
            flush=True,
        )
    spawn_points = []
    for _ in range(count):
        loc = get_nav_location_in_region(world, region)
        if loc is not None:
            spawn_points.append(carla.Transform(loc))

    if not spawn_points:
        raise RuntimeError("No navmesh spawn locations found on this map.")

    # 2) Spawn walkers (batch + tick so the server registers them).
    walker_batch = []
    walker_speeds = []
    for transform in spawn_points:
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        if walker_bp.has_attribute("speed"):
            values = walker_bp.get_attribute("speed").recommended_values
            if random.random() < PED_RUNNING_FRACTION and len(values) > 2:
                walker_speeds.append(values[2])
            else:
                walker_speeds.append(values[1])
        else:
            walker_speeds.append("1.4")

        walker_batch.append(SpawnActor(walker_bp, transform))

    walker_results = client.apply_batch_sync(walker_batch, True)

    walkers_meta = []
    speeds_meta = []
    for result, speed in zip(walker_results, walker_speeds):
        if result.error:
            print(f"  walker spawn failed: {result.error}", flush=True)
            continue
        walkers_meta.append({"id": result.actor_id})
        speeds_meta.append(speed)

    if not walkers_meta:
        return [], [], count

    # 3) Spawn AI controllers attached to each walker (batch + tick).
    controller_batch = [
        SpawnActor(controller_bp, carla.Transform(), entry["id"]) for entry in walkers_meta
    ]
    controller_results = client.apply_batch_sync(controller_batch, True)

    all_ids = []
    paired_speeds = []
    for entry, ctrl_result, speed in zip(walkers_meta, controller_results, speeds_meta):
        if ctrl_result.error:
            print(f"  controller spawn failed: {ctrl_result.error}", flush=True)
            client.apply_batch([carla.command.DestroyActor(entry["id"])])
            continue
        entry["con"] = ctrl_result.actor_id
        all_ids.append(entry["con"])
        all_ids.append(entry["id"])
        paired_speeds.append(speed)

    skipped = count - len(paired_speeds)

    # 4) Wait for a tick so transforms are ready (required before start()).
    settings = world.get_settings()
    sync_mode = bool(getattr(settings, "synchronous_mode", False))
    if sync_mode:
        world.tick()
    else:
        world.wait_for_tick()

    # 5) Crossing factor: fraction that will use crosswalks / cross streets legally.
    try:
        world.set_pedestrians_cross_factor(PED_CROSSING_FACTOR)
    except RuntimeError:
        pass

    # 6) Interleaved list from get_actors: [controller, walker, controller, walker, ...]
    all_actors = world.get_actors(all_ids)
    controllers = []
    walkers = []

    for i in range(0, len(all_ids), 2):
        controller = all_actors[i]
        walker = all_actors[i + 1]
        speed = float(paired_speeds[i // 2])
        target = get_nav_location_in_region(world, region)
        if target is None:
            continue

        controller.start()
        controller.go_to_location(target)
        controller.set_max_speed(speed)

        controllers.append(controller)
        walkers.append(walker)
        loc = walker.get_location()
        print(
            f"  walker_id={walker.id} controller_id={controller.id} "
            f"at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) speed={speed:.2f}",
            flush=True,
        )

    return walkers, controllers, skipped, sync_mode


def retarget_controllers(world, controllers):
    """Send walkers to a new random navmesh point (sidewalk / crosswalk)."""
    region = ped_spawn_region_from_env()
    for controller in controllers:
        try:
            if not controller.is_alive:
                continue
            target = get_nav_location_in_region(world, region)
            if target is not None:
                controller.go_to_location(target)
        except RuntimeError:
            continue


def cleanup(walkers, controllers):
    for controller in controllers:
        try:
            if controller.is_alive:
                controller.stop()
        except RuntimeError:
            pass

    destroyed = 0
    for actor in controllers + walkers:
        try:
            if actor.is_alive and actor.destroy():
                destroyed += 1
        except RuntimeError:
            continue
    return destroyed


def main():
    if keep_pedestrians_running():
        count = pedestrian_count_from_env()
    else:
        count = prompt_pedestrian_count()
    if count == 0:
        print("Nothing to spawn. Exiting.")
        return 0

    client, world = get_world()

    print(
        "Using CARLA built-in pedestrians (controller.ai.walker + navmesh).",
        flush=True,
    )
    print(f"Spawning {count} pedestrians on navmesh locations...", flush=True)

    walkers, controllers, skipped, sync_mode = spawn_pedestrians_with_ai(
        client, world, count
    )

    print(
        f"Spawned {len(walkers)}/{count} with AI controllers "
        f"(skipped {skipped}, crossing_factor={PED_CROSSING_FACTOR:.2f}).",
        flush=True,
    )
    if not walkers:
        return 0

    if keep_pedestrians_running():
        print(
            "Pedestrians roaming (DATASET_KEEP_PEDESTRIANS_RUNNING). "
            "Stop via Start.py Ctrl+C or kill this process.",
            flush=True,
        )
    else:
        print(
            "Pedestrians follow the navmesh (sidewalks / crosswalks). "
            "Press Ctrl+C to stop.",
            flush=True,
        )

    next_retarget = time.time() + RETARGET_INTERVAL_S
    try:
        while True:
            if sync_mode:
                world.tick()
            else:
                world.wait_for_tick()

            now = time.time()
            if now >= next_retarget:
                retarget_controllers(world, controllers)
                next_retarget = now + RETARGET_INTERVAL_S
    except KeyboardInterrupt:
        if not keep_pedestrians_running():
            print("\nCtrl+C received. Cleaning up...", flush=True)
    finally:
        if not keep_pedestrians_running():
            destroyed = cleanup(walkers, controllers)
            print(f"Destroyed {destroyed} actors.", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"CARLA error: {exc}", file=sys.stderr)
        sys.exit(1)
