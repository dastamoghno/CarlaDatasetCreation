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
import threading
import time

import carla

from carla_connect import get_world


# Primary spawn anchor: map spawn-point index 14 is the "Position 14" dataset zone.
SPAWN_CENTER_INDEX = 14
# Only vehicles within this radius of spawn point 14 are used.
# Keeps all traffic local to the capture area and avoids random far-flung spawns.
SPAWN_RADIUS_M = 200.0
TARGET_CAR_COUNT = 35
# Poisson process rate (lambda): expected spawns per second.
SPAWN_RATE_PER_SECOND = 0.5
TRAFFIC_MANAGER_PORT = 8000
MOVE_AWAY_DISTANCE_M = 8.0
MOVE_AWAY_TIMEOUT_S = 30.0
MOVE_AWAY_POLL_S = 0.25
# Waypoints for traffic_manager.set_path (lane follow; avoids set_route "RoadOption" errors).
LANE_PATH_POINTS = 120
# Free-driving mode also uses set_path to avoid NavMesh routing failures (NAV warnings).
# Longer path gives vehicles enough road ahead with lane changes still active.
FREE_DRIVING_PATH_POINTS = 300
LANE_PATH_STEP_M = 5.0
# ── Crash-prevention settings ───────────────────────────────────────────────
# Minimum gap (metres) the TM keeps behind the vehicle ahead.
SAFE_FOLLOWING_DISTANCE_M = 8.0
# Positive value → vehicles drive this % slower than the posted speed limit.
VEHICLE_SPEED_REDUCTION_PCT = 25.0
# ────────────────────────────────────────────────────────────────────────────
LABEL_REFRESH_S = 0.25
LABEL_DURATION_S = 120.0
AUTOPILOT_MONITOR_INTERVAL_S = 2.0
# Drive-in mode: role prefix to find dataset radars + clearance so cars never
# spawn inside a radar's range (avoids "pop-in"). 35 m matches RADAR_MAX_RANGE_M
# in CaptureRadarCameraData.py; +5 m safety margin.
DATASET_RADAR_ROLE_PREFIX = "dataset_radar_"
RADAR_SPAWN_CLEARANCE_M = 40.0


def distance_sq(loc_a, loc_b):
    dx = loc_a.x - loc_b.x
    dy = loc_a.y - loc_b.y
    dz = loc_a.z - loc_b.z
    return dx * dx + dy * dy + dz * dz


def wait_until_vehicle_moves_away(
    actor,
    spawn_location,
    min_distance_m=MOVE_AWAY_DISTANCE_M,
    timeout_s=MOVE_AWAY_TIMEOUT_S,
    poll_s=MOVE_AWAY_POLL_S,
):
    min_distance_sq = min_distance_m * min_distance_m
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            if not actor.is_alive:
                return False, "actor_not_alive"

            loc = actor.get_location()
        except RuntimeError:
            # Actor can disappear after crashes/cleanup; treat as failed wait, not fatal.
            return False, "actor_unavailable"

        if distance_sq(loc, spawn_location) >= min_distance_sq:
            return True, "moved_away"

        time.sleep(poll_s)

    return False, "timeout"


def get_nearby_spawn_points(spawn_points, center_transform, radius_m):
    """Return spawn points whose location is within radius_m of center_transform.

    Points are returned sorted nearest-first so vehicles fill the capture zone
    before reaching out to the edge of the radius.
    """
    radius_sq = radius_m * radius_m
    center = center_transform.location
    nearby = [
        tr for tr in spawn_points
        if distance_sq(tr.location, center) <= radius_sq
    ]
    nearby.sort(key=lambda tr: distance_sq(tr.location, center))
    return nearby


def get_radar_positions(world):
    """World-XY locations of every dataset radar (empty if none spawned yet)."""
    out = []
    for actor in world.get_actors():
        if actor.type_id != "sensor.other.radar":
            continue
        if not actor.attributes.get("role_name", "").startswith(DATASET_RADAR_ROLE_PREFIX):
            continue
        loc = actor.get_transform().location
        out.append((loc.x, loc.y))
    return out


def get_drive_in_spawn_points(
    world_map,
    spawn_points,
    center_transform,
    *,
    exclusion_radius_m,
    outer_radius_m,
    monitored_center=None,
    radar_positions=None,
    radar_clearance_m=0.0,
):
    """Spawn-outside, drive-in: pick spawn points OUTSIDE the monitored zone whose
    forward lane path actually drives INTO it.

    A candidate qualifies iff:
      1. It sits in the annulus [exclusion_radius_m, outer_radius_m] from the
         monitored centre — i.e. outside the area we observe but close enough to
         reach it.
      2. It is at least ``radar_clearance_m`` from EVERY radar — the exclusion
         radius is measured from the monitored centre, but radars are spread along
         the corridor, so a point 40 m from centre can still sit inside an end
         radar's range and cause a car to "pop in" instead of driving in. This
         gate removes that.
      3. Its forward lane-follow path enters within exclusion_radius_m of the
         monitored centre — i.e. the car will actually appear in the scene we care
         about rather than driving away from it.

    Returned farthest-first so cars enter from the approaches and stream through,
    rather than popping up at the boundary.
    """
    center = monitored_center if monitored_center is not None else center_transform.location
    excl_sq = exclusion_radius_m * exclusion_radius_m
    outer_sq = outer_radius_m * outer_radius_m
    radars = radar_positions or []
    clear_sq = radar_clearance_m * radar_clearance_m

    def clear_of_radars(loc):
        if not radars or radar_clearance_m <= 0.0:
            return True
        return all((loc.x - rx) ** 2 + (loc.y - ry) ** 2 > clear_sq for rx, ry in radars)

    annulus = [
        tr for tr in spawn_points
        if excl_sq <= distance_sq(tr.location, center) <= outer_sq
        and clear_of_radars(tr.location)
    ]
    # Validate drive-in: forward lane path must pass through the monitored zone.
    drive_in = []
    for tr in annulus:
        path = build_lane_follow_path(
            world_map, tr, num_points=FREE_DRIVING_PATH_POINTS, step_m=LANE_PATH_STEP_M
        )
        enters = any(distance_sq(p, center) <= excl_sq for p in path)
        if enters:
            drive_in.append(tr)
    # Farthest-first: long approaches stream in first.
    drive_in.sort(key=lambda tr: -distance_sq(tr.location, center))
    return drive_in


def build_lane_follow_path(world_map, spawn_transform, num_points=LANE_PATH_POINTS, step_m=LANE_PATH_STEP_M):
    """Forward path along the driving lane (no TM set_route / RoadOption junction list)."""
    wp = world_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return []

    path = []
    current = wp
    for _ in range(num_points):
        path.append(current.transform.location)
        nxt = current.next(step_m)
        if not nxt:
            break
        current = nxt[0]
    return path


def spawn_center_index_from_env(default=SPAWN_CENTER_INDEX) -> int:
    raw = os.environ.get("DATASET_SPAWN_CENTER_INDEX", "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def spawn_radius_m_from_env(default=SPAWN_RADIUS_M) -> float:
    raw = os.environ.get("DATASET_SPAWN_RADIUS_M", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def target_car_count_from_env(default=TARGET_CAR_COUNT) -> int:
    raw = os.environ.get("DATASET_TARGET_CAR_COUNT", "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def spawn_exclusion_radius_m_from_env(default=0.0) -> float:
    """Inner 'monitored zone' radius (m). When > 0, NO cars spawn inside it; they
    spawn in the annulus [exclusion, spawn_radius] and must drive in. 0 disables
    (legacy fill-the-zone behaviour)."""
    raw = os.environ.get("DATASET_SPAWN_EXCLUSION_RADIUS_M", "").strip()
    try:
        return max(0.0, float(raw)) if raw else default
    except ValueError:
        return default


def vehicle_speed_reduction_pct_from_env(default=VEHICLE_SPEED_REDUCTION_PCT) -> float:
    """Override TM speed_difference (% slower than posted limit).

    Negative values drive ABOVE the speed limit, e.g. -50 → 1.5× limit.
    Default 25.0 = 25% slower; Town10HD default-limit 30 km/h → effective 22 km/h.
    """
    raw = os.environ.get("DATASET_VEHICLE_SPEED_REDUCTION_PCT", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def safe_following_distance_m_from_env(default=SAFE_FOLLOWING_DISTANCE_M) -> float:
    raw = os.environ.get("DATASET_SAFE_FOLLOWING_DISTANCE_M", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def free_vehicle_driving_from_env() -> bool:
    raw = os.environ.get("DATASET_FREE_VEHICLE_DRIVING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def keep_traffic_running_from_env() -> bool:
    return os.environ.get("DATASET_KEEP_TRAFFIC_RUNNING", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def apply_free_driving_policy(traffic_manager, actor, world_map=None, spawn_transform=None):
    """Autopilot with lane changes and normal junction behavior.

    Providing world_map + spawn_transform causes an explicit set_path call so the
    Traffic Manager follows valid waypoints instead of self-routing via NavMesh queries
    (which produce 'WARNING: NAV: Failed to set request to go to ...' in the CARLA logs
    when the chosen destination lands off the driveable surface).
    """
    if hasattr(traffic_manager, "auto_lane_change"):
        traffic_manager.auto_lane_change(actor, True)
    if hasattr(traffic_manager, "random_left_lanechange_percentage"):
        traffic_manager.random_left_lanechange_percentage(actor, 20.0)
    if hasattr(traffic_manager, "random_right_lanechange_percentage"):
        traffic_manager.random_right_lanechange_percentage(actor, 20.0)
    if hasattr(traffic_manager, "ignore_lights_percentage"):
        traffic_manager.ignore_lights_percentage(actor, 0.0)
    if hasattr(traffic_manager, "ignore_signs_percentage"):
        traffic_manager.ignore_signs_percentage(actor, 0.0)
    if hasattr(traffic_manager, "ignore_vehicles_percentage"):
        traffic_manager.ignore_vehicles_percentage(actor, 0.0)

    # Crash prevention: enforce a safe gap and cap speed.
    if hasattr(traffic_manager, "distance_to_leading_vehicle"):
        traffic_manager.distance_to_leading_vehicle(actor, safe_following_distance_m_from_env())
    if hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
        traffic_manager.vehicle_percentage_speed_difference(actor, vehicle_speed_reduction_pct_from_env())

    if world_map is not None and spawn_transform is not None and hasattr(traffic_manager, "set_path"):
        path = build_lane_follow_path(
            world_map, spawn_transform, num_points=FREE_DRIVING_PATH_POINTS
        )
        if len(path) >= 2:
            try:
                traffic_manager.set_path(actor, path)
            except RuntimeError:
                pass  # Non-fatal: TM will fall back to its own routing.


def apply_straight_driving_policy(traffic_manager, actor, world_map, spawn_transform):
    """
    Keep vehicles in-lane without traffic_manager.set_route(['Straight', ...]).
    That API logs 'We couldn't find the RoadOption...' from CARLA when a junction
    has no straight topology (stderr, not a Python exception).
    """
    if hasattr(traffic_manager, "auto_lane_change"):
        traffic_manager.auto_lane_change(actor, False)
    if hasattr(traffic_manager, "random_left_lanechange_percentage"):
        traffic_manager.random_left_lanechange_percentage(actor, 0.0)
    if hasattr(traffic_manager, "random_right_lanechange_percentage"):
        traffic_manager.random_right_lanechange_percentage(actor, 0.0)

    if hasattr(traffic_manager, "keep_right_rule_percentage"):
        traffic_manager.keep_right_rule_percentage(actor, 100.0)
    elif hasattr(traffic_manager, "keep_slow_lane_rule_percentage"):
        traffic_manager.keep_slow_lane_rule_percentage(actor, 100.0)

    # Crash prevention: enforce a safe gap and cap speed.
    if hasattr(traffic_manager, "distance_to_leading_vehicle"):
        traffic_manager.distance_to_leading_vehicle(actor, safe_following_distance_m_from_env())
    if hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
        traffic_manager.vehicle_percentage_speed_difference(actor, vehicle_speed_reduction_pct_from_env())

    if hasattr(traffic_manager, "set_path"):
        path = build_lane_follow_path(world_map, spawn_transform)
        if len(path) >= 2:
            try:
                traffic_manager.set_path(actor, path)
            except RuntimeError as exc:
                print(
                    f"Note: lane path not set for vehicle {actor.id}: {exc}",
                    flush=True,
                )


def draw_vehicle_labels(
    world,
    labeled_actors,
    duration_s=LABEL_DURATION_S,
    refresh_s=LABEL_REFRESH_S,
):
    if duration_s <= 0.0 or not labeled_actors:
        return

    end_time = time.time() + duration_s
    while time.time() < end_time:
        any_alive = False
        for actor, label in labeled_actors:
            try:
                if not actor.is_alive:
                    continue
                any_alive = True
                world.debug.draw_string(
                    actor.get_location() + carla.Location(z=1.8),
                    label,
                    draw_shadow=False,
                    color=carla.Color(0, 200, 255),
                    life_time=refresh_s + 0.05,
                    persistent_lines=False,
                )
            except RuntimeError:
                continue

        if not any_alive:
            break
        time.sleep(refresh_s)


def monitor_autopilot_until_interrupted(
    traffic_manager_port, spawned_ids, poll_s=AUTOPILOT_MONITOR_INTERVAL_S
):
    """Keep this process alive and re-enable autopilot if a vehicle loses it."""
    client, world = get_world()

    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    print("Autopilot monitor active. Press Ctrl+C to stop this script.")
    while True:
        actors = world.get_actors(spawned_ids)
        for actor in actors:
            try:
                if not actor.is_alive:
                    continue
                # Some CARLA builds do not expose an autopilot state getter.
                # Re-applying autopilot keeps behavior consistent and is safe.
                actor.set_autopilot(True, traffic_manager_port)
                if free_vehicle_driving_from_env():
                    apply_free_driving_policy(traffic_manager, actor)
            except (RuntimeError, AttributeError):
                # Vehicle may have been removed asynchronously; ignore and continue.
                continue
        time.sleep(poll_s)


def recirculate_traffic_until_interrupted(
    traffic_manager_port,
    spawned_ids,
    center_index,
    *,
    poll_s=AUTOPILOT_MONITOR_INTERVAL_S,
):
    """Sustained flow: keep a fixed fleet cycling THROUGH the monitored zone.

    A car that drifts past ``recycle_radius_m`` from the monitored centre (i.e. it
    has driven out the far side and is wandering off) is destroyed and a fresh car
    is spawned at a random drive-in point in the annulus. Because both the
    destruction (beyond recycle radius) and the respawn (in the annulus, outside
    the exclusion radius) happen OUTSIDE the monitored zone, the observed scene
    only ever sees cars driving through — never popping in/out.

    Net effect: a bounded fleet of N cars produces an unbounded stream of corridor
    traversals over the whole capture.
    """
    client, world = get_world()
    world_map = world.get_map()
    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    vehicle_bps = get_vehicle_blueprints(world)

    spawn_points = world_map.get_spawn_points()
    center_location = spawn_points[center_index].location
    exclusion_radius_m = spawn_exclusion_radius_m_from_env()
    outer_radius_m = spawn_radius_m_from_env()
    # Recycle a car once it is well past the spawn annulus (driven out the far side).
    recycle_radius_m = outer_radius_m + 20.0
    recycle_sq = recycle_radius_m * recycle_radius_m

    drive_in_points = get_drive_in_spawn_points(
        world_map,
        spawn_points,
        spawn_points[center_index],
        exclusion_radius_m=exclusion_radius_m,
        outer_radius_m=outer_radius_m,
        radar_positions=get_radar_positions(world),
        radar_clearance_m=RADAR_SPAWN_CLEARANCE_M,
    )
    if not drive_in_points:
        print("Recirculation: no drive-in points; falling back to plain autopilot monitor.",
              flush=True)
        monitor_autopilot_until_interrupted(traffic_manager_port, spawned_ids, poll_s)
        return

    print(
        f"Recirculation manager active: fleet={len(spawned_ids)} cars, "
        f"recycle beyond {recycle_radius_m:.0f} m, respawn in annulus "
        f"[{exclusion_radius_m:.0f}, {outer_radius_m:.0f}] m. Ctrl+C to stop.",
        flush=True,
    )

    recycle_count = 0
    while True:
        for idx, actor_id in enumerate(list(spawned_ids)):
            try:
                actor = world.get_actor(actor_id)
                needs_recycle = actor is None or not actor.is_alive
                if not needs_recycle:
                    loc = actor.get_location()
                    if distance_sq(loc, center_location) > recycle_sq:
                        needs_recycle = True
                if not needs_recycle:
                    # Keep autopilot + policy fresh.
                    actor.set_autopilot(True, traffic_manager_port)
                    if free_vehicle_driving_from_env():
                        apply_free_driving_policy(traffic_manager, actor)
                    continue

                # Destroy the drifted/dead car.
                if actor is not None and actor.is_alive:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass

                # Respawn at a random drive-in point (retry a few collisions).
                new_actor = None
                for transform in random.sample(drive_in_points, min(len(drive_in_points), 8)):
                    bp = random.choice(vehicle_bps)
                    if bp.has_attribute("color"):
                        bp.set_attribute(
                            "color",
                            random.choice(bp.get_attribute("color").recommended_values),
                        )
                    new_actor = world.try_spawn_actor(bp, transform)
                    if new_actor is not None:
                        try:
                            new_actor.set_autopilot(True, traffic_manager_port)
                            if free_vehicle_driving_from_env():
                                apply_free_driving_policy(
                                    traffic_manager, new_actor, world_map, transform
                                )
                            else:
                                apply_straight_driving_policy(
                                    traffic_manager, new_actor, world_map, transform
                                )
                        except (RuntimeError, AttributeError):
                            try:
                                new_actor.destroy()
                            except RuntimeError:
                                pass
                            new_actor = None
                            continue
                        break
                if new_actor is not None:
                    spawned_ids[idx] = new_actor.id
                    recycle_count += 1
                    if recycle_count % 10 == 0:
                        print(f"  [recirculation] {recycle_count} cars recycled", flush=True)
            except (RuntimeError, AttributeError):
                continue
        time.sleep(poll_s)


def get_vehicle_blueprints(world):
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    usable = []

    for bp in blueprints:
        if bp.id.endswith("isetta"):
            continue
        if bp.has_attribute("number_of_wheels"):
            if int(bp.get_attribute("number_of_wheels").as_int()) != 4:
                continue
        usable.append(bp)

    if not usable:
        raise RuntimeError("No usable 4-wheel vehicle blueprints found.")
    return usable


def try_spawn_cars(
    world,
    traffic_manager,
    center_index=SPAWN_CENTER_INDEX,
    spawn_radius_m=SPAWN_RADIUS_M,
    target_count=TARGET_CAR_COUNT,
    spawn_rate_per_second=SPAWN_RATE_PER_SECOND,
    traffic_manager_port=TRAFFIC_MANAGER_PORT,
    move_away_distance_m=MOVE_AWAY_DISTANCE_M,
    move_away_timeout_s=MOVE_AWAY_TIMEOUT_S,
):
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in current map.")
    if spawn_rate_per_second <= 0.0:
        raise RuntimeError("spawn_rate_per_second must be > 0.")
    if center_index < 0 or center_index >= len(spawn_points):
        raise RuntimeError(
            f"SPAWN_CENTER_INDEX {center_index} is out of range "
            f"(map has {len(spawn_points)} spawn points, indices 0–{len(spawn_points)-1})."
        )

    center_transform = spawn_points[center_index]

    exclusion_radius_m = spawn_exclusion_radius_m_from_env()
    if exclusion_radius_m > 0.0:
        # Spawn-outside, drive-in: no cars inside the monitored zone; they enter it.
        radar_positions = get_radar_positions(world)
        candidate_points = get_drive_in_spawn_points(
            world_map,
            spawn_points,
            center_transform,
            exclusion_radius_m=exclusion_radius_m,
            outer_radius_m=spawn_radius_m,
            radar_positions=radar_positions,
            radar_clearance_m=RADAR_SPAWN_CLEARANCE_M,
        )
        if not candidate_points:
            raise RuntimeError(
                f"No drive-in spawn points found in annulus "
                f"[{exclusion_radius_m:.0f}, {spawn_radius_m:.0f}] m of spawn index "
                f"{center_index} whose lane path enters the monitored zone. "
                "Widen SPAWN_RADIUS_M or lower SPAWN_EXCLUSION_RADIUS_M."
            )
        print(
            f"Spawn mode: DRIVE-IN | monitored centre index={center_index} "
            f"({center_transform.location.x:.1f}, {center_transform.location.y:.1f}) | "
            f"spawn annulus [{exclusion_radius_m:.0f}, {spawn_radius_m:.0f}] m | "
            f"drive-in candidates={len(candidate_points)}",
            flush=True,
        )
    else:
        # Legacy: all spawn points within the capture radius, nearest-first.
        candidate_points = get_nearby_spawn_points(spawn_points, center_transform, spawn_radius_m)
        if not candidate_points:
            raise RuntimeError(
                f"No spawn points found within {spawn_radius_m:.0f} m of spawn index {center_index}. "
                "Try increasing SPAWN_RADIUS_M."
            )
        print(
            f"Spawn zone: index={center_index} "
            f"({center_transform.location.x:.1f}, {center_transform.location.y:.1f}), "
            f"radius={spawn_radius_m:.0f} m, candidates={len(candidate_points)}",
            flush=True,
        )

    vehicle_bps = get_vehicle_blueprints(world)
    spawned_ids = []
    labeled_actors = []

    for transform in candidate_points:
        if len(spawned_ids) >= target_count:
            break

        # Exponential inter-arrival delay => Poisson spawn process.
        delay_s = random.expovariate(spawn_rate_per_second)
        time.sleep(delay_s)

        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        if bp.has_attribute("driver_id"):
            driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
            bp.set_attribute("driver_id", driver_id)

        actor = world.try_spawn_actor(bp, transform)
        if actor is None:
            print(f"Skipped blocked spawn point after {delay_s:.2f}s delay.")
            continue

        try:
            actor.set_autopilot(True, traffic_manager_port)
            if free_vehicle_driving_from_env():
                apply_free_driving_policy(traffic_manager, actor, world_map, transform)
            else:
                apply_straight_driving_policy(traffic_manager, actor, world_map, transform)
        except RuntimeError:
            print("Spawned actor could not enable autopilot; destroying and skipping.")
            if actor.is_alive:
                actor.destroy()
            continue

        moved_away, reason = wait_until_vehicle_moves_away(
            actor,
            transform.location,
            min_distance_m=move_away_distance_m,
            timeout_s=move_away_timeout_s,
        )
        if not moved_away:
            print(
                "Spawned actor did not clear spawn zone "
                f"(reason={reason}); destroying and skipping."
            )
            if actor.is_alive:
                actor.destroy()
            continue

        spawned_ids.append(actor.id)
        label = f"CAR {len(spawned_ids):02d}"
        labeled_actors.append((actor, label))
        print(
            f"Spawned {len(spawned_ids)}/{target_count} "
            f"(actor_id={actor.id}) after {delay_s:.2f}s delay. "
            f"autopilot=on, driving={'free' if free_vehicle_driving_from_env() else 'lane_keep'}, "
            f"label={label}, "
            f"cleared_spawn>={move_away_distance_m:.1f}m"
        )

    return (
        len(spawned_ids),
        spawned_ids,
        center_transform,
        len(spawn_points),
        len(candidate_points),
        spawn_rate_per_second,
        traffic_manager_port,
        move_away_distance_m,
        move_away_timeout_s,
        labeled_actors,
    )


def main():
    client, world = get_world()

    (
        count,
        ids,
        center_transform,
        spawn_point_total,
        candidate_count,
        spawn_rate,
        traffic_manager_port,
        move_away_distance_m,
        move_away_timeout_s,
        labeled_actors,
    ) = try_spawn_cars(
        world,
        traffic_manager=client.get_trafficmanager(TRAFFIC_MANAGER_PORT),
        center_index=spawn_center_index_from_env(),
        spawn_radius_m=spawn_radius_m_from_env(),
        target_count=target_car_count_from_env(),
    )

    print(f"Spawn point total on map: {spawn_point_total}")
    print(
        f"Spawn center: index={spawn_center_index_from_env()} "
        f"({center_transform.location.x:.2f}, {center_transform.location.y:.2f}, "
        f"{center_transform.location.z:.2f})"
    )
    print(f"Spawn radius: {spawn_radius_m_from_env():.0f} m  |  candidates in radius: {candidate_count}")
    print(
        f"Poisson spawn rate: {spawn_rate:.2f}/s "
        f"(mean interval {1.0 / spawn_rate:.2f}s)"
    )
    print(f"Traffic Manager port: {traffic_manager_port}")
    if free_vehicle_driving_from_env():
        print(
            "Driving policy: free autopilot "
            f"(lane changes on, speed -{VEHICLE_SPEED_REDUCTION_PCT:.0f}%, "
            f"following gap {SAFE_FOLLOWING_DISTANCE_M:.0f} m)."
        )
    else:
        print(
            "Lane-keep policy: lane changes disabled; TM path follows spawn lane "
            f"({LANE_PATH_POINTS} x {LANE_PATH_STEP_M:.0f} m, "
            f"speed -{VEHICLE_SPEED_REDUCTION_PCT:.0f}%, "
            f"following gap {SAFE_FOLLOWING_DISTANCE_M:.0f} m)."
        )
    print(
        f"Spawn gating: next spawn waits until previous moved "
        f">={move_away_distance_m:.1f} m (timeout {move_away_timeout_s:.1f}s)"
    )
    print(f"Requested cars: {target_car_count_from_env()}")
    print(f"Successfully spawned: {count}")
    if ids:
        print("Spawned vehicle actor IDs:", ", ".join(str(actor_id) for actor_id in ids))
    if ids:
        if keep_traffic_running_from_env():
            if labeled_actors and LABEL_DURATION_S > 0:
                threading.Thread(
                    target=draw_vehicle_labels,
                    args=(world, labeled_actors),
                    kwargs={"duration_s": LABEL_DURATION_S},
                    daemon=True,
                ).start()
            print(
                "Vehicles roaming with autopilot (DATASET_KEEP_TRAFFIC_RUNNING). "
                "Stop via Start.py Ctrl+C.",
                flush=True,
            )
            try:
                if spawn_exclusion_radius_m_from_env() > 0.0:
                    recirculate_traffic_until_interrupted(
                        traffic_manager_port, ids, spawn_center_index_from_env()
                    )
                else:
                    monitor_autopilot_until_interrupted(traffic_manager_port, ids)
            except KeyboardInterrupt:
                print("Autopilot monitor stopped.")
        else:
            print(f"Drawing labels for {LABEL_DURATION_S:.0f}s")
            draw_vehicle_labels(world, labeled_actors)
            try:
                monitor_autopilot_until_interrupted(traffic_manager_port, ids)
            except KeyboardInterrupt:
                print("Autopilot monitor stopped by user.")


if __name__ == "__main__":
    main()
