import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import math
import os
import random
import threading
import time

import carla

from carla_connect import get_world
from world.bicycle_sidewalk import (
    BicycleSidewalkManager,
    bicycle_sidewalk_share_from_env,
)
from world.vehicle_classes import (
    bp_is_bicycle,
    bp_is_motorcycle,
    bp_is_truck,
    vehicle_class_from_type_id,
)


# Primary spawn anchor: map spawn-point index 144 is the monitored corridor
# (road_id 10, ~(22.9, -60.9)) covered by the 8-radar layout. Index 14 was the
# legacy "Position 14" zone (~(-113, -25)) — far from the radars; do not use it.
SPAWN_CENTER_INDEX = 144
# Outer spawn radius (drive-in annulus upper bound). The radar corridor is ~95 m
# long and cars must spawn >= RADAR_SPAWN_CLEARANCE_M from every radar, so the
# nearest valid drive-in point is ~74 m out; 100 m yields ~28 approach points that
# reach the radar FOV while staying close enough to arrive quickly. 80 m gives only
# ~3; 150 m pushes spawns 120 m+ out (too far to arrive in a short capture).
SPAWN_RADIUS_M = 100.0
# Fleet size cycled through the corridor. Higher = more cars simultaneously in view;
# the short corridor + 30 km/h can stop-and-go if pushed too far, but with the drive-in
# spread ~30 keeps a denser stream flowing. Override via DATASET_TARGET_CAR_COUNT.
TARGET_CAR_COUNT = 30
# Fleet class fractions (car = remainder). DATASET_TARGET_CAR_COUNT is total fleet
# size across all types (legacy name). Override via DATASET_VEHICLE_*_FRACTION or
# derive truck share from DATASET_TARGET_TRUCK_COUNT.
DEFAULT_TRUCK_FRACTION = 0.10
DEFAULT_MOTORCYCLE_FRACTION = 0.0
DEFAULT_BICYCLE_FRACTION = 0.0
DEFAULT_BICYCLE_ROAD_SPEED_REDUCTION_PCT = 40.0
DEFAULT_MOTORCYCLE_ROAD_SPEED_REDUCTION_PCT = 0.0
TWO_WHEELER_SPAWN_ATTEMPTS = 3
# Poisson process rate (lambda): expected spawns per second. Higher = fleet fills
# faster (more cars on the road sooner). Paired with the smaller MOVE_AWAY_DISTANCE_M.
SPAWN_RATE_PER_SECOND = 1.0
TRAFFIC_MANAGER_PORT = 8000
MOVE_AWAY_DISTANCE_M = 4.0   # next spawn once previous clears this far; lower = faster fill
MOVE_AWAY_TIMEOUT_S = 30.0
MOVE_AWAY_POLL_S = 0.25
# Waypoints for traffic_manager.set_path (lane follow; avoids set_route "RoadOption" errors).
LANE_PATH_POINTS = 120
# Free-driving mode also uses set_path to avoid NavMesh routing failures (NAV warnings).
# Longer path gives vehicles enough road ahead with lane changes still active.
FREE_DRIVING_PATH_POINTS = 300
LANE_PATH_STEP_M = 5.0
# ── Crash-prevention settings ───────────────────────────────────────────────
# Minimum gap (metres) the TM keeps behind the vehicle ahead. 5 m (~0.6 s headway at
# 30 km/h) keeps the corridor flowing; tighter packing (e.g. 3.5) just gridlocks the
# single 30 km/h corridor instead of adding moving cars in view.
SAFE_FOLLOWING_DISTANCE_M = 5.0
# Positive → drive this % slower than the posted limit; NEGATIVE → faster.
# 0 = follow the posted limit (~30 km/h on Town10HD's signless corridor). Use a
# negative value only if you want cars to exceed the limit for stronger Doppler.
VEHICLE_SPEED_REDUCTION_PCT = 0.0
# ────────────────────────────────────────────────────────────────────────────
LABEL_REFRESH_S = 0.25
LABEL_DURATION_S = 120.0
AUTOPILOT_MONITOR_INTERVAL_S = 2.0
# ── Stuck recycling (recirculation) ─────────────────────────────────────────
# A car is recycled as "stuck" only if it has not progressed for STUCK_TIMEOUT_S
# AND is neither stopped at a red light nor queued behind another vehicle — i.e. it
# is genuinely wedged, never merely waiting/parked. Generous timeout so normal
# light cycles and brief jams clear on their own first.
STUCK_TIMEOUT_S = 75.0
STUCK_MOVE_EPS_M = 1.5          # progress < this since last checkpoint = "not moving"
STUCK_QUEUE_AHEAD_M = 7.0       # a fleet car this close ahead = queued, not wedged
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

    Returned nearest-first so the closest valid approaches are used first and cars
    reach the monitored zone quickly (important for short captures); farther
    approaches fill in behind them.
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
    # Nearest-first: closest approaches are used first so cars reach the corridor
    # quickly (critical for short captures); farther approaches fill in behind them.
    drive_in.sort(key=lambda tr: distance_sq(tr.location, center))
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


def spawn_exclusion_radius_m_from_env(default=30.0) -> float:
    """Inner 'monitored zone' radius (m). When > 0, NO cars spawn inside it; they
    spawn in the annulus [exclusion, spawn_radius] and must drive in. 0 disables
    (legacy fill-the-zone behaviour). Default 30 m → drive-in on; note the real
    out-of-view guarantee is RADAR_SPAWN_CLEARANCE_M (>= radar range), so this value
    mostly controls how deep the drive-in path must reach, not where cars spawn."""
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
        traffic_manager.random_left_lanechange_percentage(actor, 5.0)
    if hasattr(traffic_manager, "random_right_lanechange_percentage"):
        traffic_manager.random_right_lanechange_percentage(actor, 5.0)
    if hasattr(traffic_manager, "ignore_lights_percentage"):
        traffic_manager.ignore_lights_percentage(actor, 0.0)
    if hasattr(traffic_manager, "ignore_signs_percentage"):
        traffic_manager.ignore_signs_percentage(actor, 0.0)
    if hasattr(traffic_manager, "ignore_vehicles_percentage"):
        traffic_manager.ignore_vehicles_percentage(actor, 0.0)
    # Always yield to pedestrians: 0% chance of ignoring a walker, so the TM brakes
    # for crossers/jaywalkers instead of running them over.
    if hasattr(traffic_manager, "ignore_walkers_percentage"):
        traffic_manager.ignore_walkers_percentage(actor, 0.0)

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

    # Yield to other vehicles and pedestrians (0% ignore) so cars brake for
    # crossers instead of running them over.
    if hasattr(traffic_manager, "ignore_vehicles_percentage"):
        traffic_manager.ignore_vehicles_percentage(actor, 0.0)
    if hasattr(traffic_manager, "ignore_walkers_percentage"):
        traffic_manager.ignore_walkers_percentage(actor, 0.0)

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


def _vehicle_queued_ahead(actor, loc, fleet_ids, world, ahead_m=None):
    """True if another fleet vehicle sits close ahead of ``actor`` — i.e. it is queued
    in traffic, not wedged on geometry. Used to NOT recycle cars waiting behind others."""
    ahead_m = STUCK_QUEUE_AHEAD_M if ahead_m is None else ahead_m
    try:
        yaw = math.radians(actor.get_transform().rotation.yaw)
    except RuntimeError:
        return False
    fx, fy = math.cos(yaw), math.sin(yaw)
    ahead_sq = ahead_m * ahead_m
    for other in world.get_actors(fleet_ids):
        if other.id == actor.id:
            continue
        try:
            if not other.is_alive:
                continue
            ol = other.get_location()
        except RuntimeError:
            continue
        dx, dy = ol.x - loc.x, ol.y - loc.y
        if (dx * dx + dy * dy) <= ahead_sq and (dx * fx + dy * fy) > 0.0:
            return True
    return False


def _car_is_wedged(actor, loc, now, last_progress, fleet_ids, world):
    """True only if the car is genuinely stuck (wedged) — NOT if it is legitimately
    waiting: stopped at a red light, parked at one, or queued behind another vehicle.
    In those cases the progress timer is refreshed so the car is never recycled.

    ``last_progress`` maps actor_id -> (x, y, t_last_progress) and is mutated here.
    """
    aid = actor.id
    prev = last_progress.get(aid)
    moved_sq = 0.0 if prev is None else (loc.x - prev[0]) ** 2 + (loc.y - prev[1]) ** 2
    if prev is None or moved_sq >= STUCK_MOVE_EPS_M ** 2:
        last_progress[aid] = (loc.x, loc.y, now)   # made progress → reset timer
        return False
    # Stationary. Refresh the timer for legitimate waits so they never count as stuck.
    try:
        if actor.is_at_traffic_light() and (
            actor.get_traffic_light_state() == carla.TrafficLightState.Red
        ):
            last_progress[aid] = (loc.x, loc.y, now)   # waiting at red light
            return False
    except (RuntimeError, AttributeError):
        pass
    if _vehicle_queued_ahead(actor, loc, fleet_ids, world):
        last_progress[aid] = (loc.x, loc.y, now)       # queued behind traffic
        return False
    # Genuinely stationary, alone, not at a red light → stuck iff it has lasted long.
    return (now - prev[2]) >= STUCK_TIMEOUT_S


def corridor_lane_spawn_points(
    world_map,
    center_location,
    radar_positions,
    *,
    clearance_m,
    per_lane=8,
    spacing_m=12.0,
    step_m=4.0,
    max_back_m=160.0,
    reach_m=25.0,
    max_forward_m=200.0,
    return_directions=False,
):
    """Spawn transforms feeding the monitored corridor in BOTH travel directions,
    placed on each lane's approach just outside the radar clearance so cars drive
    THROUGH the radar zone.

    For every driving lane on the corridor road (both directions) we walk UPSTREAM
    (against travel) along the *straightest* route — at junction forks we follow the
    branch whose heading bends least, so we ride the corridor's natural approach
    (which may legitimately leave road 10 onto a connecting road for eastbound, or
    curve north for westbound) instead of turning onto a cross street. A spawn point
    is kept only when, walking FORWARD from it, the straightest route actually
    re-enters the radar zone (comes within ``reach_m`` of a radar). Direction
    (eastbound/westbound) is read from the travel heading at the closest approach to
    the corridor centre, so it is robust to curved approaches.

    With ``return_directions=True`` returns ``(transforms, dirs)`` where ``dirs[i]``
    is ``"E"`` or ``"W"``; otherwise returns just the transform list.
    """
    wp0 = world_map.get_waypoint(
        center_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp0 is None:
        return ([], []) if return_directions else []
    cx, cy = center_location.x, center_location.y
    # All driving lanes on the corridor road (both directions) via left/right neighbours.
    lanes, seen = [], set()
    cur = wp0
    while cur is not None and cur.lane_id not in seen and cur.lane_type == carla.LaneType.Driving:
        seen.add(cur.lane_id); lanes.append(cur); cur = cur.get_left_lane()
    cur = wp0.get_right_lane()
    while cur is not None and cur.lane_id not in seen and cur.lane_type == carla.LaneType.Driving:
        seen.add(cur.lane_id); lanes.append(cur); cur = cur.get_right_lane()

    clr_sq = clearance_m * clearance_m
    reach_sq = reach_m * reach_m

    def clear_of_radars(loc):
        if not radar_positions:
            return True
        return all((loc.x - rx) ** 2 + (loc.y - ry) ** 2 >= clr_sq for rx, ry in radar_positions)

    def straightest(cands, ref_yaw):
        """Pick the branch whose heading bends least from ref_yaw, preferring
        non-junction branches so we ride the through-route, not a turn."""
        if not cands:
            return None
        def cost(w):
            d = abs((w.transform.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0)
            return d + (90.0 if w.is_junction else 0.0)
        return min(cands, key=cost)

    def forward_probe(wp):
        """Walk FORWARD (straightest) from wp; return (reaches_zone, dir) where dir
        is 'E'/'W' from the travel heading nearest the corridor centre, or None."""
        cur = wp
        steps = int(max_forward_m / step_m)
        best_d2 = None
        best_yaw = None
        reached = False
        for _ in range(steps):
            loc = cur.transform.location
            for rx, ry in (radar_positions or [(cx, cy)]):
                if (loc.x - rx) ** 2 + (loc.y - ry) ** 2 <= reach_sq:
                    reached = True
                    break
            d2 = (loc.x - cx) ** 2 + (loc.y - cy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2; best_yaw = cur.transform.rotation.yaw
            nxt = straightest(cur.next(step_m), cur.transform.rotation.yaw)
            if nxt is None:
                break
            cur = nxt
        if best_yaw is None:
            return False, None
        direction = "E" if math.cos(math.radians(best_yaw)) >= 0.0 else "W"
        return reached, direction

    skip = max(1, int(round(spacing_m / step_m)))
    nmax = int(max_back_m / step_m)
    out, dirs = [], []
    for lane in lanes:
        cur, got, i = lane, 0, 0
        while i < nmax and got < per_lane:
            loc = cur.transform.location
            steps = 1
            if not cur.is_junction and clear_of_radars(loc):
                reaches, direction = forward_probe(cur)
                if reaches and direction is not None:
                    out.append(
                        carla.Transform(
                            carla.Location(loc.x, loc.y, loc.z + 0.5), cur.transform.rotation
                        )
                    )
                    dirs.append(direction)
                    got += 1
                    steps = skip
            moved = False
            for _ in range(steps):
                prevs = cur.previous(step_m)  # walk UPSTREAM toward the approach
                nxt = straightest(prevs, cur.transform.rotation.yaw)
                if nxt is None:
                    break
                cur = nxt; i += 1; moved = True
            if not moved:
                break
    return (out, dirs) if return_directions else out


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
    car_bps, truck_bps, motorcycle_bps, bicycle_bps = get_fleet_blueprint_pools(world)
    fractions = fleet_fractions_from_env()
    log_fleet_mix_requested(fractions)

    spawn_points = world_map.get_spawn_points()
    center_location = spawn_points[center_index].location
    exclusion_radius_m = spawn_exclusion_radius_m_from_env()
    outer_radius_m = spawn_radius_m_from_env()
    # Recycle a car once it is well past the spawn annulus (driven out the far side).
    recycle_radius_m = outer_radius_m + 20.0
    recycle_sq = recycle_radius_m * recycle_radius_m
    last_progress = {}  # actor_id -> (x, y, t_last_progress); for stuck detection
    max_dwell_s = actor_max_dwell_s_from_env()
    spawn_time = {aid: time.time() for aid in spawned_ids}  # actor_id -> first-seen t

    # Both-direction respawn pool: corridor lanes (E + W), straight through the zone.
    drive_in_points = corridor_lane_spawn_points(
        world_map, spawn_points[center_index].location, get_radar_positions(world),
        clearance_m=RADAR_SPAWN_CLEARANCE_M,
    )
    if not drive_in_points:
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
        now = time.time()
        for idx, actor_id in enumerate(list(spawned_ids)):
            try:
                actor = world.get_actor(actor_id)
                needs_recycle = actor is None or not actor.is_alive
                spawn_time.setdefault(actor_id, now)
                if not needs_recycle:
                    loc = actor.get_location()
                    if distance_sq(loc, center_location) > recycle_sq:
                        # Drove out the far side of the corridor.
                        needs_recycle = True
                    elif max_dwell_s and (now - spawn_time.get(actor_id, now)) > max_dwell_s:
                        # Dwell cap: cycle this actor so no single id dominates labels.
                        needs_recycle = True
                    elif _car_is_wedged(actor, loc, now, last_progress, spawned_ids, world):
                        # Genuinely wedged — NOT a red-light / queued / parked wait.
                        needs_recycle = True
                        print(
                            f"  [recirculation] car {actor_id} wedged "
                            f">{STUCK_TIMEOUT_S:.0f}s (not at light/queue); recycling",
                            flush=True,
                        )
                if not needs_recycle:
                    # Keep autopilot + policy fresh.
                    actor.set_autopilot(True, traffic_manager_port)
                    if free_vehicle_driving_from_env():
                        apply_free_driving_policy(traffic_manager, actor)
                    continue
                last_progress.pop(actor_id, None)  # forget recycled/dead id
                spawn_time.pop(actor_id, None)

                # Destroy the drifted/dead car.
                if actor is not None and actor.is_alive:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass

                # Respawn at a random drive-in point (retry a few collisions).
                new_actor = None
                for transform in random.sample(drive_in_points, min(len(drive_in_points), 8)):
                    bp = pick_fleet_bp(
                        car_bps, truck_bps, motorcycle_bps, bicycle_bps, fractions
                    )
                    if bp.has_attribute("color"):
                        bp.set_attribute(
                            "color",
                            random.choice(bp.get_attribute("color").recommended_values),
                        )
                    new_actor = world.try_spawn_actor(bp, transform)
                    if new_actor is not None:
                        try:
                            _apply_fleet_autopilot(
                                traffic_manager,
                                new_actor,
                                bp,
                                world_map,
                                transform,
                                traffic_manager_port,
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
                    spawn_time[new_actor.id] = now
                    recycle_count += 1
                    if recycle_count % 10 == 0:
                        print(f"  [recirculation] {recycle_count} cars recycled", flush=True)
            except (RuntimeError, AttributeError):
                continue
        time.sleep(poll_s)


def get_fleet_blueprint_pools(world):
    """Return (car_bps, truck_bps, motorcycle_bps, bicycle_bps) for TM fleet spawning."""
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    car_bps, truck_bps, motorcycle_bps, bicycle_bps = [], [], [], []

    for bp in blueprints:
        if bp.id.endswith("isetta"):
            continue
        wheels = None
        if bp.has_attribute("number_of_wheels"):
            wheels = int(bp.get_attribute("number_of_wheels").as_int())

        if bp_is_motorcycle(bp):
            if wheels is None or wheels == 2:
                motorcycle_bps.append(bp)
            continue
        if bp_is_bicycle(bp):
            if wheels is None or wheels == 2:
                bicycle_bps.append(bp)
            continue
        if wheels is not None and wheels != 4:
            continue
        if bp_is_truck(bp):
            truck_bps.append(bp)
        else:
            car_bps.append(bp)

    if not (car_bps or truck_bps or motorcycle_bps or bicycle_bps):
        raise RuntimeError("No usable vehicle blueprints found.")
    return car_bps, truck_bps, motorcycle_bps, bicycle_bps


def motorcycle_fraction_from_env(default=DEFAULT_MOTORCYCLE_FRACTION) -> float:
    raw = os.environ.get("DATASET_VEHICLE_MOTORCYCLE_FRACTION", "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    return default


def bicycle_fraction_from_env(default=DEFAULT_BICYCLE_FRACTION) -> float:
    raw = os.environ.get("DATASET_VEHICLE_BICYCLE_FRACTION", "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    return default


def fleet_fractions_from_env() -> dict[str, float]:
    """Return fleet fractions summing to 1.0.

    ``bicycle`` is the road/TM share; ``bicycle_sidewalk`` is the kinematic sidewalk
    share. Together they equal DATASET_VEHICLE_BICYCLE_FRACTION (default split 50/50).
    Clamp motorcycle + total bicycle first, then truck; remainder is cars."""
    motorcycle = motorcycle_fraction_from_env()
    bicycle_total = bicycle_fraction_from_env()
    truck = truck_fraction_from_env()
    sidewalk_share = bicycle_sidewalk_share_from_env()
    motorcycle = max(0.0, min(1.0, motorcycle))
    bicycle_total = max(0.0, min(1.0, bicycle_total))
    if motorcycle + bicycle_total > 1.0:
        scale = 1.0 / (motorcycle + bicycle_total)
        motorcycle *= scale
        bicycle_total *= scale
    truck = max(0.0, min(1.0 - motorcycle - bicycle_total, truck))
    car = max(0.0, 1.0 - motorcycle - bicycle_total - truck)
    road_bike = bicycle_total * (1.0 - sidewalk_share)
    sw_bike = bicycle_total * sidewalk_share
    return {
        "car": car,
        "truck": truck,
        "motorcycle": motorcycle,
        "bicycle": road_bike,
        "bicycle_sidewalk": sw_bike,
        "bicycle_total": bicycle_total,
    }


def sidewalk_bicycle_target_count(fleet_target: int, fractions: dict[str, float]) -> int:
    return max(0, round(fleet_target * fractions.get("bicycle_sidewalk", 0.0)))


def log_fleet_mix_requested(fractions: dict[str, float]) -> None:
    bt = fractions.get("bicycle_total", fractions["bicycle"])
    share = bicycle_sidewalk_share_from_env()
    print(
        f"Fleet mix (requested): car={fractions['car']:.1%} "
        f"truck={fractions['truck']:.1%} "
        f"motorcycle={fractions['motorcycle']:.1%} "
        f"bicycle={bt:.1%} (road={fractions['bicycle']:.1%} "
        f"sidewalk={fractions.get('bicycle_sidewalk', 0.0):.1%}, split={share:.0%})",
        flush=True,
    )


def log_fleet_class_summary(world, spawned_ids, *, label: str) -> None:
    counts: dict[str, int] = {}
    for actor_id in spawned_ids:
        try:
            actor = world.get_actor(actor_id)
            if actor is None or not actor.is_alive:
                continue
            cls = vehicle_class_from_type_id(actor.type_id)
            counts[cls] = counts.get(cls, 0) + 1
        except RuntimeError:
            continue
    total = sum(counts.values())
    if total == 0:
        print(f"{label}: (empty fleet)", flush=True)
        return
    parts = ", ".join(
        f"{cls}={n} ({n / total:.0%})" for cls, n in sorted(counts.items())
    )
    print(f"{label}: {parts}", flush=True)


def log_bicycle_placement_summary(
    world,
    spawned_ids,
    sidewalk_manager: BicycleSidewalkManager | None,
) -> None:
    """Road bicycles are in the TM fleet list; sidewalk bikes are kinematic-only."""
    road_bikes = 0
    for actor_id in spawned_ids:
        try:
            actor = world.get_actor(actor_id)
            if actor is None or not actor.is_alive:
                continue
            if vehicle_class_from_type_id(actor.type_id) == "bicycle":
                road_bikes += 1
        except RuntimeError:
            continue
    sidewalk_bikes = len(sidewalk_manager.bikes) if sidewalk_manager is not None else 0
    total = road_bikes + sidewalk_bikes
    if total == 0:
        print("Bicycle placement: (none)", flush=True)
        return
    print(
        f"Bicycle placement: road={road_bikes} (TM fleet), "
        f"sidewalk={sidewalk_bikes} (kinematic), total={total}",
        flush=True,
    )


def truck_fraction_from_env(default=DEFAULT_TRUCK_FRACTION) -> float:
    """Target truck fraction. DATASET_VEHICLE_TRUCK_FRACTION wins; else derived from
    DATASET_TARGET_TRUCK_COUNT vs the (total) fleet count; else the default."""
    raw = os.environ.get("DATASET_VEHICLE_TRUCK_FRACTION", "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    tc = os.environ.get("DATASET_TARGET_TRUCK_COUNT", "").strip()
    if tc:
        try:
            t = max(0, int(tc))
            total = max(1, target_car_count_from_env())
            return max(0.0, min(1.0, t / total))
        except ValueError:
            pass
    return default


def pick_fleet_bp(car_bps, truck_bps, motorcycle_bps, bicycle_bps, fractions):
    """Sample a fleet blueprint by class fraction, then uniform within class.

    Used at both initial fill and recirculation so the realized mix matches the target."""
    pools = [
        (fractions["motorcycle"], motorcycle_bps),
        (fractions["bicycle"], bicycle_bps),
        (fractions["truck"], truck_bps),
        (fractions["car"], car_bps),
    ]
    r = random.random()
    cum = 0.0
    for frac, pool in pools:
        cum += frac
        if r < cum:
            if pool:
                return random.choice(pool)
            break
    weighted = [(frac, pool) for frac, pool in pools if pool and frac > 0]
    if not weighted:
        all_pools = [p for p in (motorcycle_bps, bicycle_bps, truck_bps, car_bps) if p]
        if not all_pools:
            raise RuntimeError("No vehicle blueprints in any pool.")
        return random.choice(random.choice(all_pools))
    total = sum(frac for frac, _ in weighted)
    r2 = random.random() * total
    cum = 0.0
    for frac, pool in weighted:
        cum += frac
        if r2 < cum:
            return random.choice(pool)
    return random.choice(weighted[-1][1])


def bicycle_road_speed_reduction_pct_from_env(
    default=DEFAULT_BICYCLE_ROAD_SPEED_REDUCTION_PCT,
) -> float:
    raw = os.environ.get("DATASET_BICYCLE_ROAD_SPEED_REDUCTION_PCT", "").strip()
    if raw:
        try:
            return max(0.0, min(100.0, float(raw)))
        except ValueError:
            pass
    return default


def motorcycle_road_speed_reduction_pct_from_env(
    default=DEFAULT_MOTORCYCLE_ROAD_SPEED_REDUCTION_PCT,
) -> float:
    raw = os.environ.get("DATASET_MOTORCYCLE_ROAD_SPEED_REDUCTION_PCT", "").strip()
    if raw:
        try:
            return max(0.0, min(100.0, float(raw)))
        except ValueError:
            pass
    return default


def _apply_fleet_autopilot(
    traffic_manager,
    actor,
    bp,
    world_map,
    transform,
    traffic_manager_port,
) -> None:
    actor.set_autopilot(True, traffic_manager_port)
    if bp_is_bicycle(bp) and hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
        traffic_manager.vehicle_percentage_speed_difference(
            actor, bicycle_road_speed_reduction_pct_from_env()
        )
    if bp_is_motorcycle(bp) and hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
        mc_pct = motorcycle_road_speed_reduction_pct_from_env()
        if mc_pct > 0:
            traffic_manager.vehicle_percentage_speed_difference(actor, mc_pct)
    if free_vehicle_driving_from_env():
        apply_free_driving_policy(traffic_manager, actor, world_map, transform)
    else:
        apply_straight_driving_policy(traffic_manager, actor, world_map, transform)


def actor_max_dwell_s_from_env(default=0.0) -> float:
    """Force-recycle any fleet vehicle alive longer than this many seconds.

    Caps how many radar points a single actor can accumulate so no instance
    dominates the labels (one capture had a single truck = 30% of all points).
    0 / unset = disabled (legacy behavior). The campaign sets ~25 s."""
    raw = os.environ.get("DATASET_ACTOR_MAX_DWELL_S", "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val > 0 else 0.0


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
        # Both-direction feed: spawn on the corridor road's own lanes (E + W) just
        # outside radar clearance so cars drive STRAIGHT through the monitored zone.
        candidate_points, candidate_dirs = corridor_lane_spawn_points(
            world_map, center_transform.location, radar_positions,
            clearance_m=RADAR_SPAWN_CLEARANCE_M, return_directions=True,
        )
        # Shuffle points + their direction labels together so the initial fill mixes
        # directions but the labels stay aligned to their transforms.
        _paired = list(zip(candidate_points, candidate_dirs))
        random.shuffle(_paired)
        candidate_points = [p for p, _ in _paired]
        candidate_dirs = [d for _, d in _paired]
        if not candidate_points:
            # Fallback: generic annulus drive-in if the corridor road can't be walked.
            candidate_points = get_drive_in_spawn_points(
                world_map,
                spawn_points,
                center_transform,
                exclusion_radius_m=exclusion_radius_m,
                outer_radius_m=spawn_radius_m,
                radar_positions=radar_positions,
                radar_clearance_m=RADAR_SPAWN_CLEARANCE_M,
            )
            candidate_dirs = []  # generic fallback has no verified direction labels
        if not candidate_points:
            raise RuntimeError(
                f"No drive-in spawn points found for the corridor or the annulus "
                f"[{exclusion_radius_m:.0f}, {spawn_radius_m:.0f}] m of spawn index "
                f"{center_index}. Widen SPAWN_RADIUS_M or lower SPAWN_EXCLUSION_RADIUS_M."
            )
        if candidate_dirs:
            _eb = sum(1 for d in candidate_dirs if d == "E")
            _dir_str = f"eastbound={_eb}, westbound={len(candidate_dirs) - _eb}"
        else:
            _dir_str = "generic fallback (no direction labels)"
        print(
            f"Spawn mode: DRIVE-IN (corridor both-way) | centre index={center_index} "
            f"({center_transform.location.x:.1f}, {center_transform.location.y:.1f}) | "
            f"candidates={len(candidate_points)} ({_dir_str})",
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

    car_bps, truck_bps, motorcycle_bps, bicycle_bps = get_fleet_blueprint_pools(world)
    fractions = fleet_fractions_from_env()
    log_fleet_mix_requested(fractions)
    spawned_ids = []
    labeled_actors = []

    for transform in candidate_points:
        if len(spawned_ids) >= target_count:
            break

        # Exponential inter-arrival delay => Poisson spawn process.
        delay_s = random.expovariate(spawn_rate_per_second)
        time.sleep(delay_s)

        bp = pick_fleet_bp(car_bps, truck_bps, motorcycle_bps, bicycle_bps, fractions)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        if bp.has_attribute("driver_id"):
            driver_id = random.choice(bp.get_attribute("driver_id").recommended_values)
            bp.set_attribute("driver_id", driver_id)

        is_two_wheeler = bp_is_bicycle(bp) or bp_is_motorcycle(bp)
        spawn_transforms = [transform]
        if is_two_wheeler and len(candidate_points) > 1:
            others = [t for t in candidate_points if t != transform]
            extra = random.sample(others, min(TWO_WHEELER_SPAWN_ATTEMPTS - 1, len(others)))
            spawn_transforms.extend(extra)

        actor = None
        used_transform = transform
        for try_tf in spawn_transforms:
            actor = world.try_spawn_actor(bp, try_tf)
            if actor is not None:
                used_transform = try_tf
                break
        if actor is None:
            print(f"Skipped blocked spawn point after {delay_s:.2f}s delay.")
            continue

        try:
            _apply_fleet_autopilot(
                traffic_manager, actor, bp, world_map, used_transform, traffic_manager_port
            )
        except RuntimeError:
            print("Spawned actor could not enable autopilot; destroying and skipping.")
            if actor.is_alive:
                actor.destroy()
            continue

        moved_away, reason = wait_until_vehicle_moves_away(
            actor,
            used_transform.location,
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

    log_fleet_class_summary(world, spawned_ids, label="Fleet mix (realized after fill)")

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

    fractions = fleet_fractions_from_env()
    sw_target = sidewalk_bicycle_target_count(target_car_count_from_env(), fractions)
    sidewalk_manager = None
    if sw_target > 0:
        _, _, _, bicycle_bps = get_fleet_blueprint_pools(world)
        if bicycle_bps:
            sidewalk_manager = BicycleSidewalkManager.from_world(
                world,
                bicycle_bps,
                target_count=sw_target,
                max_dwell_s=actor_max_dwell_s_from_env(),
                client=client,
            )
            sidewalk_manager.spawn_up_to(sw_target)

    log_bicycle_placement_summary(world, ids, sidewalk_manager)

    try:
        if ids:
            print("Spawned vehicle actor IDs:", ", ".join(str(actor_id) for actor_id in ids))
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
                if sidewalk_manager is not None:
                    threading.Thread(
                        target=sidewalk_manager.run_until_interrupt,
                        kwargs={"poll_s": 0.05},
                        daemon=True,
                    ).start()
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
    finally:
        if sidewalk_manager is not None:
            destroyed = sidewalk_manager.destroy_all()
            if destroyed:
                print(
                    f"[bicycle/sidewalk] Destroyed {destroyed} sidewalk bicycle(s) on exit.",
                    flush=True,
                )


if __name__ == "__main__":
    main()
