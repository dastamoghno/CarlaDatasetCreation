"""Kinematic sidewalk bicycles on the pedestrian navmesh.

CARLA Traffic Manager cannot route vehicles on sidewalks. Each bicycle spawns on a
navmesh point (same sampler as ``SpawnPedestriansAcrossMap``), builds a walkable path
toward a random nav target, and moves with ``set_transform`` (physics off — velocity
commands do not work in that mode).

No walker pilots — a single bicycle actor avoids the stacked ped+bike glitch.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field

import carla

from world.SpawnPedestriansAcrossMap import (
    get_nav_location_in_region,
    ped_retarget_interval_s_from_env,
    ped_spawn_region_from_env,
)
from world.vehicle_classes import SIDEWALK_BICYCLE_ROLE

CORRIDOR_Y = -61.0
CROSS_X_MIN = -25.0
CROSS_X_MAX = 60.0
CROSS_Y_SOUTH = -72.0
CROSS_Y_NORTH = -54.0
CORRIDOR_X_MIN = CROSS_X_MIN - 5.0
CORRIDOR_X_MAX = CROSS_X_MAX + 5.0

BIKE_NAV_PILOT_ROLE = "bicycle_nav_pilot"
DEFAULT_SIDEWALK_SPEED_MPS = 1.6
DEFAULT_SIDEWALK_SHARE = 0.5
PATH_STEP_M = 2.0
PATH_ARRIVE_M = 0.8
PATH_MAX_POINTS = 180
STUCK_RETARGET_S = 8.0
STUCK_MOVE_M = 0.5
SPAWN_ATTEMPTS_PER_BIKE = 25
TRAFFIC_MANAGER_PORT = 8000


def bicycle_sidewalk_share_from_env(default=DEFAULT_SIDEWALK_SHARE) -> float:
    raw = os.environ.get("DATASET_BICYCLE_SIDEWALK_FRACTION", "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    return default


def sidewalk_speed_mps_from_env(default=DEFAULT_SIDEWALK_SPEED_MPS) -> float:
    raw = os.environ.get("DATASET_BICYCLE_SIDEWALK_SPEED_MPS", "").strip()
    if raw:
        try:
            return max(0.5, min(6.0, float(raw)))
        except ValueError:
            pass
    return default


def is_bicycle_nav_pilot(actor) -> bool:
    """Legacy walker pilots from older builds — excluded from radar if any remain."""
    try:
        return (
            actor.type_id.startswith("walker.pedestrian.")
            and actor.attributes.get("role_name", "") == BIKE_NAV_PILOT_ROLE
        )
    except RuntimeError:
        return False


def cleanup_leftover_nav_pilots(world) -> int:
    """Remove walker pilots left over from older sidewalk-bicycle builds."""
    n = 0
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        if not is_bicycle_nav_pilot(actor):
            continue
        try:
            if actor.is_alive and actor.destroy():
                n += 1
        except RuntimeError:
            pass
    return n


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


def bicycle_nav_region_from_env():
    if _bool_env("DATASET_BICYCLE_MAP_WIDE_NAV", False):
        return None
    return ped_spawn_region_from_env()


def _sample_nav_location(world, region):
    loc = get_nav_location_in_region(world, region)
    if loc is not None:
        return loc
    if region is not None:
        return get_nav_location_in_region(world, None)
    return None


def _dist2d(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _yaw_toward(src: carla.Location, dst: carla.Location) -> float:
    return math.degrees(math.atan2(dst.y - src.y, dst.x - src.x))


def _is_walkable_lane(wp) -> bool:
    if wp is None:
        return False
    if wp.lane_type == carla.LaneType.Sidewalk:
        return True
    biking = getattr(carla.LaneType, "Biking", None)
    return biking is not None and wp.lane_type == biking


def _ground_z(world_map, x: float, y: float, fallback_z: float) -> float:
    for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Driving):
        wp = world_map.get_waypoint(
            carla.Location(x, y, fallback_z if fallback_z else 0.5),
            project_to_road=True,
            lane_type=lane_type,
        )
        if wp is not None:
            return wp.transform.location.z + 0.35
    return fallback_z + 0.35 if fallback_z else 0.85


def _sidewalk_waypoint(world_map, loc: carla.Location):
    for project in (False, True):
        wp = world_map.get_waypoint(
            carla.Location(loc.x, loc.y, loc.z if loc.z else 0.5),
            project_to_road=project,
            lane_type=carla.LaneType.Sidewalk,
        )
        if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
            return wp
    wp = world_map.get_waypoint(loc, project_to_road=True)
    if wp is None:
        return None
    for first in (wp.get_left_lane(), wp.get_right_lane()):
        side = first
        for _ in range(4):
            if side is None:
                break
            if _is_walkable_lane(side):
                return side
            side = side.get_left_lane() if side == first else side.get_right_lane()
    return None


def _append_path_point(path: list[carla.Location], loc: carla.Location, *, min_step_m: float = 0.5) -> None:
    if not path or _dist2d(path[-1], loc) >= min_step_m:
        path.append(loc)


def _greedy_sidewalk_path(
    world_map,
    start_loc: carla.Location,
    end_loc: carla.Location,
    *,
    step_m: float = PATH_STEP_M,
    max_points: int = PATH_MAX_POINTS,
) -> list[carla.Location]:
    start_wp = _sidewalk_waypoint(world_map, start_loc)
    if start_wp is None:
        return []

    path: list[carla.Location] = [start_wp.transform.location]
    wp = start_wp
    seen: set[tuple[int, int]] = {(round(path[0].x), round(path[0].y))}

    for _ in range(max_points - 1):
        if _dist2d(path[-1], end_loc) <= step_m * 1.25:
            _append_path_point(path, end_loc, min_step_m=0.15)
            break

        nxt_list = wp.next(step_m) or wp.next(step_m * 0.5)
        if not nxt_list:
            break

        candidates = [w for w in nxt_list if _is_walkable_lane(w)]
        if not candidates:
            break

        best = min(candidates, key=lambda w: _dist2d(w.transform.location, end_loc))
        key = (round(best.transform.location.x), round(best.transform.location.y))
        if key in seen:
            break
        seen.add(key)

        if _dist2d(best.transform.location, path[-1]) < 0.15:
            break

        wp = best
        path.append(wp.transform.location)

    if path and _dist2d(path[-1], end_loc) > PATH_ARRIVE_M:
        _append_path_point(path, end_loc, min_step_m=0.15)
    return path


def _interpolated_sidewalk_path(
    world_map,
    start_loc: carla.Location,
    end_loc: carla.Location,
    *,
    step_m: float = PATH_STEP_M,
) -> list[carla.Location]:
    """Sample along the chord start→end and snap each sample to the nearest sidewalk."""
    dist = _dist2d(start_loc, end_loc)
    if dist < step_m:
        return [start_loc, end_loc]

    steps = max(2, min(int(dist / step_m) + 1, PATH_MAX_POINTS))
    path: list[carla.Location] = []
    for i in range(steps):
        t = i / (steps - 1)
        ix = start_loc.x + (end_loc.x - start_loc.x) * t
        iy = start_loc.y + (end_loc.y - start_loc.y) * t
        wp = _sidewalk_waypoint(world_map, carla.Location(ix, iy, start_loc.z))
        if wp is not None:
            _append_path_point(path, wp.transform.location)
        else:
            _append_path_point(path, carla.Location(ix, iy, start_loc.z))

    if path and _dist2d(path[-1], end_loc) > PATH_ARRIVE_M:
        _append_path_point(path, end_loc, min_step_m=0.15)
    return path


def _corridor_stroll_path(start_loc: carla.Location) -> list[carla.Location]:
    """East-west curb stroll in the radar corridor (matches pedestrian STROLL band)."""
    y = CROSS_Y_SOUTH if abs(start_loc.y - CROSS_Y_SOUTH) <= abs(start_loc.y - CROSS_Y_NORTH) else CROSS_Y_NORTH
    x0 = max(CORRIDOR_X_MIN, min(CORRIDOR_X_MAX, start_loc.x))
    x1 = random.uniform(CROSS_X_MIN, CROSS_X_MAX)
    if abs(x1 - x0) < PATH_STEP_M:
        x1 = CROSS_X_MIN if x0 > (CROSS_X_MIN + CROSS_X_MAX) * 0.5 else CROSS_X_MAX

    path = [carla.Location(x0, y, start_loc.z)]
    dx = 1.0 if x1 >= x0 else -1.0
    x = x0
    while (dx > 0 and x < x1) or (dx < 0 and x > x1):
        x += dx * PATH_STEP_M
        if (dx > 0 and x >= x1) or (dx < 0 and x <= x1):
            x = x1
        _append_path_point(path, carla.Location(x, y, start_loc.z))
    return path


def build_navmesh_sidewalk_path(
    world_map,
    start_loc: carla.Location,
    end_loc: carla.Location,
    *,
    step_m: float = PATH_STEP_M,
    max_points: int = PATH_MAX_POINTS,
) -> list[carla.Location]:
    """Walkable path toward a pedestrian navmesh destination."""
    del max_points  # greedy builder uses PATH_MAX_POINTS internally
    greedy = _greedy_sidewalk_path(world_map, start_loc, end_loc, step_m=step_m)
    if len(greedy) >= 4:
        return greedy

    interp = _interpolated_sidewalk_path(world_map, start_loc, end_loc, step_m=step_m)
    if len(interp) >= 3:
        return interp

    if greedy:
        return greedy
    if interp:
        return interp

    stroll = _corridor_stroll_path(start_loc)
    if len(stroll) >= 2:
        return stroll
    return [start_loc, end_loc]


def discover_sidewalk_routes(world, *, min_path_pts: int = 4, max_routes: int = 6):
    region = bicycle_nav_region_from_env()
    world_map = world.get_map()
    routes = []
    attempts = 0
    while len(routes) < max_routes and attempts < max_routes * 10:
        attempts += 1
        start = _sample_nav_location(world, region)
        end = _sample_nav_location(world, region)
        if start is None or end is None:
            continue
        path = build_navmesh_sidewalk_path(world_map, start, end)
        if len(path) < min_path_pts:
            continue
        yaw = _yaw_toward(start, path[1] if len(path) > 1 else end)
        tf = carla.Transform(
            carla.Location(start.x, start.y, start.z + 0.35),
            carla.Rotation(yaw=yaw),
        )
        routes.append((f"nav_{len(routes)}", tf, path))
    return routes


def _prepare_kinematic(actor) -> None:
    try:
        actor.set_autopilot(False, TRAFFIC_MANAGER_PORT)
    except RuntimeError:
        pass
    actor.set_simulate_physics(False)


def advance_along_path(
    actor: carla.Actor,
    world_map,
    path: list[carla.Location],
    path_index: int,
    *,
    speed_mps: float,
    dt_s: float,
) -> int:
    if not path or dt_s <= 0.0:
        return path_index

    while path_index < len(path):
        target = path[path_index]
        loc = actor.get_location()
        dist = _dist2d(loc, target)
        if dist <= PATH_ARRIVE_M:
            path_index += 1
            continue

        step = min(max(speed_mps * dt_s, 0.04), dist)
        t = step / dist
        nx = loc.x + (target.x - loc.x) * t
        ny = loc.y + (target.y - loc.y) * t
        nz = _ground_z(world_map, nx, ny, target.z)
        yaw = _yaw_toward(loc, target)
        actor.set_transform(
            carla.Transform(
                carla.Location(nx, ny, nz),
                carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
            )
        )
        return path_index

    return path_index


@dataclass
class _SidewalkBike:
    actor: carla.Actor
    path: list[carla.Location] = field(default_factory=list)
    path_index: int = 0
    spawn_time: float = field(default_factory=time.time)
    last_retarget: float = field(default_factory=time.time)
    last_progress: float = field(default_factory=time.time)
    last_path_index: int = 0
    last_loc: carla.Location | None = None


class BicycleSidewalkManager:
    """Single-actor sidewalk bicycles following pedestrian navmesh targets."""

    def __init__(
        self,
        world,
        bicycle_bps,
        *,
        target_count: int,
        max_dwell_s: float = 0.0,
        speed_mps: float | None = None,
        nav_region=None,
        client=None,
    ) -> None:
        del client
        self.world = world
        self.world_map = world.get_map()
        self.bicycle_bps = list(bicycle_bps)
        self.target_count = max(0, target_count)
        self.max_dwell_s = max(0.0, max_dwell_s)
        self.speed_mps = speed_mps if speed_mps is not None else sidewalk_speed_mps_from_env()
        self.nav_region = nav_region if nav_region is not None else bicycle_nav_region_from_env()
        self.retarget_interval_s = ped_retarget_interval_s_from_env()
        self.bikes: list[_SidewalkBike] = []
        self._last_spawn_error = ""

        removed = cleanup_leftover_nav_pilots(world)
        if removed:
            print(f"[bicycle/sidewalk] Removed {removed} legacy nav-pilot walker(s).", flush=True)

    @classmethod
    def from_world(
        cls,
        world,
        bicycle_bps,
        *,
        target_count: int,
        max_dwell_s: float = 0.0,
        client=None,
    ):
        return cls(
            world,
            bicycle_bps,
            target_count=target_count,
            max_dwell_s=max_dwell_s,
            client=client,
        )

    def _assign_route(self, entry: _SidewalkBike) -> bool:
        end = _sample_nav_location(self.world, self.nav_region)
        if end is None:
            end = _sample_nav_location(self.world, None)
        if end is None:
            return False
        try:
            start = entry.actor.get_location()
        except RuntimeError:
            return False
        if start.x == 0.0 and start.y == 0.0:
            return False
        path = build_navmesh_sidewalk_path(self.world_map, start, end)
        if len(path) < 2:
            path = _corridor_stroll_path(start)
        if len(path) < 2:
            return False
        entry.path = path
        entry.path_index = 0
        entry.last_path_index = 0
        entry.last_loc = start
        return True

    def _spawn_transform(self, nav_loc: carla.Location, path: list[carla.Location]) -> carla.Transform:
        # Same spawn height as AI pedestrians (navmesh z), not a forced sidewalk snap.
        goal = path[1] if len(path) > 1 else nav_loc
        yaw = _yaw_toward(nav_loc, goal)
        return carla.Transform(
            carla.Location(nav_loc.x, nav_loc.y, nav_loc.z + 0.35),
            carla.Rotation(yaw=yaw),
        )

    def spawn_one(self) -> bool:
        if not self.bicycle_bps:
            self._last_spawn_error = "no bicycle blueprints"
            return False

        nav_loc = _sample_nav_location(self.world, self.nav_region)
        if nav_loc is None:
            nav_loc = _sample_nav_location(self.world, None)
        if nav_loc is None:
            self._last_spawn_error = "no navmesh location"
            return False

        end = _sample_nav_location(self.world, self.nav_region)
        if end is None:
            end = _sample_nav_location(self.world, None)
        if end is None:
            self._last_spawn_error = "no navmesh target"
            return False

        path = build_navmesh_sidewalk_path(self.world_map, nav_loc, end)
        if len(path) < 2:
            path = _corridor_stroll_path(nav_loc)
        if len(path) < 2:
            self._last_spawn_error = "path too short"
            return False

        tf = self._spawn_transform(nav_loc, path)
        bp = random.choice(self.bicycle_bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", SIDEWALK_BICYCLE_ROLE)
        actor = self.world.try_spawn_actor(bp, tf)
        if actor is None:
            self._last_spawn_error = "bicycle spawn collision"
            return False

        _prepare_kinematic(actor)
        spawn_loc = actor.get_location()
        if _dist2d(spawn_loc, path[0]) > 1.0:
            path = [spawn_loc] + path

        now = time.time()
        self.bikes.append(
            _SidewalkBike(
                actor=actor,
                path=path,
                spawn_time=now,
                last_retarget=now,
                last_progress=now,
                last_loc=spawn_loc,
            )
        )
        self._last_spawn_error = ""
        return True

    def spawn_up_to(self, count: int) -> int:
        spawned = 0
        attempts = 0
        max_attempts = max(count * SPAWN_ATTEMPTS_PER_BIKE, count)
        while spawned < count and attempts < max_attempts:
            attempts += 1
            if self.spawn_one():
                spawned += 1
        if spawned:
            scope = (
                "map-wide navmesh"
                if self.nav_region is None
                else f"ped region r={self.nav_region[2]:.0f}m"
            )
            print(
                f"[bicycle/sidewalk] Spawned {spawned}/{count} on pedestrian navmesh "
                f"({scope}, {self.speed_mps:.1f} m/s kinematic).",
                flush=True,
            )
        elif count > 0:
            print(
                f"[bicycle/sidewalk] Failed after {attempts} attempt(s): "
                f"{self._last_spawn_error or 'unknown'}",
                flush=True,
            )
        return spawned

    def _destroy_entry(self, entry: _SidewalkBike) -> None:
        try:
            if entry.actor.is_alive:
                entry.actor.destroy()
        except RuntimeError:
            pass

    def _maybe_retarget(self, entry: _SidewalkBike, now: float) -> None:
        try:
            loc = entry.actor.get_location()
        except RuntimeError:
            return

        if entry.last_loc is not None and _dist2d(loc, entry.last_loc) >= STUCK_MOVE_M:
            entry.last_loc = loc
            entry.last_progress = now

        if entry.path_index > entry.last_path_index:
            entry.last_path_index = entry.path_index
            entry.last_progress = now
            return

        stuck = now - entry.last_progress >= STUCK_RETARGET_S
        due = now - entry.last_retarget >= self.retarget_interval_s
        finished = bool(entry.path) and entry.path_index >= len(entry.path)
        if stuck or due or finished:
            if self._assign_route(entry):
                entry.last_retarget = now
                entry.last_progress = now

    def _tick(self, entry: _SidewalkBike, dt_s: float, now: float) -> None:
        try:
            if not entry.actor.is_alive:
                return
            if not entry.path:
                self._assign_route(entry)
                return
            entry.path_index = advance_along_path(
                entry.actor,
                self.world_map,
                entry.path,
                entry.path_index,
                speed_mps=self.speed_mps,
                dt_s=dt_s,
            )
            self._maybe_retarget(entry, now)
        except RuntimeError:
            pass

    def _needs_recycle(self, entry: _SidewalkBike, now: float) -> bool:
        try:
            if not entry.actor.is_alive:
                return True
            if self.max_dwell_s and (now - entry.spawn_time) > self.max_dwell_s:
                return True
        except RuntimeError:
            return True
        return False

    def tick_and_maintain(self, dt_s: float = 0.05) -> None:
        now = time.time()
        alive: list[_SidewalkBike] = []
        for entry in self.bikes:
            if self._needs_recycle(entry, now):
                self._destroy_entry(entry)
                continue
            self._tick(entry, dt_s, now)
            alive.append(entry)
        self.bikes = alive
        attempts = 0
        while len(self.bikes) < self.target_count and attempts < SPAWN_ATTEMPTS_PER_BIKE:
            attempts += 1
            if not self.spawn_one():
                break

    def _tick_delta_s(self) -> float:
        try:
            return max(0.02, self.world.get_snapshot().timestamp.delta_seconds)
        except RuntimeError:
            return 0.05

    def run_until_interrupt(self, poll_s: float = 0.05) -> None:
        while True:
            try:
                self.world.wait_for_tick()
            except RuntimeError:
                time.sleep(poll_s)
            self.tick_and_maintain(self._tick_delta_s())

    def destroy_all(self) -> int:
        n = len(self.bikes)
        for entry in list(self.bikes):
            self._destroy_entry(entry)
        self.bikes.clear()
        return n
