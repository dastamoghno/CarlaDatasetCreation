#!/usr/bin/env python3
"""Quick bicycle spawn smoke test for Town10HD_Opt.

Checks that CARLA bicycle blueprints spawn and move on:
  - road (Traffic Manager autopilot on Driving lanes)
  - sidewalk (navmesh targets + kinematic path follow, like pedestrians)

Usage (CARLA running, Town10HD_Opt loaded):
  python tools/test_bicycle_spawn.py --skip-probe --sidewalk 4
  python tools/test_bicycle_spawn.py --draw-paths --sidewalk 3
  python tools/test_bicycle_spawn.py --road 2 --sidewalk 2   # compare road TM vs sidewalk

Press Enter or Ctrl+C to stop and destroy spawned actors.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc)
_dc.bootstrap(__file__)

import carla

from carla_connect import get_world
from _kbhit_compat import enter_pressed
from world.SpawnCarsAtPosition14 import (
    SPAWN_CENTER_INDEX,
    apply_free_driving_policy,
    get_fleet_blueprint_pools,
    safe_following_distance_m_from_env,
)
from world.bicycle_sidewalk import (
    CORRIDOR_X_MAX,
    CORRIDOR_X_MIN,
    CORRIDOR_Y,
    CROSS_Y_NORTH,
    CROSS_Y_SOUTH,
    BicycleSidewalkManager,
    cleanup_leftover_nav_pilots,
    discover_sidewalk_routes,
)

TRAFFIC_MANAGER_PORT = 8000
BICYCLE_SPEED_REDUCTION_PCT = 40.0
BICYCLE_SIDEWALK_TARGET_MPS = 1.6
JUNCTION_CROSSWALK = carla.Location(-31.0, -61.0, 0.5)


@dataclass
class SpawnResult:
    blueprint_id: str
    mode: str
    location: str
    spawned: bool
    autopilot_ok: bool = False
    moved_m: float = 0.0
    alive: bool = False
    notes: str = ""


@dataclass
class LiveActor:
    actor: carla.Actor
    label: str
    mode: str  # "road" | "sidewalk"
    spawn_loc: carla.Location
    path: list[carla.Location] = field(default_factory=list)
    path_index: int = 0


def _wheel_count(bp) -> str:
    if bp.has_attribute("number_of_wheels"):
        return str(int(bp.get_attribute("number_of_wheels").as_int()))
    return "?"


def _dist_m(a: carla.Location, b: carla.Location) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _in_corridor_vicinity(loc: carla.Location) -> bool:
    return (
        CORRIDOR_X_MIN <= loc.x <= CORRIDOR_X_MAX
        and CROSS_Y_SOUTH - 2.0 <= loc.y <= CROSS_Y_NORTH + 2.0
    )


def _lane_kind(world_map, loc: carla.Location) -> str:
    """Lane at the actor's actual position (no forced snap to sidewalk)."""
    wp = world_map.get_waypoint(loc, project_to_road=False)
    if wp is None:
        wp = world_map.get_waypoint(loc, project_to_road=True)
    if wp is None:
        return "unknown"
    snap = _dist_m(loc, wp.transform.location)
    kind = wp.lane_type.name.lower()
    if snap > 4.0:
        return f"offlane({snap:.0f}m)"
    return kind


def _draw_paths(world, routes, life_s: float = 300.0) -> None:
    debug = world.debug
    for name, _, path in routes:
        color = carla.Color(0, 200, 80)
        for a, b in zip(path, path[1:]):
            debug.draw_line(
                carla.Location(a.x, a.y, a.z + 0.4),
                carla.Location(b.x, b.y, b.z + 0.4),
                thickness=0.08,
                color=color,
                life_time=life_s,
            )


def _road_probe_transforms(world) -> list[tuple[str, carla.Transform]]:
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    transforms: list[tuple[str, carla.Transform]] = []

    if spawn_points:
        idx = min(SPAWN_CENTER_INDEX, len(spawn_points) - 1)
        transforms.append(("spawn_index_144", spawn_points[idx]))

    for loc, name in (
        (JUNCTION_CROSSWALK, "junction_crosswalk"),
        (carla.Location(10.0, CORRIDOR_Y, 0.5), "corridor_mid"),
        (carla.Location(40.0, CORRIDOR_Y, 0.5), "corridor_east"),
    ):
        wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is not None:
            tf = wp.transform
            tf.location.z += 0.3
            transforms.append((name, tf))

    return transforms


def _apply_bicycle_tm(traffic_manager, actor, world_map, transform) -> bool:
    try:
        actor.set_autopilot(True, TRAFFIC_MANAGER_PORT)
    except RuntimeError:
        return False
    if hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
        traffic_manager.vehicle_percentage_speed_difference(
            actor, BICYCLE_SPEED_REDUCTION_PCT
        )
    if hasattr(traffic_manager, "distance_to_leading_vehicle"):
        traffic_manager.distance_to_leading_vehicle(
            actor, safe_following_distance_m_from_env()
        )
    apply_free_driving_policy(traffic_manager, actor, world_map, transform)
    return True


def _prepare_kinematic_bicycle(actor) -> None:
    """Physics/throttle on sidewalks is unreliable — drive like a guided crosser."""
    try:
        actor.set_autopilot(False, TRAFFIC_MANAGER_PORT)
    except RuntimeError:
        pass
    actor.set_simulate_physics(False)


def _probe_blueprint(
    world,
    traffic_manager,
    bp,
    transform: carla.Transform,
    mode: str,
    loc_name: str,
    *,
    path: list[carla.Location] | None = None,
    settle_s: float = 8.0,
) -> SpawnResult:
    world_map = world.get_map()
    result = SpawnResult(
        blueprint_id=bp.id,
        mode=mode,
        location=loc_name,
        spawned=False,
    )
    actor = world.try_spawn_actor(bp, transform)
    if actor is None:
        result.notes = "try_spawn_actor returned None (collision / invalid pose)"
        return result

    result.spawned = True
    spawn_loc = actor.get_location()
    spawn_lane = _lane_kind(world_map, spawn_loc)

    if mode == "road":
        if not _apply_bicycle_tm(traffic_manager, actor, world_map, transform):
            result.notes = "set_autopilot failed"
    elif mode == "sidewalk":
        _prepare_kinematic_bicycle(actor)
        if spawn_lane != "sidewalk":
            result.notes = f"spawn lane={spawn_lane} (expected sidewalk)"
        deadline = time.time() + settle_s
        while time.time() < deadline:
            time.sleep(0.05)
    else:
        result.notes = f"unknown mode {mode}"

    try:
        if actor.is_alive:
            end_loc = actor.get_location()
            result.moved_m = _dist_m(spawn_loc, end_loc)
            result.alive = True
            end_lane = _lane_kind(world_map, end_loc)
            if mode == "road" and result.moved_m < 1.0:
                result.notes = f"road: barely moved ({result.moved_m:.1f} m)"
            elif mode == "sidewalk":
                result.notes = (
                    f"spawn={spawn_lane} end={end_lane} moved={result.moved_m:.1f}m; "
                    + result.notes
                )
    except RuntimeError:
        result.alive = False
        result.notes = "actor died during settle"

    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass
    return result


def run_probe(world, bicycle_bps, sidewalk_routes, *, settle_s: float) -> list[SpawnResult]:
    client, _ = get_world()
    traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)

    road_transforms = _road_probe_transforms(world)
    results: list[SpawnResult] = []

    print(f"\n=== Bicycle blueprint pool ({len(bicycle_bps)}) ===")
    for bp in bicycle_bps:
        print(f"  {bp.id}  wheels={_wheel_count(bp)}")

    if not bicycle_bps:
        print("No bicycle blueprints found — check CARLA version / blueprint filter.")
        return results

    print(f"\n=== Navmesh sample routes: {len(sidewalk_routes)} ===")
    for name, tf, path in sidewalk_routes[:8]:
        print(
            f"  {name}: ({tf.location.x:.1f}, {tf.location.y:.1f}) "
            f"nav_pts={len(path)}"
        )
    if len(sidewalk_routes) > 8:
        print(f"  ... and {len(sidewalk_routes) - 8} more")

    print(f"\n=== Road spawn + TM probe ({settle_s:.0f}s settle each) ===")
    for bp in bicycle_bps:
        tf_name, tf = road_transforms[0]
        results.append(
            _probe_blueprint(
                world, traffic_manager, bp, tf, "road", tf_name, settle_s=settle_s
            )
        )

    if sidewalk_routes:
        print(f"\n=== Sidewalk spawn probe SKIPPED (use live --sidewalk test) ===")
    else:
        print(
            "\n=== Sidewalk probe SKIPPED: no navmesh sample routes ===",
            flush=True,
        )

    print("\n=== Probe summary ===")
    print(f"{'blueprint':<35} {'mode':<9} {'loc':<16} {'spawn':<6} {'move_m':>7}  notes")
    for r in results:
        print(
            f"{r.blueprint_id:<35} {r.mode:<9} {r.location:<16} "
            f"{'yes' if r.spawned else 'NO':<6} {r.moved_m:7.1f}  {r.notes}"
        )
    return results


def spawn_live_actors(
    world,
    bicycle_bps,
    *,
    road_count: int,
    sidewalk_count: int,
) -> tuple[list[LiveActor], BicycleSidewalkManager | None]:
    if not bicycle_bps:
        raise RuntimeError("No bicycle blueprints available.")

    client, _ = get_world()
    traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    world_map = world.get_map()
    road_transforms = _road_probe_transforms(world)
    live: list[LiveActor] = []

    for i in range(road_count):
        bp = random.choice(bicycle_bps)
        tf_name, tf = random.choice(road_transforms)
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            print(f"[road {i + 1}] spawn FAILED at {tf_name} ({bp.id})")
            continue
        if not _apply_bicycle_tm(traffic_manager, actor, world_map, tf):
            print(f"[road {i + 1}] autopilot FAILED ({bp.id}); destroying")
            actor.destroy()
            continue
        label = f"ROAD-{i + 1:02d}"
        live.append(LiveActor(actor=actor, label=label, mode="road", spawn_loc=tf.location))
        print(f"[road {i + 1}] {label} id={actor.id} bp={bp.id} at {tf_name} (Driving/TM)")

    sidewalk_mgr: BicycleSidewalkManager | None = None
    if sidewalk_count > 0:
        sidewalk_mgr = BicycleSidewalkManager.from_world(
            world, bicycle_bps, target_count=sidewalk_count, client=client
        )
        sidewalk_mgr.spawn_up_to(sidewalk_count)
        for i, entry in enumerate(sidewalk_mgr.bikes):
            loc = entry.actor.get_location()
            lane = _lane_kind(world_map, loc)
            label = f"SW-{i + 1:02d}"
            live.append(
                LiveActor(
                    actor=entry.actor,
                    label=label,
                    mode="sidewalk",
                    spawn_loc=loc,
                )
            )
            print(
                f"[sidewalk {i + 1}] {label} id={entry.actor.id} "
                f"lane={lane} path_pts={len(entry.path)} (navmesh kinematic)"
            )

    return live, sidewalk_mgr


def _print_live_status(
    world,
    live: list[LiveActor],
    sidewalk_mgr: BicycleSidewalkManager | None = None,
) -> None:
    world_map = world.get_map()
    parts = []
    for entry in live:
        try:
            if not entry.actor.is_alive:
                parts.append(f"{entry.label}=DEAD")
                continue
            loc = entry.actor.get_location()
            vel = entry.actor.get_velocity()
            phys_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            moved = _dist_m(entry.spawn_loc, loc)
            lane = _lane_kind(world_map, loc)
            in_band = _in_corridor_vicinity(loc)
            if entry.mode == "sidewalk":
                wp = ""
                if sidewalk_mgr is not None:
                    for mgr_entry in sidewalk_mgr.bikes:
                        if mgr_entry.actor.id == entry.actor.id:
                            wp = f" wp {mgr_entry.path_index}/{len(mgr_entry.path)}"
                            break
                parts.append(
                    f"{entry.label}@({loc.x:.0f},{loc.y:.0f}) lane={lane} "
                    f"corridor={'Y' if in_band else 'N'}{wp} moved={moved:.0f}m"
                )
            else:
                parts.append(
                    f"{entry.label}@{loc.x:.0f},{loc.y:.0f} lane={lane} "
                    f"{phys_speed:.1f}m/s moved={moved:.0f}m"
                )
        except RuntimeError:
            parts.append(f"{entry.label}=gone")
    if parts:
        print("  " + " | ".join(parts), flush=True)


def run_live(
    world,
    bicycle_bps,
    *,
    road_count: int,
    sidewalk_count: int,
    duration_s: float | None = None,
) -> None:
    live, sidewalk_mgr = spawn_live_actors(
        world,
        bicycle_bps,
        road_count=road_count,
        sidewalk_count=sidewalk_count,
    )
    if not live:
        print("No bicycles spawned — run with --probe-only to diagnose.")
        return

    stop_hint = (
        f"auto-stop after {duration_s:.0f}s"
        if duration_s is not None
        else "Press Enter or Ctrl+C to stop"
    )
    print(
        f"\nLive test: {sum(1 for e in live if e.mode == 'road')} road (TM/Driving), "
        f"{sum(1 for e in live if e.mode == 'sidewalk')} sidewalk (navmesh path). "
        f"Bikes follow pedestrian navmesh targets on sidewalk waypoints. "
        f"{stop_hint}.\n",
        flush=True,
    )
    deadline = (time.time() + duration_s) if duration_s is not None else None
    last_status = 0.0
    try:
        while True:
            if deadline is not None and time.time() >= deadline:
                print("\nDuration elapsed.", flush=True)
                break
            if enter_pressed():
                print("\nEnter pressed — stopping.", flush=True)
                break
            try:
                world.wait_for_tick()
            except RuntimeError:
                time.sleep(0.05)
            if sidewalk_mgr is not None:
                try:
                    dt_s = max(0.02, world.get_snapshot().timestamp.delta_seconds)
                except RuntimeError:
                    dt_s = 0.05
                sidewalk_mgr.tick_and_maintain(dt_s)
            now = time.time()
            if now - last_status >= 2.0:
                _print_live_status(world, live, sidewalk_mgr)
                last_status = now
    except KeyboardInterrupt:
        print("\nCtrl+C — stopping.", flush=True)
    finally:
        destroyed = 0
        if sidewalk_mgr is not None:
            destroyed += sidewalk_mgr.destroy_all()
        for entry in live:
            if entry.mode == "sidewalk":
                continue
            try:
                if entry.actor.is_alive and entry.actor.destroy():
                    destroyed += 1
            except RuntimeError:
                pass
        print(f"Destroyed {destroyed} bicycle(s).", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-only", action="store_true", help="run spawn probes and exit")
    ap.add_argument("--skip-probe", action="store_true", help="skip probes; go straight to live test")
    ap.add_argument("--road", type=int, default=0, help="road/TM bicycles (leave 0 for sidewalk-only test)")
    ap.add_argument(
        "--sidewalk",
        type=int,
        default=3,
        help="sidewalk/crosswalk kinematic bicycles",
    )
    ap.add_argument(
        "--draw-paths",
        action="store_true",
        help="draw green/blue debug lines for discovered sidewalk routes in CARLA",
    )
    ap.add_argument(
        "--cross",
        type=int,
        default=0,
        help="deprecated alias; use --sidewalk (old road-crossing mode removed)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional max seconds (default: run until Enter or Ctrl+C)",
    )
    ap.add_argument("--settle", type=float, default=8.0, help="probe settle seconds")
    args = ap.parse_args()

    if args.cross:
        print(
            "Note: --cross (road perpendicular crossing) was removed; using --sidewalk instead.",
            flush=True,
        )
        if args.sidewalk == 2:
            args.sidewalk = args.cross

    _, world = get_world()
    removed = cleanup_leftover_nav_pilots(world)
    if removed:
        print(f"Cleaned up {removed} legacy bicycle nav-pilot walker(s).", flush=True)
    map_name = world.get_map().name.split("/")[-1]
    print(f"Map: {map_name}")
    if "Town10" not in map_name:
        print("Warning: this script targets Town10HD_Opt corridor coordinates.", flush=True)

    _, _, _, bicycle_bps = get_fleet_blueprint_pools(world)
    sidewalk_routes = discover_sidewalk_routes(world)
    if args.draw_paths:
        if not sidewalk_routes:
            print(
                "WARNING: could not sample navmesh points for debug draw "
                "(is pedestrian navigation enabled on this map?).",
                flush=True,
            )
        else:
            _draw_paths(world, sidewalk_routes)
            print(
                f"Drew {len(sidewalk_routes)} sample navmesh route(s) in CARLA (green).",
                flush=True,
            )

    if not args.skip_probe:
        run_probe(world, bicycle_bps, sidewalk_routes, settle_s=args.settle)

    if args.probe_only:
        return 0

    run_live(
        world,
        bicycle_bps,
        road_count=args.road,
        sidewalk_count=args.sidewalk,
        duration_s=args.duration,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
