"""Post-process a labeled radar capture to align with RadarScenes-style features.

Two transforms, applied in-place to radar_data_labeled.csv:

1. Pedestrian Doppler fix.
   CARLA's controller.ai.walker moves walkers kinematically, so walker.get_velocity()
   returns 0 and every pedestrian radar return has velocity_mps == 0 even when the
   pedestrian is clearly moving. Fix: compute bulk velocity by central difference
   on actor_frames.jsonl positions, project onto sensor->actor line of sight,
   optionally add Gaussian micro-Doppler noise. Vehicle rows are untouched.

2. RCS unit calibration.
   `rcs_proxy_m2` is a geometric OBB-silhouette area in m^2 — not a real
   electromagnetic cross-section. RadarScenes (and most real-world radar datasets)
   report RCS in dBsm. We add an `rcs_dBsm` column computed as
       rcs_dBsm = 10*log10(rcs_proxy_m2) + offset[class]
   where offset[class] is calibrated so the per-class median matches RadarScenes-
   like target medians (pedestrian -11, car 7, truck 15 dBsm).

Run:
    python -m capture.PostProcessDataset --capture-dir <path>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc)
_dc.bootstrap(__file__)

LABELED_CSV = "radar_data_labeled.csv"
ACTOR_FRAMES_JSONL = "actor_frames.jsonl"

# RadarScenes-style class-conditional median dBsm targets.
# Sources: Schumann et al. (RadarScenes), Yamada 79 GHz pedestrian RCS (median -11.1 dBsm),
# automotive RCS surveys (car ~10 dBsm, large vehicle ~20 dBsm).
DEFAULT_TARGET_MEDIAN_DBSM = {
    "pedestrian": -11.0,
    "car": 7.0,
    "truck": 15.0,
    "bus": 12.0,
    "bicycle": -5.0,
    "motorcycle": 0.0,
}
# Two-scale dB-domain noise: (per-actor offset, per-actor-per-frame jitter).
# Per-actor offset captures inter-individual RCS variation (different car models,
# different pedestrians); persistent across frames so the GNN sees consistent
# actor "signatures". Per-frame jitter captures aspect-angle dependence (a car
# broadside has very different RCS than its rear). Total σ ≈ sqrt(a^2 + f^2).
# Targets:  pedestrian σ_total ≈ 4 dB, car ≈ 3.5 dB, truck ≈ 3 dB
# matching the per-class spread visible in RadarScenes.
DEFAULT_RCS_DB_NOISE = {
    # class:        (σ_actor_dB, σ_frame_dB)
    "pedestrian":   (3.0, 4.0),    # sqrt(9 + 16) = 5.0 dB
    "car":          (2.5, 3.5),    # sqrt(6.25 + 12.25) ≈ 4.3 dB
    "truck":        (2.0, 3.0),    # sqrt(4 + 9) ≈ 3.6 dB
    "bus":          (2.0, 3.0),
    "bicycle":      (2.5, 3.0),
    "motorcycle":   (2.0, 2.5),
}
DEFAULT_MICRO_DOPPLER_SIGMA = 1.0
# Central-difference stride (in world ticks). At ~70 Hz tick rate, stride=10 spans
# ~140 ms — long enough to dilute CARLA's occasional navmesh batch-updates (the
# controller sometimes accumulates several ticks of motion into a single position
# jump, which a short stride would read as a spurious >4 m/s spike), but short
# enough that walker direction changes (~1 s timescale) are still tracked.
DEFAULT_FD_STRIDE = 10
# Hard ceiling on computed bulk walker speed (m/s). CARLA walker.recommended_values
# tops out around 2-3 m/s; anything above is a navmesh artifact, not a real walker
# velocity. We clamp the bulk-speed magnitude (preserving direction) and report a
# count so glitches don't go silent.
DEFAULT_MAX_WALKER_SPEED_MPS = 3.0


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def load_walker_positions(frames_path):
    """Return {(frame_id, actor_id): (x, y, z)} for pedestrians only."""
    walker_pos = {}
    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = int(rec["frame"])
            for a in rec.get("actors", []):
                if a.get("kind") != "pedestrian":
                    continue
                loc = a["location"]
                walker_pos[(frame, int(a["id"]))] = (
                    float(loc["x"]),
                    float(loc["y"]),
                    float(loc["z"]),
                )
    return walker_pos


def estimate_frame_dt(labeled_path):
    """Mean simulation tick interval (seconds), estimated from CSV (frame, timestamp) pairs."""
    seen = {}
    with labeled_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = int(row["frame"])
            if fr not in seen:
                seen[fr] = float(row["timestamp"])
    if len(seen) < 2:
        return 1.0 / 30.0
    frames = sorted(seen.keys())
    f0, fN = frames[0], frames[-1]
    return (seen[fN] - seen[f0]) / (fN - f0)


def class_offsets_from_data(labeled_path, target_median_dbsm):
    """Median(rcs_proxy_m2) -> dBsm shift so post-shift median matches target."""
    rcs_by_class = defaultdict(list)
    with labeled_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = row["rcs_proxy_m2"].strip()
            klass = row["matched_actor_class"].strip()
            if not r or not klass:
                continue
            try:
                v = float(r)
            except ValueError:
                continue
            if v > 0:
                rcs_by_class[klass].append(v)
    offsets = {}
    info = {}
    for klass, xs in rcs_by_class.items():
        med = median(xs)
        if med is None or med <= 0:
            continue
        cur_dbsm = 10.0 * math.log10(med)
        tgt = target_median_dbsm.get(klass)
        if tgt is None:
            continue
        offsets[klass] = tgt - cur_dbsm
        info[klass] = (med, cur_dbsm, tgt, offsets[klass], len(xs))
    return offsets, info


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument(
        "--micro-doppler-sigma",
        type=float,
        default=DEFAULT_MICRO_DOPPLER_SIGMA,
        help="Gaussian stddev (m/s) added to bulk pedestrian Doppler. "
             "0 disables noise (option 1 only).",
    )
    p.add_argument(
        "--fd-stride",
        type=int,
        default=DEFAULT_FD_STRIDE,
        help="Central-difference stride in world ticks for walker velocity.",
    )
    p.add_argument(
        "--max-walker-speed",
        type=float,
        default=DEFAULT_MAX_WALKER_SPEED_MPS,
        help="Hard ceiling on bulk walker speed (m/s). Above this the value is "
             "treated as a navmesh batch-update artifact and clamped.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rcs-db-noise",
        action="store_true",
        default=True,
        help="Add two-scale dB-domain noise (per-actor offset + per-frame jitter) "
             "to rcs_dBsm so class distributions overlap realistically.",
    )
    p.add_argument(
        "--no-rcs-db-noise",
        dest="rcs_db_noise",
        action="store_false",
        help="Disable RCS dB noise (median-only calibration).",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (defaults to overwriting the input).")
    args = p.parse_args()

    capture_dir = args.capture_dir
    labeled = capture_dir / LABELED_CSV
    frames_path = capture_dir / ACTOR_FRAMES_JSONL
    out = args.out or labeled

    if not labeled.is_file():
        sys.exit(f"Missing {labeled}")
    if not frames_path.is_file():
        sys.exit(f"Missing {frames_path}")

    print(f"[1/4] Loading walker positions from {frames_path.name} ...", flush=True)
    walker_pos = load_walker_positions(frames_path)
    walker_actors = {aid for (_, aid) in walker_pos.keys()}
    print(f"      {len(walker_pos):,} (frame, walker) entries; "
          f"{len(walker_actors)} distinct walkers", flush=True)

    print(f"[2/4] Estimating mean Δt from {labeled.name} ...", flush=True)
    dt_mean = estimate_frame_dt(labeled)
    print(f"      Δt ≈ {dt_mean * 1000:.2f} ms  (~{1.0/max(dt_mean,1e-9):.0f} Hz)", flush=True)

    print(f"[3/4] Calibrating RCS dBsm offsets ...", flush=True)
    offsets, info = class_offsets_from_data(labeled, DEFAULT_TARGET_MEDIAN_DBSM)
    if not offsets:
        sys.exit("No matched rows with rcs_proxy_m2 — nothing to calibrate.")
    for klass, (med_m2, cur_db, tgt_db, off, n) in info.items():
        print(f"      {klass:<12} n={n:>7,}  median(m²)={med_m2:.4f}  "
              f"current_dBsm={cur_db:+.2f}  target_dBsm={tgt_db:+.2f}  "
              f"offset={off:+.2f} dB", flush=True)

    print(f"[4/4] Rewriting {out.name} (in-place rewrite is atomic via .tmp swap) ...",
          flush=True)
    if args.rcs_db_noise:
        print(f"      RCS dB noise: per-actor offset + per-frame jitter "
              f"(targets pedestrian σ≈4, car σ≈3.2, truck σ≈3 dB)", flush=True)
    rng = random.Random(args.seed)
    fd = args.fd_stride

    # Deterministic per-(class, actor) and per-(class, actor, frame) offsets.
    # Cached because the same actor and (actor, frame) appear in many rows;
    # caching makes within-frame returns share an offset so they stay correlated.
    actor_offset_cache: dict[tuple[str, int], float] = {}
    frame_offset_cache: dict[tuple[str, int, int], float] = {}

    def rcs_db_noise(klass: str, actor_id: int, frame_id: int) -> float:
        sigmas = DEFAULT_RCS_DB_NOISE.get(klass)
        if sigmas is None:
            return 0.0
        sa, sf = sigmas
        key_a = (klass, actor_id)
        ao = actor_offset_cache.get(key_a)
        if ao is None:
            seed_a = f"{args.seed}|{klass}|{actor_id}|actor"
            ao = random.Random(seed_a).gauss(0.0, sa)
            actor_offset_cache[key_a] = ao
        key_f = (klass, actor_id, frame_id)
        fo = frame_offset_cache.get(key_f)
        if fo is None:
            seed_f = f"{args.seed}|{klass}|{actor_id}|{frame_id}"
            fo = random.Random(seed_f).gauss(0.0, sf)
            frame_offset_cache[key_f] = fo
        return ao + fo

    stats_walker_fixed = 0
    stats_walker_skipped = 0
    stats_walker_clamped = 0
    stats_rcs_dbsm_written = 0
    max_walker_speed = max(0.1, float(args.max_walker_speed))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with labeled.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        in_fields = list(reader.fieldnames or [])
        out_fields = in_fields + (["rcs_dBsm"] if "rcs_dBsm" not in in_fields else [])
        with tmp.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()
            for row in reader:
                # -- velocity fix for pedestrians --
                aid_raw = row["matched_actor_id"].strip()
                kind = row["matched_actor_kind"].strip()
                if aid_raw and kind == "pedestrian":
                    try:
                        aid = int(aid_raw)
                    except ValueError:
                        aid = None
                    if aid is not None:
                        f0 = int(row["frame"])
                        p_prev = walker_pos.get((f0 - fd, aid))
                        p_next = walker_pos.get((f0 + fd, aid))
                        p_now = walker_pos.get((f0, aid))
                        if p_prev is not None and p_next is not None and p_now is not None:
                            dt = 2.0 * fd * dt_mean
                            vx = (p_next[0] - p_prev[0]) / dt
                            vy = (p_next[1] - p_prev[1]) / dt
                            vz = (p_next[2] - p_prev[2]) / dt
                            # Clamp bulk-speed magnitude (preserve direction) so
                            # navmesh batch-update glitches don't leak into Doppler.
                            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
                            if speed > max_walker_speed:
                                scale = max_walker_speed / speed
                                vx *= scale; vy *= scale; vz *= scale
                                stats_walker_clamped += 1
                            sx = float(row["sensor_world_x_m"])
                            sy = float(row["sensor_world_y_m"])
                            sz = float(row["sensor_world_z_m"])
                            ux = p_now[0] - sx
                            uy = p_now[1] - sy
                            uz = p_now[2] - sz
                            n = math.sqrt(ux * ux + uy * uy + uz * uz)
                            if n > 1e-6:
                                ux /= n; uy /= n; uz /= n
                                v_rad = vx * ux + vy * uy + vz * uz
                                if args.micro_doppler_sigma > 0:
                                    v_rad += rng.gauss(0.0, args.micro_doppler_sigma)
                                row["velocity_mps"] = f"{v_rad:.6f}"
                                stats_walker_fixed += 1
                            else:
                                stats_walker_skipped += 1
                        else:
                            stats_walker_skipped += 1

                # -- RCS dBsm --
                r = row["rcs_proxy_m2"].strip()
                klass = row["matched_actor_class"].strip()
                if r and klass in offsets:
                    try:
                        v = float(r)
                        if v > 0:
                            dbsm = 10.0 * math.log10(v) + offsets[klass]
                            if args.rcs_db_noise and aid_raw:
                                try:
                                    dbsm += rcs_db_noise(
                                        klass, int(aid_raw), int(row["frame"])
                                    )
                                except ValueError:
                                    pass
                            row["rcs_dBsm"] = f"{dbsm:.4f}"
                            stats_rcs_dbsm_written += 1
                        else:
                            row["rcs_dBsm"] = ""
                    except ValueError:
                        row["rcs_dBsm"] = ""
                else:
                    row["rcs_dBsm"] = ""

                writer.writerow(row)
    tmp.replace(out)
    print(flush=True)
    print(f"  Walker velocities rewritten: {stats_walker_fixed:,}", flush=True)
    print(f"  Walker rows skipped (missing neighbour frame): {stats_walker_skipped:,}",
          flush=True)
    if stats_walker_clamped:
        pct = 100.0 * stats_walker_clamped / max(stats_walker_fixed, 1)
        print(
            f"  Walker speeds clamped to {max_walker_speed:.1f} m/s "
            f"(navmesh artifacts): {stats_walker_clamped:,} ({pct:.2f}%)",
            flush=True,
        )
    else:
        print(f"  No navmesh batch-update artifacts detected "
              f"(no clamps above {max_walker_speed:.1f} m/s).", flush=True)
    print(f"  rcs_dBsm cells written: {stats_rcs_dbsm_written:,}", flush=True)
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
