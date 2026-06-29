#!/usr/bin/env python3
"""Build or extend capture-level train/val/test + k-fold splits for campaign data.

Reads per-capture class counts from ``radar_labeling_qa/summary.json`` (density
per radar per frame × radar frame count) and writes ``config/campaign_split_<seed>.json``.

Classes: car, truck, pedestrian, motorcycle, bicycle.

Usage:
  # Extend an existing split (preserves holdout/kfold assignments; adds motorcycle/bicycle):
  python tools/build_campaign_split.py --extend config/campaign_split_20260608.json

  # Build from manifest CSV (capture_dir column required):
  python tools/build_campaign_split.py --manifest config/campaign_manifest_20260608.csv --seed 20260608
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

DC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ("car", "truck", "pedestrian", "motorcycle", "bicycle")
RARE_CLASSES = ("motorcycle", "bicycle")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(DC_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except OSError:
        return ""


def _zero_counts() -> dict[str, float]:
    return {c: 0.0 for c in DEFAULT_CLASSES}


def _load_summary_block(capture_dir: Path) -> dict:
    """Load the snapshot block from summary.json, tolerating both layouts.

    RadarLabelingTestReport nests the snapshot under ``"summary"``; older files
    had it at the top level. Reading the wrong level silently yields zero counts
    (radar_frames=0), so always resolve via this helper.
    """
    summary_path = capture_dir / "radar_labeling_qa" / "summary.json"
    if not summary_path.is_file():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def counts_from_qa(capture_dir: Path) -> dict[str, float]:
    """Return approximate labeled-point counts per class from QA summary."""
    summary = _load_summary_block(capture_dir)
    density = summary.get("density_per_radar_per_frame", {}) or {}
    radar_frames = float(density.get("radar_frames", 0) or 0)
    by_cls = density.get("by_vehicle_class", {})
    ped_density = float(density.get("pedestrian", 0) or 0)
    counts = _zero_counts()
    for cls in ("car", "truck", "motorcycle", "bicycle"):
        counts[cls] = float(by_cls.get(cls, 0) or 0) * radar_frames
    counts["pedestrian"] = ped_density * radar_frames
    return counts


def actors_from_qa(capture_dir: Path) -> dict[str, int]:
    """Return distinct matched-actor counts per class from QA summary.

    Reads ``unique_actors_by_vehicle_class`` (emitted by RadarLabelingTestReport)
    plus ``unique_pedestrians_matched``. Captures whose summary predates the field
    return zeros, in which case the split's instance floor is skipped (the
    point-share floor still applies).
    """
    summary = _load_summary_block(capture_dir)
    density = summary.get("density_per_radar_per_frame", {}) or {}
    by_cls = density.get("unique_actors_by_vehicle_class", {}) or {}
    out = {c: 0 for c in DEFAULT_CLASSES}
    for cls in ("car", "truck", "motorcycle", "bicycle"):
        out[cls] = int(by_cls.get(cls, 0) or 0)
    out["pedestrian"] = int(summary.get("unique_pedestrians_matched", 0) or 0)
    return out


def _composition(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {c: 0.0 for c in DEFAULT_CLASSES}
    return {c: round(counts[c] / total, 6) for c in DEFAULT_CLASSES}


def _sum_counts(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    out = _zero_counts()
    for c in rows.values():
        for cls in DEFAULT_CLASSES:
            out[cls] += c.get(cls, 0.0)
    return out


def _l1_comp(target: dict[str, float], actual: dict[str, float]) -> float:
    return sum(abs(target.get(c, 0) - actual.get(c, 0)) for c in DEFAULT_CLASSES) / len(DEFAULT_CLASSES)


def _rare_floor_violation(
    splits: dict[str, list[str]],
    comps: dict[str, dict[str, float]],
    global_comp: dict[str, float],
    actors_by_id: dict[str, dict[str, int]],
    *,
    min_share_ratio: float,
    min_instances: int,
) -> float:
    """Shortfall below the rare-class coverage floor, summed over all splits.

    For every split and every rare class, require:
      - point share >= ``min_share_ratio`` * global share, AND
      - >= ``min_instances`` distinct actors (only when actor counts are present).
    Returns 0.0 when satisfied; >0 otherwise. The search adds this (heavily
    weighted) to the objective so a floor-violating split always loses to a
    feasible one. This is a *coverage* guarantee (every split can train AND
    measure the rare class), distinct from training-time class weighting.
    """
    have_actor_data = any(
        actors_by_id.get(i, {}).get(c, 0) > 0
        for ids in splits.values() for i in ids for c in RARE_CLASSES
    )
    violation = 0.0
    for name, ids in splits.items():
        comp = comps[name]
        for c in RARE_CLASSES:
            shortfall = min_share_ratio * global_comp.get(c, 0.0) - comp.get(c, 0.0)
            if shortfall > 0:
                violation += shortfall
            if have_actor_data:
                n_actors = sum(actors_by_id.get(i, {}).get(c, 0) for i in ids)
                if n_actors < min_instances:
                    # Normalize into the same ~[0,1] scale as a share shortfall.
                    violation += (min_instances - n_actors) / max(min_instances, 1)
    return violation


def _search_holdout(
    capture_ids: list[str],
    counts_by_id: dict[str, dict[str, float]],
    *,
    sizes: tuple[int, int, int],
    alpha: float,
    seed: int,
    actors_by_id: dict[str, dict[str, int]] | None = None,
    min_share_ratio: float = 0.5,
    min_instances: int = 3,
) -> dict:
    """Greedy random search for train/val/test with balanced composition and a
    rare-class coverage floor (every split keeps enough motorcycle/bicycle)."""
    n = len(capture_ids)
    train_n, val_n, test_n = sizes
    assert train_n + val_n + test_n == n
    actors_by_id = actors_by_id or {}
    global_comp = _composition(_sum_counts(counts_by_id))
    rng = random.Random(seed)
    best = None
    floor_weight = 1000.0   # >> any L_comp/L_size term, so feasibility wins
    for _ in range(5000):
        perm = capture_ids[:]
        rng.shuffle(perm)
        train = sorted(perm[:train_n], key=int)
        val = sorted(perm[train_n : train_n + val_n], key=int)
        test = sorted(perm[train_n + val_n :], key=int)
        comps = {
            "train": _composition(_sum_counts({i: counts_by_id[i] for i in train})),
            "val": _composition(_sum_counts({i: counts_by_id[i] for i in val})),
            "test": _composition(_sum_counts({i: counts_by_id[i] for i in test})),
        }
        l_comp = (
            _l1_comp(global_comp, comps["train"])
            + _l1_comp(global_comp, comps["val"])
            + _l1_comp(global_comp, comps["test"])
        ) / 3.0
        l_size = abs(len(train) - train_n) + abs(len(val) - val_n) + abs(len(test) - test_n)
        violation = _rare_floor_violation(
            {"train": train, "val": val, "test": test}, comps, global_comp, actors_by_id,
            min_share_ratio=min_share_ratio, min_instances=min_instances,
        )
        obj = alpha * l_comp + (1 - alpha) * l_size + floor_weight * violation
        if best is None or obj < best["objective"]:
            best = {
                "train": train,
                "val": val,
                "test": test,
                "composition": comps,
                "L_comp": round(l_comp, 6),
                "L_size": round(l_size, 6),
                "rare_floor_violation": round(violation, 6),
                "rare_floor_ok": violation == 0.0,
                "objective": round(obj, 6),
            }
    return best


def _kfold_assign(capture_ids: list[str], k: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    perm = capture_ids[:]
    rng.shuffle(perm)
    folds: dict[str, int] = {}
    for i, cid in enumerate(perm):
        folds[cid] = i % k
    return folds


def extend_split(path: Path, out_path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["classes"] = list(DEFAULT_CLASSES)
    captures = data.get("captures", {})
    for _run, cap in captures.items():
        counts = cap.get("counts", {})
        for cls in DEFAULT_CLASSES:
            counts.setdefault(cls, 0.0)
        cap["counts"] = counts

    all_counts = {k: v.get("counts", _zero_counts()) for k, v in captures.items()}
    global_comp = _composition(_sum_counts(all_counts))
    data["global_composition"] = global_comp

    holdout = data.get("holdout", {})
    if holdout:
        holdout["global_composition"] = global_comp
        for split in ("train", "val", "test"):
            ids = [str(i) for i in holdout.get(split, [])]
            if ids:
                holdout["composition"][split] = _composition(
                    _sum_counts({i: all_counts[i] for i in ids if i in all_counts})
                )
        data["holdout"] = holdout

    if "kfold" in data and data["kfold"].get("fold_composition"):
        kf = data["kfold"]
        fold_comp = {}
        folds = kf.get("folds", {})
        by_fold: dict[int, list[str]] = {}
        for cid, f in folds.items():
            by_fold.setdefault(int(f), []).append(str(cid))
        for f, ids in sorted(by_fold.items()):
            fold_comp[str(f)] = _composition(
                _sum_counts({i: all_counts[i] for i in ids if i in all_counts})
            )
        kf["fold_composition"] = fold_comp
        data["kfold"] = kf

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Extended split -> {out_path}")


def build_from_manifest(
    manifest: Path,
    seed: int,
    out_path: Path,
    *,
    min_share_ratio: float = 0.5,
    min_instances: int = 3,
    sizes: tuple[int, int, int] | None = None,
) -> None:
    import csv

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    captures: dict[str, dict] = {}
    counts_by_id: dict[str, dict[str, float]] = {}
    actors_by_id: dict[str, dict[str, int]] = {}
    dropped: dict[str, str] = {}
    for i, row in enumerate(rows, start=1):
        cap_dir = (row.get("capture_dir") or "").strip()
        if not cap_dir or cap_dir == "FAILED":
            dropped[str(i)] = cap_dir or "FAILED"
            continue
        p = Path(cap_dir)
        counts = counts_from_qa(p)
        actors = actors_from_qa(p)
        rid = str(i)
        counts_by_id[rid] = counts
        actors_by_id[rid] = actors
        captures[rid] = {
            "capture_dir": cap_dir,
            "dir_name": p.name,
            "seed": row.get("DATASET_SEED", ""),
            "minutes": float(row.get("minutes", 0) or 0),
            "counts": counts,
            "actor_counts": actors,
            "source": "qa_summary",
        }

    capture_ids = sorted(counts_by_id.keys(), key=int)
    if len(capture_ids) < 3:
        raise SystemExit(f"Need >= 3 good captures; got {len(capture_ids)}")

    # Explicit --sizes wins (frozen benchmark splits pin exact counts); else the
    # default ~64/18/18 formula (7/2/2 for 11 captures; drop bad runs first).
    n = len(capture_ids)
    if sizes is not None:
        train_n, val_n, test_n = sizes
        if train_n + val_n + test_n != n:
            raise SystemExit(
                f"--sizes {train_n}/{val_n}/{test_n} sums to {train_n + val_n + test_n}, "
                f"but there are {n} good captures (after drops). Adjust to match."
            )
        if min(train_n, val_n, test_n) < 1:
            raise SystemExit("--sizes must give every partition >= 1 capture")
    else:
        train_n = max(1, int(math.floor(n * 0.64)))
        val_n = max(1, int(math.floor(n * 0.18)))
        test_n = n - train_n - val_n
        if test_n < 1:
            test_n = 1
            train_n = n - val_n - test_n

    holdout = _search_holdout(
        capture_ids,
        counts_by_id,
        sizes=(train_n, val_n, test_n),
        alpha=0.5,
        seed=0,
        actors_by_id=actors_by_id,
        min_share_ratio=min_share_ratio,
        min_instances=min_instances,
    )
    if not holdout.get("rare_floor_ok", True):
        print(
            "WARNING: no split satisfied the rare-class coverage floor "
            f"(min_share_ratio={min_share_ratio}, min_instances={min_instances}; "
            f"residual violation={holdout.get('rare_floor_violation')}). "
            "A split is short on motorcycle/bicycle — add captures or raise the "
            "two-wheeler fractions in run_campaign.py CONT_AXES.",
            flush=True,
        )
    k = 5
    folds = _kfold_assign(capture_ids, k, seed=0)

    out = {
        "created_for": manifest.name,
        "git_commit": _git_commit(),
        "classes": list(DEFAULT_CLASSES),
        "params": {
            "sizes": [train_n, val_n, test_n],
            "k": k,
            "seed": 0,
            "alpha_size": 0.5,
            "rare_classes": list(RARE_CLASSES),
            "rare_min_share_ratio": min_share_ratio,
            "rare_min_instances": min_instances,
            "rare_floor_ok": holdout.get("rare_floor_ok", True),
            "dropped_runs": [int(x) for x in dropped.keys()],
            "split_level": "capture (group); never split a capture",
        },
        "dropped": dropped,
        "global_composition": _composition(_sum_counts(counts_by_id)),
        "captures": captures,
        "holdout": {
            "sizes": [train_n, val_n, test_n],
            **holdout,
            "global_composition": _composition(_sum_counts(counts_by_id)),
        },
        "kfold": {
            "k": k,
            "seed": 0,
            "folds": folds,
            "fold_sizes": {str(f): sum(1 for v in folds.values() if v == f) for f in range(k)},
            "fold_composition": {
                str(f): _composition(
                    _sum_counts(
                        {cid: counts_by_id[cid] for cid, fv in folds.items() if fv == f}
                    )
                )
                for f in range(k)
            },
            "rotation": (
                "for fold f in 0..k-1:  test=f,  val=(f+1)%k,  train=the rest. "
                "Report mean+-std over the k folds."
            ),
        },
        "notes": [
            "Capture-level split: leakage-free because temporally-correlated frames "
            "of a capture never straddle splits.",
            "Classes include motorcycle and bicycle (may be zero on pre-2-wheeler captures).",
        ],
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Built split ({n} captures) -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extend", type=Path, help="extend an existing split JSON in place")
    ap.add_argument("--manifest", type=Path, help="campaign manifest CSV with capture_dir column")
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument(
        "--min-rare-share-ratio",
        type=float,
        default=0.5,
        help="each split must hold >= this fraction of the global motorcycle/bicycle "
             "point share (default 0.5)",
    )
    ap.add_argument(
        "--min-rare-instances",
        type=int,
        default=3,
        help="each split must hold >= this many distinct motorcycle/bicycle actors "
             "when QA actor counts are available (default 3)",
    )
    ap.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="explicit train,val,test capture counts (e.g. 7,3,4); must sum to the "
             "number of good captures after drops. Overrides the default 64/18/18 formula.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: config/campaign_split_<seed>.json)",
    )
    args = ap.parse_args()

    sizes = None
    if args.sizes:
        try:
            parts = tuple(int(x) for x in args.sizes.split(","))
        except ValueError:
            ap.error("--sizes must be three comma-separated integers, e.g. 7,3,4")
        if len(parts) != 3:
            ap.error("--sizes must have exactly three values: train,val,test")
        sizes = parts
    out = args.out or (DC_ROOT / "config" / f"campaign_split_{args.seed}.json")

    if args.extend:
        extend_split(args.extend, out)
        return 0
    if args.manifest:
        build_from_manifest(
            args.manifest,
            args.seed,
            out,
            min_share_ratio=args.min_rare_share_ratio,
            min_instances=args.min_rare_instances,
            sizes=sizes,
        )
        return 0
    ap.error("pass --extend or --manifest")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
