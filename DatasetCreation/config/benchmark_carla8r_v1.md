# CARLA-8R Radar Semantic Segmentation Benchmark — v1 (2026-06-27)

Canonical, frozen split for the 8-radar CARLA corridor dataset. Public release:
session IDs and partitions below are **frozen and citable**. Do not regenerate;
cite this file + the pinned commit.

## Provenance (pin all of these)

| Item | Value |
|---|---|
| Generator | CARLA, map `Town10HD_Opt` (single map, all sessions) |
| Campaign | `tools/run_campaign.py`, base seed **20260627**, Latin-Hypercube diversity axes |
| Code commit | **92726fe** (capture + `PostProcessDataset.py`) |
| Rig | 8 radars, 4 paired gantry stations along a ~94 m corridor; z=3 m, pitch 8° |
| Radar config | HFOV 120°, VFOV 40°, range 35 m, 15000 pts/s, 20 Hz/sensor, **async** (not frame-locked) |
| Post-process | per-capture dynamic class-median RCS calibration + Swerling-1 + per-actor Gaussian + specular spikes + aspect; static RCS from RadarScenes inverse-CDF (per-voxel). **No cross-session statistics.** |
| Canonical model input | `radar_data_labeled.carla_noisy1_vis1_v2.npz` — baked `*_noisy` geometry + `rcs_dBsm`, `visible==1` filter, deterministic (no load-time RNG) |
| Classes (head) | RadarScenes 6-class `ClassificationLabel`: car / pedestrian / pedestrian_group / **two_wheeler** / large_vehicle / static. CARLA→head: car,van→car; truck,bus,trailer→large_vehicle; **bicycle+motorcycle→two_wheeler**; pedestrian→pedestrian. Present in CARLA: car, pedestrian, two_wheeler, large_vehicle (pedestrian_group empty). Raw `matched_actor_class` (incl. separate bicycle/motorcycle) is preserved in the CSV for diagnostics. |
| Label | point-wise semantic label per radar detection (not BEV grid); instance ids present (`matched_actor_id`) |

## Sessions (immutable IDs = capture dir name)

Split **unit = session** (capture); a session is never split across partitions
(leakage-free: temporally-correlated 20 Hz frames stay together).

| run | session id | seed | partition |
|---|---|---|---|
| 1 | sensor_capture_20260627_053201 | 20261628 | **test** |
| 2 | sensor_capture_20260627_054907 | 20261629 | train |
| 3 | sensor_capture_20260627_060634 | 20261630 | **test** |
| 4 | sensor_capture_20260627_062520 | 20261631 | val |
| 5 | sensor_capture_20260627_064316 | 20261632 | val |
| 6 | sensor_capture_20260627_070123 | 20261633 | train |
| 7 | sensor_capture_20260627_072419 | 20261634 | **DROPPED** (data-quality: stuck vehicle = 90% of matches) |
| 8 | sensor_capture_20260627_074536 | 20261635 | train |
| 9 | sensor_capture_20260627_080252 | 20261636 | train |
| 10 | sensor_capture_20260627_082109 | 20261637 | train |
| 11 | sensor_capture_20260627_084356 | 20261638 | train |
| 12 | sensor_capture_20260627_090022 | 20261639 | train |
| 13 | sensor_capture_20260627_092218 | 20261640 | **test** |
| 14 | sensor_capture_20260627_093924 | 20261641 | **test** |
| 15 | sensor_capture_20260627_095751 | 20261642 | val |

**Canonical holdout (single):** train = 7 sessions {2,6,8,9,10,11,12},
val = 3 {4,5,15}, test = 4 {1,3,13,14}. Machine-readable:
`config/campaign_split_20260627.json` (`holdout` block). Rare-class floor satisfied
in every partition (≥0.5× global share AND ≥3 distinct moto/bike actors:
train 47/69, val 17/27, test 21/42).

## Protocol

- **Eval:** single holdout above. Train on `train`, early-stop / select on `val`,
  report once on `test`.
- **Primary metric:** foreground **mIoU** over the moving `ClassificationLabel`s present
  in CARLA — **car, pedestrian, two_wheeler, large_vehicle** (pedestrian_group is empty
  here; exclude) — plus per-class IoU. Bicycle and motorcycle are a single `two_wheeler`
  class at the head (RadarScenes convention); do **not** split them. Optionally report a
  **motorcycle-subset diagnostic** (IoU/recall on raw moto-labeled points) to confirm the
  two_wheeler class isn't bicycle-only — that is what the campaign's moto enrichment buys.
  **Static/background is excluded from the headline** — the CARLA map is identical
  across all sessions, so static background is shared across train/val/test and its
  IoU is not a generalization signal.
- **Uncertainty (required):** report bootstrap CI over test frames. **Caveat to state
  explicitly:** the test set is only **4 independent scenes**, so frame-bootstrap CIs
  *understate* true session-level variance. For an honest session-level interval,
  publish the companion **5-fold GroupKFold** (already defined in the JSON `kfold`
  block, sizes 3/3/3/3/2) as mean±std — this is the recommended robustness number even
  though the canonical headline is the single holdout.

## Scope / known limits (disclose in the benchmark card)

- **Same-site only.** One map, fixed rig, **no weather/lighting/time-of-day variation**.
  Supports a same-site IID claim; cross-site/cross-condition generalization is out of
  scope by construction. No condition-held-out slice exists.
- **Sessions are exchangeable** (LHS-sampled, captured back-to-back) → random GroupKFold
  is valid; forward-chaining is not applicable.
- **Two-wheeler is one class; motorcycle is a sparse *subtype*, not a head class**
  (25k pts / 85 instances over 14 sessions). The split's per-subtype floor (≥3 bicycle
  AND ≥3 motorcycle actors/partition) guarantees every fold's `two_wheeler` spans both
  slow-bicycle and fast-motorcycle Doppler — so no fold learns "two-wheeler = bicycle".
  The moto-subset diagnostic is low-power; report it as a check, not a headline number.
- Optional stress slice: hold out high-traffic-density sessions (only constructible
  covariate axis beyond class).

## Class-balance across splits (head classes: car / pedestrian / two_wheeler / large_vehicle)

Foreground point fractions — train .39/.34/.103/.17, val .38/.34/.069/.21,
test .38/.37/.081/.17 — are NOT equalized, and intentionally so:
- The headline metric is **mean per-class IoU** (each class weighted equally,
  independent of point share), so unequal fractions do not bias mIoU; they only
  widen a class's IoU confidence interval.
- The remaining imbalance is **structural**, not a bad draw: the best possible
  head-class-balanced 7/3/4 (subtype floor kept) still puts val two_wheeler at ~.074
  and val large_veh at ~.20 — capture-level grouping + LHS-lumpy sessions (a few
  truck-heavy: runs 2/9/14; a few bike-heavy: run 10) leave no assignment that
  equalizes them. Re-shuffling the frozen split buys ~0.005 — not worth it.
- Every head class is well-sampled in every split (smallest: val two_wheeler
  20k pts / 44 instances). Report **per-class IoU + bootstrap CIs**; the thinner
  spots surface honestly as wider CIs. Caveat in the results table: test
  large_vehicle rests on ~10 truck instances; the motorcycle-subtype diagnostic
  on val has 17 moto instances.

## Implementation note — empty `pedestrian_group`

CARLA emits zero `pedestrian_group` points, but the head stays 6-class (RadarScenes
`ClassificationLabel`) so the class id space is unchanged. Absent classes are handled
by **present-masking** (a class with zero ground-truth support is excluded from mIoU,
from class-weighting, and from per-class reports — marked N/A / nan). Verified across
the code paths:
- `train_radar_semseg_msg_attn.py` (fusion): weights + val/selection mIoU present-masked.
- `train_radar_semseg_msg.py` (no-fusion baseline): same, fixed this round (previously
  the empty class got a ~1000 inv-freq weight that distorted every real class's weight,
  and was counted as 0 IoU).
- `eval_carla.py`, `fusion_diag.py`, `percapture_eval.py`: present-masked headline mIoU.
- `eval_sensor_noise.py`, `eval_sensor_dropout.py`: `iou_from_confusion` returns nan for
  zero-GT classes; all mIoU aggregates use `nanmean` (fixed this round).
When every class is present (RadarScenes) present-masking is a no-op — identical numbers.
