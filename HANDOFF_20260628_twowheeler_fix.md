# Handoff — CARLA-8R benchmark + two-wheeler micro-Doppler fix (2026-06-28)

## TL;DR
- Built a frozen public benchmark (**CARLA-8R v1**) from the 14-session 20260627 capture set.
- Diagnosed and **fixed** the long-standing "two-wheelers are unlearnable" problem: it was a
  radar-**synthesis velocity-signature gap** (synthetic bicycles were dim + zero-Doppler → looked
  like static clutter). Extended the pedestrian micro-Doppler treatment to two-wheelers,
  **calibrated to the real RadarScenes two-wheeler Doppler distribution**.
- **Result (held-out test, early fusion): two_wheeler IoU 0.052 → 0.412 (~8×); foreground mIoU 0.368 → 0.448.**

---

## 1. Dataset location

**New 15-capture campaign (seed 20260627):** `/scratch/tamoghnd/dataset_captures/sensor_capture_20260627_*`
- 15 dirs; **run 7 (`_072419`) is DROPPED** (stuck-car) → **14 used**.
- Each capture dir holds: `radar_data_labeled.csv` (post-processed, **now carries the two-wheeler fix**),
  `actor_frames.jsonl` (actor world positions — source of the bulk velocity),
  `run_meta.json` (has `postprocess_seed`), `radar_data_labeled.carla_noisy1_vis1_v2.npz` (feature cache),
  `cleaning.json`.
- **Backups of the pre-fix data:** `radar_data_labeled.csv.pretwfix` and
  `radar_data_labeled.carla_noisy1_vis1_v2.npz.pretwfix` in every capture dir. Full rollback =
  restore the `.pretwfix` files.

**Config (under `CarlaDatasetCreation/`):**
- Manifest: `DatasetCreation/config/campaign_manifest_20260627.csv`
- Split (7/3/4, capture-level, frozen): `DatasetCreation/config/campaign_split_20260627.json`
- Benchmark card: `config/benchmark_carla8r_v1.md`

**Split (run → capture):**
| | runs | captures |
|---|---|---|
| **train (7)** | 2,6,8,9,10,11,12 | 054907, 070123, 074536, 080252, 082109, 084356, 090022 |
| **val (3)** | 4,5,15 | 062520, 064316, 095751 |
| **test (4)** | 1,3,13,14 | 053201, 060634, 092218, 093924 |

Metric: **foreground mIoU** (present-class mean, **STATIC excluded** — same map everywhere).
PEDESTRIAN_GROUP is absent in CARLA → present-masked (excluded from the headline). Rig = 8-radar
infrastructure, static (no ego motion). RadarScenes comparison: its split is 102/28/28 (65/18/18);
our lower train fraction is a consequence of having 14 sessions vs 158, not a methodology flaw.
For more robust eval, the 5-fold CV is defined in the split JSON.

---

## 2. Latest results — early fusion, held-out test (BEFORE vs AFTER the fix)

| metric | **BEFORE** (old features) | **AFTER** (micro-Doppler) | Δ |
|---|---|---|---|
| **TWO_WHEELER IoU** | **0.052** | **0.412** | **+0.36 (~8×)** |
| foreground mIoU | 0.368 | **0.448** | +0.080 |
| present-class mIoU | 0.491 | 0.556 | +0.065 |
| CAR | 0.228 | 0.251 | +0.023 |
| PEDESTRIAN | 0.791 | 0.741 | −0.050 |
| LARGE_VEHICLE | 0.403 | 0.389 | −0.014 |
| STATIC | 0.981 | 0.988 | +0.007 |

- The two_wheeler lift drives the +0.08 foreground gain. Same recipe, same split — **only the
  two-wheeler Doppler changed**, so it's fully attributable to the fix.
- PED −0.05 is the expected, acceptable cost (bicycles now move at ped speed → mild
  bicycle↔ped confusion). Net win.
- BN gap benign both runs (eval-mode > train-mode → headline not deflated).
- **CAR stays ~0.25 — a SEPARATE issue** (static-rig cars are mostly stopped/tangential →
  geometry-only weak Doppler; see memory `car-doppler-degeneracy-20260627`). Not addressed by the
  two-wheeler fix.

**Checkpoints:**
- BEFORE: `PointNet/Pnet_pytorch/log/sem_seg/c20260627_early/checkpoints/best_model.pth`
- AFTER:  `PointNet/Pnet_pytorch/log/sem_seg/c20260627_early_twfix/checkpoints/best_model.pth`

---

## 3. The two-wheeler fix (what changed)

**File:** `CarlaDatasetCreation/DatasetCreation/capture/PostProcessDataset.py` (+165/−4, **only file changed, UNCOMMITTED**).
- New `[A2]` block: two-wheelers get a **real bulk radial velocity** (central-differenced from
  `actor_frames.jsonl` world positions, like the pedestrian fix) **+ per-class micro-Doppler spread**.
- `DEFAULT_MICRO_DOPPLER_SIGMA_BY_CLASS = {pedestrian 1.0 (unchanged), bicycle 3.8, motorcycle 4.3}`.
- **Calibrated to RadarScenes:** merged two_wheeler signed-vr std 2.84 → **4.06** (RadarScenes 3.7–4.6),
  kept **median ~0** (RadarScenes' +1..+2.8 is an ego-compensation approach-bias; our rig is static →
  match the SPREAD, not the offset). RCS left as-is (already ~−12 vs RadarScenes −11).
- Pedestrian + all FMCW columns **byte-identical** (key-seeded RNG isolates the shared sequence),
  deterministic, `py_compile` OK.
- The fix lives in **`vr`** and the **derived `vx`/`vy`** (`vx=vr·cos(world_az)`, `vy=vr·sin`), so it
  serves **every encoder**: PointNet `[x,y,vr,rcs]`, RadarGNN-3d `[rcs,vr,degree]`, RadarGNN-5d
  `[rcs,vx,vy,time,degree]`.

**Regeneration script:** `CarlaDatasetCreation/regenerate_twowheeler_20260627.sh`
- Re-ran PostProcessDataset on all 14 captures (each with its `run_meta.postprocess_seed`), 14/14 ok.
- Backs up CSV (`.pretwfix`) + npz (`.pretwfix`) first; removes live npz → rebuilds from new CSV at next training.
- Verified (082109): bicycle `|v|` median 0.005 → 2.597, std 0.20 → 3.84.

**Retrain launcher:** `run_carla20260627_early_twfix.sh` (fresh log_dir so it trains from scratch, not resume).

---

## 4. Other key findings this session

**Fusion collapse (resolved, not a blocker):** the **attention** fusion arms (self_attn / TACA) with
the staged frozen-warmup recipe **collapse to all-static** on the new benchmark. Root cause =
**fragile staged-warmup × a weak Phase-1 encoder** (car 0.24 on the harder new data vs 0.45 on the
old set), NOT the dataset/code/cache (all exonerated; definitive A/B with the Jun-18 encoder + old
data recovers). **Early fusion** (no attention block, end-to-end from scratch) **trains fine** — that's
why it's the arm used here. The TACA≈self_attn fusion-equivalence goal was **already met on the old
11-capture data** (`fusion-no-gain-on-8radar-rig`, `radargnn-selfattn-11capture-result`).

---

## 5. Open / next steps

1. **Other arms on the fixed features:** run RadarGNN-5d / no-fusion on the regenerated captures
   (fix already in `vx/vy`). The **attention** arms (self_attn/TACA) still need the weak-encoder /
   fragile-warmup issue solved first.
2. **CAR** (~0.25): separate static-rig geometry-only-Doppler problem — needs its own treatment if
   you want it higher.
3. **Commit** the uncommitted `PostProcessDataset.py` change + the two new scripts.
4. Optionally run the **5-fold CV** (defined in the split JSON) for a more robust headline.

## 6. Environment / entry points
- Train env: `/home/tamoghnd/miniconda3/envs/radarscenes_segmentation/bin/python`
- Main trainer: `PointNet/Pnet_pytorch/train_radar_semseg_msg_attn.py`
  (flags: `--fusion_type {none,early,self_attn,taca}`, `--encoder_type {pointnet,radargnn}`,
   `--dataset carla --carla_campaign_split <split.json> --carla_cleaning none`)
- Eval: `PointNet/Pnet_pytorch/eval_carla.py --checkpoint <ck> --split test`
- BN contract (CLAUDE.md): select on **eval-mode** mIoU; both early runs had a benign BN gap.
