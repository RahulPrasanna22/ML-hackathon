"""First-review demo: GPS + road-network preprocessing on ONE train trip.

Reuses the production modules (pathsnap.io / preprocess / network / features / labels);
no model, no training, no Bayesian optimisation. Runs in a few seconds.

Usage (from the repository root):
    .venv\\Scripts\\python scripts\\demo_preprocessing.py --data-root "Student folder\\Student folder\\train" --trip 00000003

--data-root may point at the train/ folder itself or at the folder that contains train/.
Outputs (small, no raw dataset copies) go to reports/preprocessing_demo/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd

from pathsnap import config, io
from pathsnap.features import FEATURE_COLUMNS, compute_features, generate_candidates
from pathsnap.labels import observation_labels
from pathsnap.network import RoadNetwork, full_graph_stats
from pathsnap.preprocess import preprocess_track

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-root", default=str(config.DATA_ROOT / config.TRAIN_PARTITION),
                help="TRAIN folder (…/train) or its parent")
ap.add_argument("--trip", default=None, help="trip id (default: first train trip)")
ap.add_argument("--out", default=str(config.REPORT_DIR / "preprocessing_demo"))
ap.add_argument("--sample-rows", type=int, default=50, help="GPS fixes to export in the sample CSVs")
args = ap.parse_args()

root = Path(args.data_root)
if root.name != config.TRAIN_PARTITION:
    root = root / config.TRAIN_PARTITION
if not root.is_dir():
    sys.exit(f"TRAIN folder not found: {root}")
config.DATA_ROOT = root.parent                         # the loaders resolve <DATA_ROOT>/train/<trip>
PART = config.TRAIN_PARTITION
trips = io.list_trips(PART)
tid = args.trip or trips[0]
if tid not in trips:
    sys.exit(f"trip {tid} not in {root} ({len(trips)} trips available)")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
T = {}


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def safe_write(path: Path, writer) -> Path:
    """Write via a temp file + atomic replace.

    On Windows, if the previous output is open in an image viewer / editor preview it is
    often memory-mapped; truncating it then fails with OSError [Errno 22] (or Errno 13 for
    a plain lock, e.g. a CSV open in Excel). In that case keep the new output under a
    timestamped name instead of crashing.
    """
    import os
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    writer(tmp)
    try:
        os.replace(tmp, path)
        return path
    except OSError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
        os.replace(tmp, alt)
        print(f"  [note] {path.name} is open in another program (viewer/editor); wrote {alt.name} instead")
        return alt


# 1. raw files ---------------------------------------------------------------
section(f"1. RAW FILES  (partition=train, trip={tid}, {len(trips)} train trips found)")
d = root / tid
for p in sorted(d.iterdir()):
    print(f"  {p.name:<16} {p.stat().st_size / 1024:>9.1f} KB")
t0 = time.perf_counter()
trip = io.load_trip(PART, tid)               # track + nodes + arcs (never the .route)
T["load_ms"] = 1000 * (time.perf_counter() - t0)
tr = trip.track
dt = np.diff(tr.t.to_numpy())
print(f"\n  .track : {len(tr)} GPS fixes  columns={list(tr.columns)}")
print(tr.head(3).to_string(index=False))
print(f"  time span {tr.t.min():.0f}-{tr.t.max():.0f} s | dt median {np.median(dt):.0f} s, max {dt.max():.0f} s, "
      f"{np.mean(np.isclose(dt, 1)):.1%} steps exactly 1 s")
print(f"  lon [{tr.lon.min():.5f}, {tr.lon.max():.5f}]  lat [{tr.lat.min():.5f}, {tr.lat.max():.5f}]")
print(f"  .nodes : {len(trip.nodes)} nodes (lon, lat)   .arcs : {len(trip.arcs)} directed arcs (from, to)")

# 2. validation ----------------------------------------------------------------
section("2. VALIDATION")
route = io.load_reference_route(PART, tid)   # TRAIN only: used for validation + labels, never for features
val = io.validate_trip(trip, route)
for k in ["track_nan", "track_coord_out_of_range", "track_zero_coords", "t_non_increasing", "t_duplicate",
          "n_gaps_gt_5s", "n_gaps_gt_30s", "nodes_nan", "arcs_invalid_node_ref", "arcs_self_loops",
          "arcs_duplicate", "route_len", "route_invalid_ids", "route_connected_frac"]:
    print(f"  {k:<28} {val[k]}")
g = full_graph_stats(trip.nodes, trip.arcs)
print(f"  network: {g['weak_components']} weakly connected components (largest = {g['largest_weak_frac']:.1%} of nodes), "
      f"{g['isolated_nodes']} isolated nodes, {g['two_way_frac']:.1%} of arcs have a reverse twin")

# 3. GPS cleaning + projection ------------------------------------------------
section("3. GPS CLEANING, PROJECTION, KINEMATICS")
t0 = time.perf_counter()
df, proj, info = preprocess_track(trip.track)
T["preprocess_ms"] = 1000 * (time.perf_counter() - t0)
print(f"  cleaning: {info}")
print(f"  local metric projection centred at lon={proj.lon0:.5f}, lat={proj.lat0:.5f} "
      f"({proj.kx:.1f} m/deg lon, {proj.ky:.1f} m/deg lat)")
print(f"  path length {np.nansum(df.step_prev) / 1000:.2f} km, speed median {np.nanmedian(df.speed) * 3.6:.1f} km/h, "
      f"p95 {np.nanpercentile(df.speed, 95) * 3.6:.1f} km/h, heading valid for {df.heading_valid.mean():.1%} of fixes")

# 4. road network ------------------------------------------------------------
section("4. ROAD NETWORK (crop + spatial index + directed graph)")
t0 = time.perf_counter()
net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
T["network_ms"] = 1000 * (time.perf_counter() - t0)
print(f"  cropped to GPS bbox + {config.NETWORK_CROP_BUFFER_M:.0f} m: {net.n_arcs} / {net.n_arcs_total} arcs, "
      f"{net.n_nodes} / {net.n_nodes_total} nodes")
print(f"  arc length median {np.median(net.length):.1f} m, max {net.length.max():.0f} m; "
      f"KD-tree over {len(net._sample_owner)} densified points ({config.INDEX_SAMPLE_STEP_M:.0f} m step)")
print(f"  reference arcs lost by cropping: {sum(int(g) not in net.gid_to_local for g in route)}")

# 5. candidates -----------------------------------------------------------------
section(f"5. CANDIDATE ROAD SEGMENTS (radius {config.CANDIDATE_RADIUS_M:.0f} m, max {config.CANDIDATE_MAX_K})")
t0 = time.perf_counter()
cands = generate_candidates(df, net)
T["candidates_ms"] = 1000 * (time.perf_counter() - t0)
per = cands.groupby("obs").size()
print(f"  {len(cands)} candidate rows for {len(df)} fixes: per fix min {per.min()}, median {per.median():.0f}, max {per.max()}")
print(f"  nearest-candidate distance: median {cands[cands.dist_rank == 0].dist.median():.1f} m, "
      f"p95 {cands[cands.dist_rank == 0].dist.quantile(.95):.1f} m")

# 6. features -------------------------------------------------------------------
section(f"6. FEATURE EXTRACTION ({len(FEATURE_COLUMNS)} features per fix x candidate)")
t0 = time.perf_counter()
feats = compute_features(df, net, cands.copy())
T["features_ms"] = 1000 * (time.perf_counter() - t0)
print("  " + ", ".join(FEATURE_COLUMNS))
print(f"  missing values in feature matrix: {int(feats[FEATURE_COLUMNS].isna().sum().sum())}")
print(feats[FEATURE_COLUMNS].describe().T[["mean", "min", "max"]].round(2).head(12).to_string())

# 7. labels (TRAIN only) ------------------------------------------------------
section("7. TRAINING LABELS (TRAIN only: reference route aligned to GPS fixes, joined after features)")
lab, pos, ldist = observation_labels(df, net, route)
feats["y"] = (feats.arc.to_numpy() == lab[feats.obs.to_numpy()]).astype(int)
covered = feats.groupby("obs").y.max()
print(f"  fix -> aligned reference arc distance: median {np.median(ldist):.1f} m, p95 {np.percentile(ldist, 95):.1f} m")
print(f"  reliable labels (<= 30 m): {np.mean(ldist <= 30):.1%} | candidate miss rate: {1 - covered.mean():.2%}")
print(f"  positives {feats.y.sum()} / {len(feats)} candidate rows ({feats.y.mean():.1%})")

# 8. outputs ----------------------------------------------------------------------
section(f"8. OUTPUTS -> {out}")
n = args.sample_rows
raw_sample = trip.track.head(n).assign(index=np.arange(min(n, len(tr))))
safe_write(out / f"sample_raw_track_{tid}.csv", lambda f: raw_sample.to_csv(f, index=False))
clean_cols = ["orig_idx", "lon", "lat", "t", "x", "y", "dt_prev", "step_prev", "speed", "heading", "heading_valid",
              "speed_outlier", "gap_prev"]
safe_write(out / f"sample_clean_track_{tid}.csv", lambda f: df[clean_cols].head(n).round(4).to_csv(f, index=False))
safe_write(out / f"sample_candidate_features_{tid}.csv",
           lambda f: feats[feats.obs < n][["obs", "arc_gid"] + FEATURE_COLUMNS + ["y"]].round(4).to_csv(f, index=False))

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
near = net.arcs_near(float(df.x.median()), float(df.y.median()), 1e9)
for i in near:
    ax[0].plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="0.85", lw=.5)
rl = [net.gid_to_local[int(g)] for g in route if int(g) in net.gid_to_local]
for i in rl:
    ax[0].plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="tab:green", lw=2.5, alpha=.7)
ax[0].plot(df.x, df.y, ".", ms=2, color="tab:blue")
ax[0].set(title=f"Trip {tid}: GPS (blue), reference route (green), network (grey)", xlabel="x [m]", ylabel="y [m]",
          xlim=(df.x.min() - 200, df.x.max() + 200), ylim=(df.y.min() - 200, df.y.max() + 200), aspect="equal")
# zoom on 40 fixes with their candidates
z = df.iloc[len(df) // 2: len(df) // 2 + 40]
zc = feats[feats.obs.isin(z.index)]
for i in np.unique(zc.arc):
    ax[1].plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="tab:orange", lw=1)
for i in np.unique(zc[zc.y == 1].arc):
    ax[1].plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="tab:green", lw=3, alpha=.7)
ax[1].plot(z.x, z.y, "o", ms=3, color="tab:blue")
ax[1].set(title="Zoom: 40 fixes, candidate arcs (orange), labelled arc (green)", aspect="equal",
          xlim=(z.x.min() - 80, z.x.max() + 80), ylim=(z.y.min() - 80, z.y.max() + 80))
ax[2].hist(per, bins=np.arange(per.max() + 2) - .5, color="tab:orange")
ax[2].set(title="Candidates per GPS fix", xlabel="# candidates", ylabel="fixes")
fig.tight_layout()
safe_write(out / f"preprocessing_{tid}.png", lambda f: fig.savefig(f, dpi=110, format="png"))
plt.close(fig)

summary = {"trip_id": tid, "partition": "train", "raw": {"n_fixes": len(tr), "n_nodes": len(trip.nodes),
           "n_arcs": len(trip.arcs), "route_len": int(len(route)), "dt_max_s": float(dt.max()),
           "frac_dt_1s": float(np.mean(np.isclose(dt, 1)))},
           "validation": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in val.items()
                          if k not in ("trip_id", "partition")},
           "cleaning": info,
           "network": {"crop_arcs": net.n_arcs, "crop_nodes": net.n_nodes, **g},
           "candidates": {"rows": int(len(cands)), "per_fix_median": float(per.median()), "per_fix_max": int(per.max())},
           "features": {"n_features": len(FEATURE_COLUMNS), "rows": int(len(feats)),
                        "missing_values": int(feats[FEATURE_COLUMNS].isna().sum().sum())},
           "labels": {"median_dist_m": float(np.median(ldist)), "reliable_frac": float(np.mean(ldist <= 30)),
                      "candidate_miss_rate": float(1 - covered.mean()), "positive_rows": int(feats.y.sum())},
           "timings_ms": {k: round(v, 1) for k, v in T.items()}}
safe_write(out / f"summary_{tid}.json", lambda f: f.write_text(json.dumps(summary, indent=2, default=int)))

# offline-friendly Folium map: no external tile server (avoids OSM 403s); the road network is the background
from pathsnap.viz import base_map
m = base_map(df.lat.mean(), df.lon.mean())
fg = folium.FeatureGroup("Road network (cropped, near GPS)")
near_gps = np.unique(np.concatenate([net.arcs_near(float(x), float(y), 300.0)
                                     for x, y in zip(df.x.to_numpy()[::10], df.y.to_numpy()[::10])]))
for seg in net.arc_lonlat(near_gps):
    folium.PolyLine(seg, color="#9e9e9e", weight=1.5, opacity=.7).add_to(fg)
fg.add_to(m)
fg = folium.FeatureGroup("Reference route (TRAIN label source)")
for seg in net.arc_lonlat(rl):
    folium.PolyLine(seg, color="#2e7d32", weight=6, opacity=.6).add_to(fg)
fg.add_to(m)
fg = folium.FeatureGroup("Raw GPS fixes")
for la, lo, ti in zip(df.lat, df.lon, df.t):
    folium.CircleMarker([la, lo], radius=2, color="#1565c0", fill=True, weight=0, fill_opacity=.9,
                        tooltip=f"t={ti:.0f}s").add_to(fg)
fg.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[df.lat.min(), df.lon.min()], [df.lat.max(), df.lon.max()]])
safe_write(out / f"map_{tid}.html", lambda f: m.save(str(f)))
for p in sorted(out.glob(f"*{tid}*")):
    print(f"  {p.name}")

section("SUMMARY")
print(f"  trip {tid}: {len(tr)} raw fixes -> {len(df)} clean fixes -> {len(cands)} candidate rows x {len(FEATURE_COLUMNS)} features")
print(f"  network {len(trip.arcs)} arcs -> {net.n_arcs} after crop; candidate miss rate {1 - covered.mean():.2%}")
print("  timings: " + ", ".join(f"{k}={v:.0f} ms" for k, v in T.items()))
