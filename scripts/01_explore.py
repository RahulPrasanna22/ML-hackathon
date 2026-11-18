"""Phase 2 - dataset exploration (train + student_test partitions only).

Outputs
  reports/eda_trip_stats.csv      per-trip inventory + quality checks + network stats
  reports/eda_label_stats.csv     GPS-to-reference-arc distance / rank statistics (TRAIN drives decisions)
  reports/eda_summary.json        aggregated numbers quoted in reports/EDA.md
  reports/figures/eda_*.png       plots
  reports/maps/eda_trip_<id>.html Folium map (GPS + road network + reference route)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config, io
from pathsnap.labels import observation_labels
from pathsnap.network import RoadNetwork, full_graph_stats
from pathsnap.preprocess import preprocess_track

config.FIG_DIR.mkdir(parents=True, exist_ok=True)
config.MAP_DIR.mkdir(parents=True, exist_ok=True)

rows, label_rows = [], []
dist_samples = {"train": [], "student_test": []}
rank_samples = {"train": [], "student_test": []}
dt_all, speed_all = [], []
traj_examples = {}

for part in [config.TRAIN_PARTITION, config.STUDENT_TEST_PARTITION]:
    for tid in io.list_trips(part):
        t0 = time.time()
        trip = io.load_trip(part, tid)
        route = io.load_reference_route(part, tid)
        rep = io.validate_trip(trip, route)
        d = io.partition_dir(part) / tid
        rep.update({f"size_{p.suffix[1:]}_kb": round(p.stat().st_size / 1024, 1) for p in d.iterdir()})
        rep.update({f"has_{e}": (d / f"{tid}.{e}").exists() for e in ["track", "nodes", "arcs", "route"]})
        rep.update(full_graph_stats(trip.nodes, trip.arcs))
        df, proj, info = preprocess_track(trip.track)
        rep.update(info)
        rep["lon_min"], rep["lon_max"] = df.lon.min(), df.lon.max()
        rep["lat_min"], rep["lat_max"] = df.lat.min(), df.lat.max()
        rep["lon_c"], rep["lat_c"] = proj.lon0, proj.lat0
        rep["extent_x_km"] = (df.x.max() - df.x.min()) / 1000
        rep["extent_y_km"] = (df.y.max() - df.y.min()) / 1000
        rep["path_len_km"] = np.nansum(df.step_prev) / 1000
        rep["speed_median_ms"] = float(np.nanmedian(df.speed))
        rep["speed_p99_ms"] = float(np.nanpercentile(df.speed, 99))
        net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
        rep["crop_arcs"] = net.n_arcs
        rep["route_arcs_cropped_away"] = int(sum(int(g) not in net.gid_to_local for g in route))
        rep["route_len_km"] = float(sum(net.length[net.gid_to_local[int(g)]] for g in route if int(g) in net.gid_to_local) / 1000)
        lab, pos, dist = observation_labels(df, net, route)
        rep["label_dist_median"] = float(np.median(dist))
        rep["label_dist_p95"] = float(np.percentile(dist, 95))
        rep["label_dist_max"] = float(dist.max())
        rep["route_arcs_with_obs"] = int(len(np.unique(pos)))
        # rank of the labelled arc among nearest arcs (distance), on a subsample
        sub = np.arange(0, len(df), 5)
        for i in sub:
            cand = net.arcs_near(df.x[i], df.y[i], 200.0)
            if len(cand) == 0:
                rank_samples[part].append(999); continue
            dd, _ = net.distance_to_arcs(df.x[i], df.y[i], cand)
            order = cand[np.argsort(dd, kind="stable")]
            hit = np.nonzero(order == lab[i])[0]
            rank_samples[part].append(int(hit[0]) + 1 if len(hit) else 999)
        dist_samples[part].append(dist)
        if part == config.TRAIN_PARTITION:
            dt_all.append(np.diff(df.t.to_numpy()))
            speed_all.append(df.speed.to_numpy())
        if len(traj_examples) < 6 and part == config.TRAIN_PARTITION:
            traj_examples[tid] = (df.x.to_numpy(), df.y.to_numpy(), net, lab)
        rep["seconds"] = round(time.time() - t0, 2)
        rows.append(rep)
        print(part, tid, rep["n_obs"], "obs", rep["n_arcs"], "arcs", f"lab med {rep['label_dist_median']:.1f}m", f"{rep['seconds']}s", flush=True)

stats = pd.DataFrame(rows)
stats.to_csv(config.REPORT_DIR / "eda_trip_stats.csv", index=False)

summary = {}
for part, g in stats.groupby("partition"):
    dist = np.concatenate(dist_samples[part])
    ranks = np.array(rank_samples[part])
    summary[part] = {
        "n_trips": int(len(g)),
        "n_obs_total": int(g.n_obs.sum()),
        "n_obs_min": int(g.n_obs.min()), "n_obs_median": float(g.n_obs.median()), "n_obs_max": int(g.n_obs.max()),
        "n_nodes_median": float(g.n_nodes.median()), "n_nodes_max": int(g.n_nodes.max()),
        "n_arcs_median": float(g.n_arcs.median()), "n_arcs_max": int(g.n_arcs.max()),
        "route_len_median": float(g.route_len.median()),
        "track_nan_total": int(g.track_nan.sum()),
        "invalid_coords_total": int(g.track_coord_out_of_range.sum() + g.track_zero_coords.sum()),
        "t_non_increasing_total": int(g.t_non_increasing.sum()),
        "frac_dt_1s_mean": float(g.frac_dt_1s.mean()),
        "dt_max_overall": float(g.dt_max.max()),
        "trips_with_gap_gt_30s": int((g.n_gaps_gt_30s > 0).sum()),
        "isolated_nodes_total": int(g.isolated_nodes.sum()),
        "weak_components_median": float(g.weak_components.median()),
        "largest_weak_frac_min": float(g.largest_weak_frac.min()),
        "two_way_frac_median": float(g.two_way_frac.median()),
        "route_connected_frac_mean": float(g.route_connected_frac.mean()),
        "route_invalid_ids_total": int(g.route_invalid_ids.sum()),
        "route_arcs_cropped_away_total": int(g.route_arcs_cropped_away.sum()),
        "label_dist_percentiles": {p: float(np.percentile(dist, p)) for p in [50, 75, 90, 95, 98, 99, 99.5, 99.9]},
        "label_within_radius": {r: float(np.mean(dist <= r)) for r in [10, 20, 30, 40, 50, 75, 100]},
        "gt_rank_le_k": {k: float(np.mean(ranks <= k)) for k in [1, 2, 3, 4, 6, 8, 12, 16, 24]},
        "speed_outliers_total": int(g.n_speed_outliers.sum()),
        "duplicate_t_dropped_total": int(g.n_duplicate_t_dropped.sum()),
    }
(config.REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for part, g in stats.groupby("partition"):
    ax[0].hist(g.n_obs, bins=20, alpha=.6, label=part)
    ax[1].hist(np.log10(g.n_arcs), bins=20, alpha=.6, label=part)
    ax[2].hist(g.path_len_km, bins=20, alpha=.6, label=part)
ax[0].set(title="GPS observations per trip", xlabel="n observations", ylabel="trips")
ax[1].set(title="Road-network size per trip", xlabel="log10(#arcs)")
ax[2].set(title="GPS path length per trip", xlabel="km")
for a in ax: a.legend()
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_trip_sizes.png", dpi=120); plt.close(fig)

dt = np.concatenate(dt_all); sp = np.concatenate(speed_all)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
vals, cnts = np.unique(np.round(dt), return_counts=True)
ax[0].bar([str(int(v)) if v < 10 else ">=10" for v in np.minimum(vals, 10)], cnts, log=True)
ax[0].set(title="Sampling interval (train)", xlabel="dt [s] (rounded)", ylabel="count (log)")
ax[1].hist(sp[np.isfinite(sp)] * 3.6, bins=80, range=(0, 200)); ax[1].set(title="GPS speed (train)", xlabel="km/h")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_sampling_speed.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for part in dist_samples:
    d = np.concatenate(dist_samples[part]); ax[0].hist(np.minimum(d, 100), bins=100, alpha=.6, density=True, label=part)
    r = np.array(rank_samples[part]); ks = np.arange(1, 25); ax[1].plot(ks, [np.mean(r <= k) for k in ks], marker="o", label=part)
ax[0].set(title="Distance GPS fix -> aligned reference arc", xlabel="metres (clipped at 100)"); ax[0].legend()
ax[1].set(title="Reference arc within K nearest arcs", xlabel="K", ylabel="fraction of fixes"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_label_distance_rank.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 5))
for part, g in stats.groupby("partition"):
    ax.scatter(g.lon_c, g.lat_c, s=25, label=part)
ax.set(title="Trip locations (median GPS position)", xlabel="longitude", ylabel="latitude"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_trip_locations.png", dpi=120); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for a, (tid, (x, y, net, lab)) in zip(axs.ravel(), traj_examples.items()):
    sel = net.arcs_near(np.median(x), np.median(y), 1e9) if net.n_arcs < 20000 else np.arange(net.n_arcs)
    for i in sel[:20000]:
        a.plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="0.8", lw=.4)
    a.plot(x, y, ".", ms=1.5, color="tab:blue")
    a.set(title=f"train trip {tid}", xlim=(x.min() - 300, x.max() + 300), ylim=(y.min() - 300, y.max() + 300), aspect="equal")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_trajectories.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
s = stats.sort_values("n_arcs")
ax.bar(range(len(s)), s.n_arcs, color=["tab:orange" if p == "student_test" else "tab:blue" for p in s.partition])
ax.set(yscale="log", title="Arcs per trip network (blue=train, orange=student_test) with weak components", xlabel="trip (sorted)")
ax2 = ax.twinx(); ax2.plot(range(len(s)), s.weak_components.values, "k.", ms=4); ax2.set_ylabel("weak components")
fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_network_connectivity.png", dpi=120); plt.close(fig)

# ------------------------------------------------------------------ folium map
tid = io.list_trips(config.TRAIN_PARTITION)[2]
trip = io.load_trip(config.TRAIN_PARTITION, tid)
df, proj, _ = preprocess_track(trip.track)
net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y, buffer_m=150)
route = io.load_reference_route(config.TRAIN_PARTITION, tid)
from pathsnap.viz import base_map  # no external tiles (OSM returns 403 for file:// maps)
m = base_map(df.lat.mean(), df.lon.mean(), zoom=14)
fg_net = folium.FeatureGroup("Road network (cropped arcs)")
for seg in net.arc_lonlat(np.arange(net.n_arcs)):
    folium.PolyLine(seg, color="#888888", weight=1.5, opacity=.6).add_to(fg_net)
fg_net.add_to(m)
fg_ref = folium.FeatureGroup("Reference route (.route)")
rl = [net.gid_to_local[int(g)] for g in route if int(g) in net.gid_to_local]
for seg in net.arc_lonlat(rl):
    folium.PolyLine(seg, color="green", weight=5, opacity=.8).add_to(fg_ref)
fg_ref.add_to(m)
fg_gps = folium.FeatureGroup("Raw GPS fixes")
for la, lo in zip(df.lat[::2], df.lon[::2]):
    folium.CircleMarker([la, lo], radius=2, color="blue", fill=True, fill_opacity=.8, weight=0).add_to(fg_gps)
fg_gps.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[df.lat.min(), df.lon.min()], [df.lat.max(), df.lon.max()]])
m.save(str(config.MAP_DIR / f"eda_trip_{tid}.html"))
print("saved map for trip", tid)
