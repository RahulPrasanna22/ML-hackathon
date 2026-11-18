"""Candidate generation and feature engineering.

For each cleaned GPS fix i we query the arc spatial index for arcs within
CANDIDATE_RADIUS_M, keep the CANDIDATE_MAX_K nearest (falling back to the nearest arcs
when nothing is inside the radius) and compute one feature row per (fix, candidate arc).

Only GPS + road-network information is used (never the reference route), so the same
function serves training and inference. Context features look at neighbouring fixes
(i-w .. i+w), which are all available at inference time because a full trip is matched
offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import angle_diff, point_segment_distance
from .network import RoadNetwork

FEATURE_COLUMNS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "heading_diff", "cos_heading", "heading_valid", "heading_rank", "wide_heading_diff",
    "speed", "arc_length", "proj_t", "dist_to_arc_end",
    "has_reverse", "to_out_deg", "from_in_deg",
    "ctx_dist_mean", "ctx_dist_max", "ctx_dist_rank",
    "prev_connected", "next_connected",
    "trans_dev_prev", "trans_reachable_prev", "trans_dev_next", "trans_reachable_next",
    "dt_prev", "step_prev",
]

TRANS_CAP = 500.0


def generate_candidates(df: pd.DataFrame, net: RoadNetwork, radius: float = config.CANDIDATE_RADIUS_M,
                        max_k: int = config.CANDIDATE_MAX_K) -> pd.DataFrame:
    """Return a long table (obs, arc, dist, proj_t) of candidate arcs per fix."""
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    lists = net._tree.query_ball_point(np.c_[x, y], radius + net._half_step) if net._tree is not None else [[] for _ in range(n)]
    obs_idx, arc_idx = [], []
    for i, idx in enumerate(lists):
        if idx:
            arcs = np.unique(net._sample_owner[idx])
        else:
            arcs = net.nearest_arcs(x[i], y[i], k=max(config.CANDIDATE_MIN_K, 4))
        obs_idx.append(np.full(len(arcs), i))
        arc_idx.append(arcs)
    obs = np.concatenate(obs_idx) if obs_idx else np.zeros(0, int)
    arc = np.concatenate(arc_idx) if arc_idx else np.zeros(0, int)
    d, t = net.distance_to_arcs(x[obs], y[obs], arc)
    c = pd.DataFrame({"obs": obs, "arc": arc, "dist": d, "proj_t": t})
    c = c.sort_values(["obs", "dist", "arc"], kind="stable")
    c["dist_rank"] = c.groupby("obs").cumcount()
    min_d = c.groupby("obs")["dist"].transform("min")
    # keep inside radius (always keep at least the nearest MIN_K arcs)
    keep = ((c["dist"] <= radius) | (c["dist_rank"] < config.CANDIDATE_MIN_K)) & (c["dist_rank"] < max_k)
    c = c[keep].reset_index(drop=True)
    c["min_dist"] = min_d[keep].to_numpy()
    return c


def _route_distance(net: RoadNetwork, a: int, pa: float, b: int, pb: float, limit: float,
                    same_arc_tol: float = 15.0) -> float:
    """Directed network distance between position pa (m) on arc a and pb on arc b."""
    if a == b and pb >= pa - same_arc_tol:
        return max(pb - pa, 0.0)
    dist = net.dijkstra_from(int(net.arc_to[a]), limit)
    dn = dist.get(int(net.arc_from[b]))
    if dn is None:
        return np.inf
    return (net.length[a] - pa) + dn + pb


def compute_features(df: pd.DataFrame, net: RoadNetwork, cands: pd.DataFrame | None = None) -> pd.DataFrame:
    if cands is None:
        cands = generate_candidates(df, net)
    c = cands
    obs = c["obs"].to_numpy()
    arc = c["arc"].to_numpy()
    n = len(df)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()

    c["n_cands"] = c.groupby("obs")["arc"].transform("size").astype(np.int16)
    c["dist_minus_min"] = c["dist"] - c["min_dist"]
    c["dist_ratio"] = c["dist"] / (c["min_dist"] + 1.0)

    heading = df["heading"].to_numpy()
    hv = df["heading_valid"].to_numpy()
    hd = angle_diff(heading[obs], net.bearing[arc])
    c["heading_valid"] = hv[obs].astype(np.int8)
    c["heading_diff"] = np.where(hv[obs], hd, 90.0)
    c["cos_heading"] = np.where(hv[obs], np.cos(np.radians(hd)), 0.0)
    c["heading_rank"] = c.groupby("obs")["heading_diff"].rank(method="min").astype(np.int16) - 1

    # wide-window heading (+/-5 fixes) for robustness at low speed
    lo = np.clip(np.arange(n) - 5, 0, n - 1); hi = np.clip(np.arange(n) + 5, 0, n - 1)
    wdx, wdy = x[hi] - x[lo], y[hi] - y[lo]
    wvalid = np.hypot(wdx, wdy) >= 5.0
    wbear = (np.degrees(np.arctan2(wdx, wdy)) + 360) % 360
    c["wide_heading_diff"] = np.where(wvalid[obs], angle_diff(wbear[obs], net.bearing[arc]), 90.0)

    c["speed"] = df["speed"].to_numpy()[obs]
    c["arc_length"] = net.length[arc]
    c["dist_to_arc_end"] = np.minimum(c["proj_t"], 1 - c["proj_t"]) * net.length[arc]
    c["has_reverse"] = net.has_reverse[arc].astype(np.int8)
    c["to_out_deg"] = net.out_deg[net.arc_to[arc]].astype(np.int16)
    c["from_in_deg"] = net.in_deg[net.arc_from[arc]].astype(np.int16)

    # context: distance of neighbouring fixes to the same arc
    w = config.CONTEXT_WINDOW
    ctx = []
    for o in range(-w, w + 1):
        j = np.clip(obs + o, 0, n - 1)
        d, _ = point_segment_distance(x[j], y[j], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        ctx.append(d)
    ctx = np.vstack(ctx)
    c["ctx_dist_mean"] = ctx.mean(axis=0)
    c["ctx_dist_max"] = ctx.max(axis=0)
    c["ctx_dist_rank"] = c.groupby("obs")["ctx_dist_mean"].rank(method="min").astype(np.int16) - 1

    # previous / next fix: nearest candidate (no labels involved)
    first = c[c["dist_rank"] == 0].set_index("obs")
    near_arc = np.full(n, -1, dtype=np.int64); near_t = np.zeros(n)
    near_arc[first.index.to_numpy()] = first["arc"].to_numpy()
    near_t[first.index.to_numpy()] = first["proj_t"].to_numpy()
    prev_arc = np.where(obs > 0, near_arc[np.maximum(obs - 1, 0)], -1)
    next_arc = np.where(obs < n - 1, near_arc[np.minimum(obs + 1, n - 1)], -1)
    pa = np.maximum(prev_arc, 0); na = np.maximum(next_arc, 0)
    c["prev_connected"] = ((prev_arc >= 0) & ((prev_arc == arc) | (net.arc_to[pa] == net.arc_from[arc]))).astype(np.int8)
    c["next_connected"] = ((next_arc >= 0) & ((next_arc == arc) | (net.arc_to[arc] == net.arc_from[na]))).astype(np.int8)

    # transition consistency with neighbouring fixes' nearest arcs:
    # | network distance - straight-line GPS displacement |
    pos = c["proj_t"].to_numpy() * net.length[arc]
    step_prev = df["step_prev"].to_numpy(); step_next = df["step_next"].to_numpy()
    dev_p = np.full(len(c), TRANS_CAP); dev_n = np.full(len(c), TRANS_CAP)
    reach_p = np.zeros(len(c), np.int8); reach_n = np.zeros(len(c), np.int8)
    for k in range(len(c)):
        i = obs[k]
        if i > 0 and near_arc[i - 1] >= 0:
            a = near_arc[i - 1]
            gc = step_prev[i]
            lim = 3 * gc + 2 * config.CANDIDATE_RADIUS_M + 100
            d = _route_distance(net, a, near_t[i - 1] * net.length[a], arc[k], pos[k], lim)
            if np.isfinite(d):
                dev_p[k] = min(abs(d - gc), TRANS_CAP); reach_p[k] = 1
        if i < n - 1 and near_arc[i + 1] >= 0:
            b = near_arc[i + 1]
            gc = step_next[i]
            lim = 3 * gc + 2 * config.CANDIDATE_RADIUS_M + 100
            d = _route_distance(net, arc[k], pos[k], b, near_t[i + 1] * net.length[b], lim)
            if np.isfinite(d):
                dev_n[k] = min(abs(d - gc), TRANS_CAP); reach_n[k] = 1
    c["trans_dev_prev"], c["trans_reachable_prev"] = dev_p, reach_p
    c["trans_dev_next"], c["trans_reachable_next"] = dev_n, reach_n
    c["dt_prev"] = np.nan_to_num(df["dt_prev"].to_numpy()[obs], nan=0.0)
    c["step_prev"] = np.nan_to_num(step_prev[obs], nan=0.0)
    c["arc_gid"] = net.arc_gid[arc]
    return c
