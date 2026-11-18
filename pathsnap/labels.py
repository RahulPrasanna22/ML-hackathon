"""Derive per-observation reference labels from the reference route (TRAINING / EVALUATION ONLY).

The .route file gives the chronological arc sequence but NOT which GPS fix lies on which
arc. We align the two sequences with a monotone dynamic programme:

    C[i, j] = d(p_i, arc_{r_j}) + min_{k <= j} C[i-1, k]

i.e. fix i is assigned to route position j, positions never go backwards, and the total
point-to-segment distance is minimal. The row-wise prefix minimum makes this O(N * M).
Using route *positions* (not arc ids) handles routes that revisit an arc.

These labels are a proxy built from reference information and inherit its limitations
(e.g. fixes recorded before the route starts are clamped to the first arc).
This module must never be imported by the prediction pipeline.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment_distance


def align_track_to_route(x: np.ndarray, y: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                         bx: np.ndarray, by: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x,y: fixes (N,). ax..by: route arc endpoints in route order (M,).

    Returns (route_pos (N,), distance (N,)).
    """
    n, m = len(x), len(ax)
    if m == 0 or n == 0:
        return np.zeros(n, dtype=np.int64), np.full(n, np.nan)
    back = np.empty((n, m), dtype=np.int32)
    d0, _ = point_segment_distance(x[0], y[0], ax, ay, bx, by)
    prev = d0
    back[0] = np.arange(m)
    for i in range(1, n):
        d, _ = point_segment_distance(x[i], y[i], ax, ay, bx, by)
        # prefix min / argmin of prev
        pm = np.minimum.accumulate(prev)
        is_new = np.r_[True, prev[1:] < pm[:-1]]
        arg = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
        back[i] = arg
        prev = d + pm
    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(prev))
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    dist, _ = point_segment_distance(x, y, ax[pos], ay[pos], bx[pos], by[pos])
    return pos, dist


def observation_labels(track_df, net, route_gid: np.ndarray):
    """Return (label_arc_local (N,), route_pos (N,), dist_to_label (N,)).

    label_arc_local = -1 for route arcs that were cropped away (should not happen with
    the default buffer; reported in EDA).
    """
    route_local = np.array([net.gid_to_local.get(int(g), -1) for g in route_gid], dtype=np.int64)
    valid = route_local >= 0
    rl = route_local.copy()
    rl[~valid] = 0
    ax, ay, bx, by = net.ax[rl], net.ay[rl], net.bx[rl], net.by[rl]
    # invalid (cropped) arcs get infinite cost via far-away coordinates
    big = 1e9
    ax = np.where(valid, ax, big); bx = np.where(valid, bx, big)
    pos, dist = align_track_to_route(track_df["x"].to_numpy(), track_df["y"].to_numpy(), ax, ay, bx, by)
    return route_local[pos], pos, dist
