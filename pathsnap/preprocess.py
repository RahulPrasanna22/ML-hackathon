"""GPS preprocessing shared by training and inference (identical code path).

Steps
1. drop rows with NaN / out-of-range / (0,0) coordinates
2. stable sort by timestamp, drop duplicated timestamps (keep first)
3. project into the trip's local metric plane
4. per-step displacement, dt, speed; robust centred heading; outlier flags

No learned parameters are involved, so there is nothing that could be fitted on the
wrong partition. The original row index is kept (`orig_idx`) so predictions can be
reported against the raw observation numbering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import LocalProjection, bearing_xy


def clean_track(track: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = track.copy()
    df["orig_idx"] = np.arange(len(df))
    n0 = len(df)
    ok = df[["lon", "lat", "t"]].notna().all(axis=1)
    ok &= (df["lon"].abs() <= 180) & (df["lat"].abs() <= 90)
    ok &= ~((df["lon"] == 0) & (df["lat"] == 0))
    df = df[ok]
    n_invalid = n0 - len(df)
    df = df.sort_values("t", kind="stable")
    dup = df["t"].duplicated(keep="first")
    n_dup_t = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    return df, {"n_raw": n0, "n_invalid_dropped": int(n_invalid), "n_duplicate_t_dropped": n_dup_t, "n_clean": len(df)}


def make_projection(track: pd.DataFrame) -> LocalProjection:
    return LocalProjection(float(np.median(track["lon"])), float(np.median(track["lat"])))


def _centred_heading(x: np.ndarray, y: np.ndarray, min_disp: float, windows=(1, 2, 3, 5, 8)) -> np.ndarray:
    n = len(x)
    heading = np.full(n, np.nan)
    idx = np.arange(n)
    for w in windows:
        lo = np.clip(idx - w, 0, n - 1)
        hi = np.clip(idx + w, 0, n - 1)
        dx, dy = x[hi] - x[lo], y[hi] - y[lo]
        disp = np.hypot(dx, dy)
        sel = np.isnan(heading) & (disp >= min_disp)
        heading[sel] = bearing_xy(dx[sel], dy[sel])
    return heading


def preprocess_track(track: pd.DataFrame, proj: LocalProjection | None = None) -> tuple[pd.DataFrame, LocalProjection, dict]:
    df, info = clean_track(track)
    if proj is None:
        proj = make_projection(df)
    x, y = proj.forward(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    t = df["t"].to_numpy()
    n = len(df)
    dt_prev = np.r_[np.nan, np.diff(t)]
    step_prev = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    df["dt_prev"] = dt_prev
    df["dt_next"] = np.r_[np.diff(t), np.nan]
    df["step_prev"] = step_prev
    df["step_next"] = np.r_[step_prev[1:], np.nan]
    with np.errstate(invalid="ignore", divide="ignore"):
        df["speed_prev"] = step_prev / dt_prev
        lo = np.clip(np.arange(n) - 1, 0, n - 1)
        hi = np.clip(np.arange(n) + 1, 0, n - 1)
        span_t = t[hi] - t[lo]
        span_d = np.hypot(x[hi] - x[lo], y[hi] - y[lo])
        df["speed"] = np.where(span_t > 0, span_d / span_t, 0.0)
    df["heading"] = _centred_heading(x, y, config.HEADING_MIN_DISP_M)
    df["heading_valid"] = df["heading"].notna()
    df["speed_outlier"] = (df["speed_prev"] > config.MAX_PLAUSIBLE_SPEED_MS).fillna(False)
    df["gap_prev"] = (df["dt_prev"] > 5).fillna(False)
    info["n_speed_outliers"] = int(df["speed_outlier"].sum())
    info["n_heading_invalid"] = int((~df["heading_valid"]).sum())
    return df, proj, info
