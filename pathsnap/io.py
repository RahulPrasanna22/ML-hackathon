"""Raw dataset loaders and validators.

Verified file schemas (see reports/EDA.md):
  <trip>.track  : 3 whitespace columns, no header -> lon, lat, t (seconds, float)
  <trip>.nodes  : 2 columns -> lon, lat; the 0-based row index is the node_id
  <trip>.arcs   : 2 integer columns -> from_node, to_node; the row index is the arc_id
  <trip>.route  : 1 integer column  -> chronological ground-truth arc_ids (reference only)

IMPORTANT (leakage): `load_trip` never reads the .route file. Reference routes are only
read through `load_reference_route`, which is used exclusively by label generation for
training and by the server-side evaluation module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # columns lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat
    arcs: np.ndarray             # (n_arcs, 2) from, to (int64)
    meta: dict = field(default_factory=dict)


def partition_dir(partition: str) -> Path:
    if partition == config.EVAL_PARTITION and config.EVAL_DIR_OVERRIDE:
        return Path(config.EVAL_DIR_OVERRIDE)
    return Path(config.DATA_ROOT) / partition


def list_trips(partition: str) -> list[str]:
    d = partition_dir(partition)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def _trip_path(partition: str, trip_id: str, ext: str) -> Path:
    return partition_dir(partition) / trip_id / f"{trip_id}.{ext}"


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if df.shape[1] != ncols:
        raise ValueError(f"{path.name}: expected {ncols} columns, found {df.shape[1]}")
    return df.to_numpy(dtype=dtype)


def load_track(partition: str, trip_id: str) -> pd.DataFrame:
    arr = _read_table(_trip_path(partition, trip_id, "track"), 3, np.float64)
    return pd.DataFrame(arr, columns=["lon", "lat", "t"])


def load_network(partition: str, trip_id: str) -> tuple[np.ndarray, np.ndarray]:
    nodes = _read_table(_trip_path(partition, trip_id, "nodes"), 2, np.float64)
    arcs = _read_table(_trip_path(partition, trip_id, "arcs"), 2, np.int64)
    return nodes, arcs


def load_trip(partition: str, trip_id: str) -> Trip:
    """Load everything needed for PREDICTION (GPS + road network). Never the route."""
    track = load_track(partition, trip_id)
    nodes, arcs = load_network(partition, trip_id)
    return Trip(trip_id=trip_id, partition=partition, track=track, nodes=nodes, arcs=arcs)


def has_reference_route(partition: str, trip_id: str) -> bool:
    return _trip_path(partition, trip_id, "route").exists()


def load_reference_route(partition: str, trip_id: str) -> np.ndarray:
    """Reference arc sequence. ONLY for label generation (train) and server-side evaluation."""
    p = _trip_path(partition, trip_id, "route")
    if not p.exists():
        raise FileNotFoundError(p)
    if p.stat().st_size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_trip(trip: Trip, route: Optional[np.ndarray] = None) -> dict:
    """Return a dict of data-quality checks for one trip (no mutation)."""
    tr = trip.track
    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    dt = np.diff(t) if len(t) > 1 else np.array([])
    rep = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_obs": int(len(tr)),
        "n_nodes": int(n_nodes),
        "n_arcs": int(n_arcs),
        "track_nan": int(tr.isna().sum().sum()),
        "track_coord_out_of_range": int(np.sum((np.abs(lon) > 180) | (np.abs(lat) > 90))),
        "track_zero_coords": int(np.sum((lon == 0) & (lat == 0))),
        "nodes_nan": int(np.isnan(trip.nodes).sum()),
        "nodes_out_of_range": int(np.sum((np.abs(trip.nodes[:, 0]) > 180) | (np.abs(trip.nodes[:, 1]) > 90))),
        "arcs_invalid_node_ref": int(np.sum((trip.arcs < 0) | (trip.arcs >= n_nodes))),
        "arcs_self_loops": int(np.sum(trip.arcs[:, 0] == trip.arcs[:, 1])),
        "arcs_duplicate": int(n_arcs - len(np.unique(trip.arcs, axis=0))) if n_arcs else 0,
        "t_non_increasing": int(np.sum(dt <= 0)),
        "t_duplicate": int(np.sum(dt == 0)),
        "dt_median": float(np.median(dt)) if len(dt) else float("nan"),
        "dt_max": float(dt.max()) if len(dt) else float("nan"),
        "frac_dt_1s": float(np.mean(np.isclose(dt, 1.0))) if len(dt) else float("nan"),
        "n_gaps_gt_5s": int(np.sum(dt > 5)),
        "n_gaps_gt_30s": int(np.sum(dt > 30)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
    }
    if route is not None:
        route = np.asarray(route)
        valid = (route >= 0) & (route < n_arcs)
        rep["route_len"] = int(len(route))
        rep["route_invalid_ids"] = int(np.sum(~valid))
        if valid.all() and len(route) > 1:
            a = trip.arcs[route]
            rep["route_connected_frac"] = float(np.mean(a[:-1, 1] == a[1:, 0]))
            rep["route_repeated_consecutive"] = int(np.sum(route[:-1] == route[1:]))
            rep["route_unique_arcs"] = int(len(np.unique(route)))
    return rep
