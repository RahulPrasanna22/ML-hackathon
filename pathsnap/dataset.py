"""Build the candidate-level supervised dataset for a partition (train / student_test).

Labels come from `labels.observation_labels` (reference route alignment) and are joined
AFTER feature computation, so features can never see them.

Columns: trip_id, obs, arc, arc_gid, <FEATURE_COLUMNS>, y (1 if candidate == aligned
reference arc), label_dist (fix-to-reference-arc distance), reliable (label_dist <= 30 m).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, io
from .features import FEATURE_COLUMNS, compute_features
from .labels import observation_labels
from .network import RoadNetwork
from .preprocess import preprocess_track

RELIABLE_LABEL_M = 30.0


def build_trip(partition: str, trip_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    trip = io.load_trip(partition, trip_id)
    df, proj, info = preprocess_track(trip.track)
    net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
    feats = compute_features(df, net)
    # ---- labels (reference information; joined only after features are computed)
    route = io.load_reference_route(partition, trip_id)
    lab, pos, ldist = observation_labels(df, net, route)
    feats["y"] = (feats["arc"].to_numpy() == lab[feats["obs"].to_numpy()]).astype(np.int8)
    feats["label_dist"] = ldist[feats["obs"].to_numpy()]
    feats["reliable"] = (feats["label_dist"] <= RELIABLE_LABEL_M).astype(np.int8)
    feats.insert(0, "trip_id", trip_id)
    obs_tab = pd.DataFrame({"trip_id": trip_id, "obs": np.arange(len(df)), "label_arc_gid": net.arc_gid[lab],
                            "label_dist": ldist, "route_pos": pos})
    covered = feats.groupby("obs")["y"].max()
    obs_tab["label_in_cands"] = covered.reindex(obs_tab.obs).fillna(0).astype(int).to_numpy()
    meta = {"trip_id": trip_id, "n_obs": len(df), "n_rows": len(feats), "seconds": time.time() - t0,
            "miss_rate": float(1 - obs_tab.label_in_cands.mean()),
            "miss_rate_reliable": float(1 - obs_tab.label_in_cands[obs_tab.label_dist <= RELIABLE_LABEL_M].mean())}
    cols = ["trip_id", "obs", "arc", "arc_gid"] + FEATURE_COLUMNS + ["y", "label_dist", "reliable"]
    return feats[cols], obs_tab, meta


def build_partition(partition: str, n_jobs: int = -1, verbose: bool = True, trips: list[str] | None = None):
    trips = io.list_trips(partition) if trips is None else trips
    out = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(delayed(build_trip)(partition, t) for t in trips)
    feats = pd.concat([o[0] for o in out], ignore_index=True)
    obs = pd.concat([o[1] for o in out], ignore_index=True)
    meta = pd.DataFrame([o[2] for o in out])
    return feats, obs, meta
