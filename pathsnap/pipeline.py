"""End-to-end prediction pipeline (GPS + road network -> matched arcs + route).

This module is the ONLY prediction path, used by evaluation scripts and the API alike.
It never reads reference routes: `match_trip` receives an `io.Trip` (track + network).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, compute_features
from .hmm import HMMParams, reconstruct_route, viterbi
from .io import Trip
from .network import RoadNetwork
from .preprocess import preprocess_track


class MapMatcher:
    def __init__(self, model, feature_columns: list[str], hmm_params: HMMParams, meta: dict | None = None):
        self.model = model
        self.feature_columns = feature_columns
        self.hmm_params = hmm_params
        self.meta = meta or {}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, bundle_dir: Path | str | None = None) -> "MapMatcher":
        d = Path(bundle_dir) if bundle_dir else config.ARTIFACT_DIR / "final"
        manifest = json.loads((d / "manifest.json").read_text())
        model = joblib.load(d / manifest["model_file"])
        feats = json.loads((d / "feature_schema.json").read_text())["feature_columns"]
        if feats != FEATURE_COLUMNS:
            raise RuntimeError("Feature schema of the saved model does not match the code's FEATURE_COLUMNS")
        hp = HMMParams(**json.loads((d / "hmm_params.json").read_text()))
        return cls(model, feats, hp, manifest)

    # ---------------------------------------------------------------- predict
    def score_candidates(self, feats: pd.DataFrame) -> np.ndarray:
        X = feats[self.feature_columns].to_numpy(np.float32)
        return self.model.predict_proba(X)[:, 1]

    def match_trip(self, trip: Trip, use_hmm: bool = True) -> dict:
        timings = {}
        t0 = time.perf_counter()
        df, proj, prep_info = preprocess_track(trip.track)
        net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
        timings["preprocess_network_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feats = compute_features(df, net)
        timings["candidates_features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feats["prob"] = self.score_candidates(feats)
        timings["classifier_s"] = time.perf_counter() - t0

        # classifier-only: argmax per fix
        top = feats.sort_values(["obs", "prob"], ascending=[True, False], kind="stable").drop_duplicates("obs")
        clf_arc = np.full(len(df), -1, dtype=np.int64)
        clf_arc[top.obs.to_numpy()] = top.arc.to_numpy()
        clf_prob = np.zeros(len(df))
        clf_prob[top.obs.to_numpy()] = top.prob.to_numpy()

        result = {"df": df, "net": net, "proj": proj, "feats": feats, "prep_info": prep_info,
                  "clf_arc_local": clf_arc, "clf_prob": clf_prob}
        if use_hmm:
            t0 = time.perf_counter()
            dec = viterbi(df, net, feats, "prob", self.hmm_params)
            timings["viterbi_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            route_local, unfilled = reconstruct_route(net, dec["arc_local"])
            timings["route_reconstruction_s"] = time.perf_counter() - t0
            result.update({"hmm_arc_local": dec["arc_local"], "n_breaks": dec["n_breaks"],
                           "route_local": route_local, "route_unfilled_gaps": unfilled})
        timings["total_s"] = sum(timings.values())
        result["timings"] = timings
        return result


def matched_points(net: RoadNetwork, df: pd.DataFrame, arc_local: np.ndarray) -> np.ndarray:
    """Project each fix onto its matched arc -> (n, 2) lon/lat (NaN where unmatched)."""
    from .geo import point_segment_distance
    out = np.full((len(df), 2), np.nan)
    ok = arc_local >= 0
    a = arc_local[ok]
    _, t = point_segment_distance(df.x.to_numpy()[ok], df.y.to_numpy()[ok], net.ax[a], net.ay[a], net.bx[a], net.by[a])
    qx = net.ax[a] + t * (net.bx[a] - net.ax[a]); qy = net.ay[a] + t * (net.by[a] - net.ay[a])
    lon, lat = net.proj.inverse(qx, qy)
    out[ok, 0], out[ok, 1] = lon, lat
    return out
