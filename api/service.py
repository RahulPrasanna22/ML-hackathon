"""Service layer: prediction and evaluation are deliberately separate.

PredictionService  - loads the artifact bundle, reads ONLY track + road network, runs
                     candidates -> features -> classifier -> HMM/Viterbi -> route, and
                     serialises the prediction in the Laptop 2 contract format.
EvaluationService  - server-side only: loads the reference route AFTER prediction and
                     computes aggregate metrics. Reference arcs/geometry never enter a response.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

from pathsnap import __version__, config, io
from pathsnap.evaluation import evaluate_prediction
from pathsnap.metrics import compress
from pathsnap.pipeline import MapMatcher

log = logging.getLogger("pathsnap.api")

DISPLAY_NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "hist_gb": "HistGradientBoosting",
                 "random_forest": "Random Forest", "extra_trees": "Extra Trees", "logreg": "Logistic Regression",
                 "gaussian_nb": "Gaussian Naive Bayes", "mlp": "MLP"}
ROAD_NETWORK_RADIUS_M = 150.0


class EvaluatorConfigError(RuntimeError):
    """PATHSNAP_EVAL_DIR was set explicitly but does not point at usable evaluator trips."""


def resolve_eval_partition() -> tuple[str, bool]:
    """Which partition the API serves.

    * PATHSNAP_EVAL_DIR set  -> that folder MUST contain valid trip folders; otherwise raise
      (never fall back silently to student_test when the evaluator data was asked for).
    * not set                -> DATA_ROOT/evaluator_test if present, else student_test (dev mode,
      reported by /health as evaluation_partition_is_fallback=true).
    """
    if config.EVAL_DIR_OVERRIDE:
        d = io.partition_dir(config.EVAL_PARTITION)
        if not d.is_dir():
            raise EvaluatorConfigError(f"PATHSNAP_EVAL_DIR={config.EVAL_DIR_OVERRIDE} does not exist or is not a folder")
        trips = io.list_trips(config.EVAL_PARTITION)
        if not trips:
            raise EvaluatorConfigError(f"PATHSNAP_EVAL_DIR={d} contains no trip folders of the form <id>/<id>.track")
        incomplete = [t for t in trips if not all((d / t / f"{t}.{e}").is_file() for e in ("track", "nodes", "arcs"))]
        if incomplete:
            raise EvaluatorConfigError(f"PATHSNAP_EVAL_DIR={d}: trips missing .track/.nodes/.arcs: {incomplete}")
        return config.EVAL_PARTITION, False
    if io.list_trips(config.EVAL_PARTITION):
        return config.EVAL_PARTITION, False
    return config.STUDENT_TEST_PARTITION, True


def _segments(net, arcs_local) -> list[dict]:
    a = np.asarray(arcs_local, dtype=np.int64)
    f, t = net.node_lonlat[net.arc_from[a]], net.node_lonlat[net.arc_to[a]]
    return [{"arc_id": int(g), "coordinates": [[float(p[0]), float(p[1])], [float(q[0]), float(q[1])]]}
            for g, p, q in zip(net.arc_gid[a], f, t)]


def _route(net, arcs_local) -> dict:
    return {"arc_ids": [int(net.arc_gid[a]) for a in arcs_local], "segments": _segments(net, arcs_local)}


def _nearby_arcs(net, df, radius=ROAD_NETWORK_RADIUS_M) -> np.ndarray:
    """Arcs whose midpoint lies within `radius` of any GPS fix (nearby subset for drawing)."""
    if net.n_arcs == 0:   # road network entirely outside the cropped trip area (length.max() would raise)
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(np.c_[df.x.to_numpy(), df.y.to_numpy()])
    mx, my = (net.ax + net.bx) / 2, (net.ay + net.by) / 2
    d, _ = tree.query(np.c_[mx, my], k=1, distance_upper_bound=radius + net.length.max() / 2)
    half = net.length / 2
    return np.nonzero(d <= radius + half)[0]


def _contract_metrics(m: dict | None, n_eval: int, miss: float) -> dict | None:
    """Map internal metric names to the contract's keys (+ Laptop-1 extras)."""
    if m is None:
        return None
    return {
        "top1_accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "precision": m["macro_precision"],
        "recall": m["macro_recall"], "log_loss": m["log_loss"], "route_continuity": m["continuity"],
        "network_reachable_continuity": m["reachable_continuity"], "sequence_accuracy": m["sequence_accuracy"],
        "candidate_miss_rate": miss, "accuracy_undirected": m["accuracy_undirected"],
        "route_precision": m["route_precision_len"], "route_recall": m["route_recall_len"],
        "route_f1": m["route_f1_len"], "route_jaccard": m["route_jaccard_arcs"],
        "edit_distance": float(m["edit_distance"]), "normalized_edit_distance": m["norm_edit_distance"],
        "invalid_transitions": float(m["invalid_transitions"]), "unreachable_transitions": float(m["unreachable_transitions"]),
        "mean_deviation_m": m["mean_deviation_m"], "p95_deviation_m": m["p95_deviation_m"],
        "predicted_route_arcs": float(m["pred_route_arcs"]), "observations_evaluated": float(n_eval),
    }


class PredictionService:
    def __init__(self, bundle_dir=None):
        self.matcher = MapMatcher.load(bundle_dir)
        m = self.matcher.meta
        algo = m.get("model_name", "unknown")
        tuned = " (tuned)" if m.get("tuned") else ""
        self.model_info = {"name": f"{DISPLAY_NAMES.get(algo, algo)}{tuned} + HMM/Viterbi",
                           "version": m.get("pathsnap_version", __version__), "algorithm": algo,
                           "family": m.get("model_family"), "created_utc": m.get("created_utc"),
                           "selection_metric": m.get("selection_metric")}

    def predict(self, partition: str, trip_id: str) -> tuple[dict, dict]:
        """Raises io.TripDataError if the trip's files are unreadable or unusable (mapped to HTTP 422).

        Non-fatal data-quality notes are returned in result["data_warnings"].
        """
        trip = io.load_trip(partition, trip_id)   # track + network only (never .route)
        data_warnings = io.check_trip_for_inference(trip)
        res = self.matcher.match_trip(trip)
        res["data_warnings"] = data_warnings
        net, df = res["net"], res["df"]
        hmm_arc = res["hmm_arc_local"]
        obs = [{"index": int(i), "lon": float(lo), "lat": float(la), "t": float(t),
                "matched_arc_id": int(net.arc_gid[a]) if a >= 0 else None}
               for i, lo, la, t, a in zip(df.orig_idx, df.lon, df.lat, df.t, hmm_arc)]
        payload = {
            "num_observations": int(len(trip.track)),
            "observations": obs,
            "predicted_route": _route(net, res["route_local"]),
            "classifier_only_route": _route(net, compress(res["clf_arc_local"])),
            "road_network": {"arc_ids": [], "segments": _segments(net, _nearby_arcs(net, df))},
        }
        return res, payload


class EvaluationService:
    """Server-side scoring + in-memory store of completed evaluations."""

    def __init__(self, max_items: int = 200, max_internal: int = 10):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._internal: OrderedDict[str, dict] = OrderedDict()  # pipeline objects for /map
        self._lock = threading.Lock()
        self.max_items, self.max_internal = max_items, max_internal

    @staticmethod
    def score(partition: str, trip_id: str, result: dict):
        """Returns (hmm, classifier_only, nearest_road) metric dicts, or (None, None, None)."""
        if not io.has_reference_route(partition, trip_id):
            return None, None, None
        ref = io.load_reference_route(partition, trip_id)   # loaded only after prediction finished
        ev = evaluate_prediction(result, ref)
        n, miss = ev["n_obs_evaluated"], ev["candidate_miss_rate"]
        return (_contract_metrics(ev["classifier_hmm"], n, miss), _contract_metrics(ev["classifier_only"], n, miss),
                _contract_metrics(ev["nearest_road"], n, miss))

    def save(self, record: dict, internal: dict | None = None) -> None:
        with self._lock:
            self._store[record["evaluation_id"]] = record
            while len(self._store) > self.max_items:
                self._store.popitem(last=False)
            if internal is not None:
                self._internal[record["evaluation_id"]] = internal
                while len(self._internal) > self.max_internal:
                    self._internal.popitem(last=False)

    def get(self, evaluation_id: str):
        with self._lock:
            return self._store.get(evaluation_id)

    def get_internal(self, evaluation_id: str):
        with self._lock:
            return self._internal.get(evaluation_id)

    def __len__(self):
        return len(self._store)


MODEL_LABELS = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "random_forest": "Random Forest",
                "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM",
                "xgboost": "XGBoost", "mlp": "MLP (64-32)", "nearest_road": "Nearest road (baseline)"}


def load_model_comparison() -> dict:
    """Recorded TRAIN cross-validation results of the 8 candidate classifiers (read-only).

    Source: reports/experiments/model_comparison.csv (scripts/03_train_models.py, 5-fold GroupKFold by trip
    on the TRAIN partition). Only TRAIN-CV columns and timings are returned; student-test columns are left out
    so that nothing here can be mistaken for evaluator results. Missing values become null (never invented).
    """
    import json as _json
    import math

    import pandas as pd

    exp = config.REPORT_DIR / "experiments"
    src = exp / "model_comparison.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run scripts/03_train_models.py")
    df = pd.read_csv(src)
    manifest_path = config.ARTIFACT_DIR / "final" / "manifest.json"
    manifest = _json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    selected = manifest.get("model_name")

    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    def row(r):
        return {"key": r["model"], "name": MODEL_LABELS.get(r["model"], r["model"]), "family": r["family"],
                "cv_top1_accuracy": num(r["cv_top1_accuracy"]), "cv_top1_accuracy_std": num(r.get("cv_top1_accuracy_std")),
                "cv_macro_f1": num(r.get("cv_cand_macro_f1")), "cv_macro_f1_std": num(r.get("cv_cand_macro_f1_std")),
                "cv_log_loss": num(r.get("cv_log_loss")), "cv_route_continuity": num(r.get("cv_route_continuity")),
                "train_fold_top1_accuracy": num(r.get("cv_train_top1_accuracy")),
                "fit_time_s_per_fold": num(r.get("cv_fit_s")), "fit_time_s_all_train": num(r.get("fit_all_s")),
                "inference_ms_per_observation": num(r.get("st_inference_ms_per_obs")),
                "selected": r["model"] == selected}

    models = [row(r) for _, r in df[df.model != "nearest_road"].iterrows()]
    base = df[df.model == "nearest_road"]
    out = {"source": "reports/experiments/model_comparison.csv",
           "evaluation": "TRAIN partition, 5-fold GroupKFold cross-validation grouped by trip (80 trips)",
           "not_evaluator_results": True,
           "selection_metric": manifest.get("selection_metric"),
           "selected_model": selected,
           "models": models,
           "baseline": row(base.iloc[0]) if len(base) else None,
           "tuned_final": None,
           "notes": ["Top-1 accuracy: fraction of GPS fixes whose highest-scoring candidate is the reference arc (fixes with reliable labels).",
                     "Macro-F1: candidate-level, threshold 0.5, macro-averaged over {not reference, reference}.",
                     "Fit times were measured while other jobs shared the CPU; treat them as indicative.",
                     "Inference time: classifier scoring only, per GPS fix."]}
    tv = exp / "tuned_vs_untuned.csv"
    if tv.exists():
        t = pd.read_csv(tv)
        tuned = t[t.variant.str.startswith("tuned")]
        if len(tuned):
            r = tuned.iloc[0]
            out["tuned_final"] = {"name": f"{MODEL_LABELS.get(selected, selected)} (Optuna-tuned, deployed)",
                                  "cv_top1_accuracy": num(r.cv_top1_accuracy), "cv_top1_accuracy_std": num(r.cv_top1_std),
                                  "cv_macro_f1": num(r.cv_cand_macro_f1), "cv_log_loss": num(r.cv_log_loss),
                                  "fit_time_s_all_train": num(r.fit_all_s),
                                  "inference_ms_per_observation": num(r.st_inference_ms_per_obs)}
    return out


def new_evaluation_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
