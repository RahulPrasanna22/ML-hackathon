"""Phase 5-6: train >= 5 candidate-scoring models with trip-grouped 5-fold CV.

* Training rows: TRAIN partition, observations whose aligned reference label is reliable
  (fix within 30 m of the reference arc; see reports/EDA.md for why).
* CV: GroupKFold(5) grouped by trip_id -> no fix of a trip is in both train and validation folds.
* Student-test: each model is refit on all training rows and evaluated on the 10 student
  trips (all observations). The evaluator partition is never touched.
* Baseline A (nearest road) is evaluated with the same metric code.

Usage:  python scripts/03_train_models.py [--models logreg,lightgbm,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config, io
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, observation_metrics
from pathsnap.models import MODEL_SPECS, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(MODEL_SPECS))
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

MODEL_DIR = config.ARTIFACT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT = config.REPORT_DIR / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

tr = pd.read_parquet(config.CACHE_DIR / "features_train.parquet")
te = pd.read_parquet(config.CACHE_DIR / "features_student_test.parquet")
tr = tr[tr.reliable == 1].reset_index(drop=True)
arcs_by_trip = {t: io.load_network(p, t)[1] for p, d in [("train", tr), ("student_test", te)] for t in d.trip_id.unique()}
X, y, groups = tr[FEATURE_COLUMNS].to_numpy(np.float32), tr.y.to_numpy(), tr.trip_id.to_numpy()
Xte, yte = te[FEATURE_COLUMNS].to_numpy(np.float32), te.y.to_numpy()
print(f"train rows={len(tr)} obs={tr[['trip_id','obs']].drop_duplicates().shape[0]} | student_test rows={len(te)}", flush=True)

gkf = GroupKFold(n_splits=args.folds)
splits = list(gkf.split(X, y, groups))


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    d = df[["trip_id", "obs", "arc_gid", "y", "dist"]].copy()
    d["score"] = p
    m = candidate_metrics(d.y.to_numpy(), p)
    m.update(observation_metrics(d, arcs_by_trip))
    return m


fold_rows, summary = [], []

# ---------------- Baseline A: nearest road (no learning)
for fi, (_, va) in enumerate(splits):
    d = tr.iloc[va]
    s = -d.dist.to_numpy()
    m = observation_metrics(d.assign(score=s)[["trip_id", "obs", "arc_gid", "y", "dist", "score"]], arcs_by_trip)
    fold_rows.append({"model": "nearest_road", "fold": fi, **m})
m_te = observation_metrics(te.assign(score=-te.dist)[["trip_id", "obs", "arc_gid", "y", "dist", "score"]], arcs_by_trip)
summary.append({"model": "nearest_road", "family": "Geometric baseline (no learning)", "params": "{}",
                **{f"cv_{k}": np.mean([r[k] for r in fold_rows if r["model"] == "nearest_road"]) for k in ["top1_accuracy", "route_continuity"]},
                **{f"st_{k}": v for k, v in m_te.items()}})
print("nearest_road", summary[-1], flush=True)

for name in args.models.split(","):
    spec = MODEL_SPECS[name]
    print(f"=== {name}", flush=True)
    for fi, (tri, vai) in enumerate(splits):
        model = make_model(name)
        t0 = time.time(); model.fit(X[tri], y[tri]); t_fit = time.time() - t0
        t0 = time.time(); p = model.predict_proba(X[vai])[:, 1]; t_pred = time.time() - t0
        m = evaluate(tr.iloc[vai], p)
        # training-fold fit quality (overfitting evidence) on a 20 % subsample of training trips' rows
        sub = tri[tr.obs.to_numpy()[tri] % 5 == 0]  # every 5th observation, all its candidates
        m_tr = evaluate(tr.iloc[sub], model.predict_proba(X[sub])[:, 1])
        row = {"model": name, "fold": fi, "fit_s": t_fit, "predict_s": t_pred,
               "train_top1_accuracy": m_tr["top1_accuracy"], "train_log_loss": m_tr["log_loss"],
               "train_cand_macro_f1": m_tr["cand_macro_f1"],
               **{k: v for k, v in m.items() if k != "confusion"}, "confusion": json.dumps(m["confusion"])}
        fold_rows.append(row)
        print(f"  fold {fi}: top1={m['top1_accuracy']:.4f} macroF1={m['cand_macro_f1']:.4f} logloss={m['log_loss']:.4f} "
              f"cont={m['route_continuity']:.4f} fit={t_fit:.1f}s (train top1={m_tr['top1_accuracy']:.4f})", flush=True)
    # refit on all training rows -> student-test
    model = make_model(name)
    t0 = time.time(); model.fit(X, y); t_fit_all = time.time() - t0
    t0 = time.time(); pte = model.predict_proba(Xte)[:, 1]; t_pred = time.time() - t0
    mte = evaluate(te, pte)
    n_obs_te = te[["trip_id", "obs"]].drop_duplicates().shape[0]
    joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)
    np.save(OUT / f"st_scores_{name}.npy", pte.astype(np.float32))
    if name == "mlp":
        mlp = model[-1]
        pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                      "val_accuracy": mlp.validation_scores_}).to_csv(OUT / "mlp_training_history.csv", index=False)
    fr = [r for r in fold_rows if r["model"] == name]
    agg = {f"cv_{k}": float(np.mean([r[k] for r in fr])) for k in
           ["top1_accuracy", "top3_accuracy", "cand_macro_f1", "cand_precision", "cand_recall", "cand_f1",
            "cand_accuracy", "log_loss", "route_continuity", "candidate_miss_rate", "fit_s", "train_top1_accuracy",
            "train_log_loss", "train_cand_macro_f1"]}
    agg.update({f"cv_{k}_std": float(np.std([r[k] for r in fr])) for k in ["top1_accuracy", "cand_macro_f1", "log_loss"]})
    summary.append({"model": name, "family": spec["family"], "params": json.dumps(spec["params"], default=str),
                    **agg, "fit_all_s": t_fit_all, "st_inference_ms_per_obs": 1000 * t_pred / n_obs_te,
                    **{f"st_{k}": v for k, v in mte.items()}})
    print(f"  student-test: top1={mte['top1_accuracy']:.4f} macroF1={mte['cand_macro_f1']:.4f} "
          f"logloss={mte['log_loss']:.4f} cont={mte['route_continuity']:.4f}", flush=True)
    # write incrementally
    for fname, rows in [("cv_folds.csv", fold_rows), ("model_comparison.csv", summary)]:
        path = OUT / fname
        s = pd.DataFrame(rows)
        if path.exists():  # merge with results of models trained in earlier invocations
            prev = pd.read_csv(path)
            s = pd.concat([prev[~prev.model.isin(s.model)], s], ignore_index=True)
        s.to_csv(path, index=False)
print("done")
