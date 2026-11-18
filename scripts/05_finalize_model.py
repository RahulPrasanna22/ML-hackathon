"""Phase 10: refit the selected model with the Bayesian-optimised hyperparameters,
compare tuned vs untuned on identical folds/partitions, produce out-of-fold (OOF)
probabilities on the train partition for decoder tuning, and write the final artifact bundle.

Usage: python scripts/05_finalize_model.py --model lightgbm
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from pathsnap import __version__, config, io
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.hmm import HMMParams
from pathsnap.metrics import candidate_metrics, observation_metrics
from pathsnap.models import MODEL_SPECS, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lightgbm")
args = ap.parse_args()
EXP = config.REPORT_DIR / "experiments"
best = json.loads((EXP / f"optuna_{args.model}_best.json").read_text())
tuned_params = best["best_params"]

tr_all = pd.read_parquet(config.CACHE_DIR / "features_train.parquet")
te = pd.read_parquet(config.CACHE_DIR / "features_student_test.parquet")
tr = tr_all[tr_all.reliable == 1].reset_index(drop=True)
arcs_by_trip = {t: io.load_network(p, t)[1] for p, d in [("train", tr_all), ("student_test", te)] for t in d.trip_id.unique()}
X, y, groups = tr[FEATURE_COLUMNS].to_numpy(np.float32), tr.y.to_numpy(), tr.trip_id.to_numpy()
splits = list(GroupKFold(n_splits=5).split(X, y, groups))   # identical to scripts/03


def evaluate(df, p):
    d = df[["trip_id", "obs", "arc_gid", "y", "dist"]].assign(score=p)
    m = candidate_metrics(d.y.to_numpy(), p)
    m.update(observation_metrics(d, arcs_by_trip))
    m.pop("confusion")
    return m


rows = []
oof = np.full(len(tr_all), np.nan, dtype=np.float32)
fold_m = []
for fi, (tri, vai) in enumerate(splits):
    model = make_model(args.model, tuned_params)
    t0 = time.time(); model.fit(X[tri], y[tri]); fit_s = time.time() - t0
    fold_m.append({**evaluate(tr.iloc[vai], model.predict_proba(X[vai])[:, 1]), "fit_s": fit_s})
    val_trips = set(np.unique(groups[vai]))
    sel = tr_all.trip_id.isin(val_trips).to_numpy()
    oof[sel] = model.predict_proba(tr_all.loc[sel, FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    print(f"fold {fi}: top1={fold_m[-1]['top1_accuracy']:.4f} logloss={fold_m[-1]['log_loss']:.4f}", flush=True)
assert not np.isnan(oof).any()
pd.DataFrame({"trip_id": tr_all.trip_id, "obs": tr_all.obs, "arc": tr_all.arc, "prob": oof}).to_parquet(
    config.CACHE_DIR / "oof_train_final.parquet", index=False)

model = make_model(args.model, tuned_params)
t0 = time.time(); model.fit(X, y); fit_all = time.time() - t0
Xte = te[FEATURE_COLUMNS].to_numpy(np.float32)
t0 = time.time(); pte = model.predict_proba(Xte)[:, 1]; pred_s = time.time() - t0
n_obs_te = te[["trip_id", "obs"]].drop_duplicates().shape[0]
mte = evaluate(te, pte)

cmp_prev = pd.read_csv(EXP / "model_comparison.csv").set_index("model").loc[args.model]
tuned_row = {"variant": "tuned (Optuna)", "params": json.dumps({**MODEL_SPECS[args.model]["params"], **tuned_params}),
             "cv_top1_accuracy": np.mean([m["top1_accuracy"] for m in fold_m]),
             "cv_top1_std": np.std([m["top1_accuracy"] for m in fold_m]),
             "cv_cand_macro_f1": np.mean([m["cand_macro_f1"] for m in fold_m]),
             "cv_log_loss": np.mean([m["log_loss"] for m in fold_m]),
             "cv_route_continuity": np.mean([m["route_continuity"] for m in fold_m]),
             "st_top1_accuracy": mte["top1_accuracy"], "st_cand_macro_f1": mte["cand_macro_f1"],
             "st_log_loss": mte["log_loss"], "st_route_continuity": mte["route_continuity"],
             "fit_all_s": fit_all, "st_inference_ms_per_obs": 1000 * pred_s / n_obs_te}
untuned_row = {"variant": "untuned (defaults from MODEL_SPECS)", "params": cmp_prev["params"],
               "cv_top1_accuracy": cmp_prev["cv_top1_accuracy"], "cv_top1_std": cmp_prev["cv_top1_accuracy_std"],
               "cv_cand_macro_f1": cmp_prev["cv_cand_macro_f1"], "cv_log_loss": cmp_prev["cv_log_loss"],
               "cv_route_continuity": cmp_prev["cv_route_continuity"],
               "st_top1_accuracy": cmp_prev["st_top1_accuracy"], "st_cand_macro_f1": cmp_prev["st_cand_macro_f1"],
               "st_log_loss": cmp_prev["st_log_loss"], "st_route_continuity": cmp_prev["st_route_continuity"],
               "fit_all_s": cmp_prev["fit_all_s"], "st_inference_ms_per_obs": cmp_prev["st_inference_ms_per_obs"]}
cmp = pd.DataFrame([untuned_row, tuned_row])
cmp.to_csv(EXP / "tuned_vs_untuned.csv", index=False)
print(cmp.drop(columns="params").T.to_string())

# ------------------------------------------------------------ artifact bundle
final = config.ARTIFACT_DIR / "final"
final.mkdir(parents=True, exist_ok=True)
joblib.dump(model, final / "model.joblib", compress=3)
(final / "feature_schema.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS, "dtype": "float32",
                                                       "target": "y = candidate arc equals aligned reference arc"}, indent=2))
hp_path = final / "hmm_params.json"
if not hp_path.exists():
    hp_path.write_text(json.dumps(HMMParams().to_dict(), indent=2))
import lightgbm, xgboost
manifest = {
    "pathsnap_version": __version__,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "model_name": args.model,
    "model_family": MODEL_SPECS[args.model]["family"],
    "model_file": "model.joblib",
    "tuned": True,
    "optuna": {k: best[k] for k in ("best_value_cv_top1", "best_trial", "n_trials", "objective", "sampler")},
    "hyperparameters": {**MODEL_SPECS[args.model]["params"], **tuned_params},
    "training_data": "train partition (80 trips), observations with aligned reference label <= 30 m",
    "n_training_rows": int(len(tr)),
    "selection_metric": "cross-validated observation-level top-1 accuracy (GroupKFold by trip)",
    "metrics": {"cv": {k: float(v) for k, v in tuned_row.items() if k.startswith("cv_")},
                "student_test": {k: float(v) for k, v in tuned_row.items() if k.startswith("st_")}},
    "pipeline_config": {k: getattr(config, k) for k in ["CANDIDATE_RADIUS_M", "CANDIDATE_MAX_K", "CANDIDATE_MIN_K",
                        "INDEX_SAMPLE_STEP_M", "NETWORK_CROP_BUFFER_M", "MAX_PLAUSIBLE_SPEED_MS",
                        "HEADING_MIN_DISP_M", "CONTEXT_WINDOW", "RANDOM_SEED"]},
    "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__},
    "note": "Road networks are read per trip at inference time from the trip folder; nothing trip-specific is stored in the bundle.",
}
(final / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("bundle written to", final)
