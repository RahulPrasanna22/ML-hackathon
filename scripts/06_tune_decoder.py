"""Phase 8: tune HMM/Viterbi decoder parameters (Optuna TPE) on TRAIN trips only.

Emissions use OUT-OF-FOLD probabilities of the final model (scripts/05), so the decoder is
tuned on probabilities that look like real test-time outputs. Objective: mean per-trip
pointwise accuracy of the decoded sequence (fixes with reliable labels). The student-test
partition is then used to confirm default vs tuned decoder parameters. The evaluator
partition is never used.

Usage: python scripts/06_tune_decoder.py --trials 30 --n-trips 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import pandas as pd

from pathsnap import config, io
from pathsnap.hmm import HMMParams, viterbi
from pathsnap.network import RoadNetwork
from pathsnap.preprocess import preprocess_track

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--n-trips", type=int, default=24)
args = ap.parse_args()

feats = pd.read_parquet(config.CACHE_DIR / "features_train.parquet")
oof = pd.read_parquet(config.CACHE_DIR / "oof_train_final.parquet")
assert (feats.obs.values == oof.obs.values).all() and (feats.arc.values == oof.arc.values).all()
feats["prob"] = oof.prob.values
obs_tab = pd.read_parquet(config.CACHE_DIR / "obs_train.parquet")

rng = np.random.default_rng(config.RANDOM_SEED)
trips = sorted(feats.trip_id.unique())
sel = sorted(rng.choice(trips, size=min(args.n_trips, len(trips)), replace=False))
print("tuning trips:", sel, flush=True)

cache = []
for tid in sel:
    trip = io.load_trip("train", tid)
    df, proj, _ = preprocess_track(trip.track)
    net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
    c = feats[feats.trip_id == tid].reset_index(drop=True)
    o = obs_tab[obs_tab.trip_id == tid].sort_values("obs")
    cache.append((tid, df, net, c, o.label_arc_gid.to_numpy(), (o.label_dist <= 30).to_numpy()))


def run(params: HMMParams, data) -> tuple[float, float]:
    accs, secs = [], 0.0
    for tid, df, net, c, lab, rel in data:
        t0 = time.time()
        dec = viterbi(df, net, c, "prob", params)
        secs += time.time() - t0
        pred = np.where(dec["arc_local"] >= 0, net.arc_gid[np.maximum(dec["arc_local"], 0)], -1)
        if rel.any():
            accs.append(np.mean(pred[rel] == lab[rel]))
    return float(np.mean(accs)), secs


# classifier-only and default-decoder references
clf_acc = []
for tid, df, net, c, lab, rel in cache:
    top = c.sort_values(["obs", "prob"], ascending=[True, False]).drop_duplicates("obs")
    pred = np.full(len(df), -1); pred[top.obs.values] = top.arc_gid.values
    clf_acc.append(np.mean(pred[rel] == lab[rel]))
default_acc, default_s = run(HMMParams(), cache)
print(f"classifier-only acc={np.mean(clf_acc):.4f} | default HMM acc={default_acc:.4f} ({default_s:.1f}s)", flush=True)


def objective(trial):
    p = HMMParams(
        beta=trial.suggest_float("beta", 1.0, 100.0, log=True),
        emission_weight=trial.suggest_float("emission_weight", 0.2, 3.0, log=True),
        sigma=trial.suggest_categorical("sigma", [0.0, 5.0, 10.0, 20.0, 40.0]),
        same_arc_tol=trial.suggest_float("same_arc_tol", 2.0, 30.0),
        break_penalty=trial.suggest_float("break_penalty", 5.0, 200.0, log=True),
        reverse_penalty=trial.suggest_float("reverse_penalty", 0.0, 10.0),
    )
    acc, secs = run(p, cache)
    trial.set_user_attr("decode_seconds", secs)
    return acc


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=8),
                            storage=f"sqlite:///{(config.ARTIFACT_DIR / 'optuna' / 'hmm_decoder.db').as_posix()}",
                            study_name="pathsnap_hmm", load_if_exists=True)
study.enqueue_trial({"beta": 10.0, "emission_weight": 1.0, "sigma": 0.0, "same_arc_tol": 15.0, "break_penalty": 50.0,
                     "reverse_penalty": 0.0})
t_opt = time.time()
study.optimize(objective, n_trials=args.trials)
t_opt = time.time() - t_opt
# Selection is made on TRAIN out-of-fold results only.
best = HMMParams(**{**HMMParams().to_dict(), **study.best_params})
(config.ARTIFACT_DIR / "final" / "hmm_params.json").write_text(json.dumps(best.to_dict(), indent=2))

# CONFIRMATION ONLY on student-test (final model probabilities, all observations).
# These numbers are reported, never used to choose parameters.
st = pd.read_parquet(config.CACHE_DIR / "features_student_test.parquet")
import joblib
from pathsnap.features import FEATURE_COLUMNS
final_model = joblib.load(config.ARTIFACT_DIR / "final" / "model.joblib")
st["prob"] = final_model.predict_proba(st[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
st_obs = pd.read_parquet(config.CACHE_DIR / "obs_student_test.parquet")
st_cache = []
for tid in sorted(st.trip_id.unique()):
    trip = io.load_trip("student_test", tid)
    df, proj, _ = preprocess_track(trip.track)
    net = RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y)
    o = st_obs[st_obs.trip_id == tid].sort_values("obs")
    st_cache.append((tid, df, net, st[st.trip_id == tid].reset_index(drop=True), o.label_arc_gid.to_numpy(), np.ones(len(o), bool)))
st_default, _ = run(HMMParams(), st_cache)
st_best, st_s = run(best, st_cache)

res = {"tuning_trips": [str(t) for t in sel], "n_trials": len(study.trials), "optimisation_seconds": t_opt,
       "selection": "best TRAIN out-of-fold accuracy; student-test numbers are confirmation only",
       "train_subset_classifier_only_acc": float(np.mean(clf_acc)),
       "train_subset_default_hmm_acc": default_acc, "train_subset_best_hmm_acc": study.best_value,
       "student_test_default_hmm_acc": st_default, "student_test_tuned_hmm_acc": st_best,
       "student_test_decode_seconds": st_s, "best_params": best.to_dict()}
print(json.dumps(res, indent=2))
out = config.REPORT_DIR / "experiments"
(out / "hmm_tuning.json").write_text(json.dumps(res, indent=2))
study.trials_dataframe().to_csv(out / "hmm_tuning_trials.csv", index=False)
print("hmm_params.json <- TRAIN-tuned parameters")
