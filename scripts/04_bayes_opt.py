"""Phase 7: Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Objective : maximise mean observation-level Top-1 accuracy over 5 trip-grouped folds
            (GroupKFold on the 80 TRAIN trips; the route decoder consumes the top candidate
            and its probability, so ranking quality per fix is what matters). CV log loss and
            candidate macro-F1 are stored per trial as user attributes.
Budget    : to fit the laptop budget, trials train on every 2nd fix of the training folds
            (fixes are 1 Hz and highly redundant) and validate on ALL fixes of the held-out
            trips. The best configuration is then refit on all training rows.
Records   : artifacts/optuna/<model>.db (SQLite study), reports/experiments/optuna_<model>_trials.csv,
            best params JSON, and plots in reports/figures/.

Usage: python scripts/04_bayes_opt.py --model lightgbm --trials 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config, io
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, observation_metrics
from pathsnap.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lightgbm")
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--timeout", type=int, default=3600)
args = ap.parse_args()


def space(trial: optuna.Trial, name: str) -> dict:
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "hist_gb":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 300, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
        }
    raise KeyError(name)


tr = pd.read_parquet(config.CACHE_DIR / "features_train.parquet")
tr = tr[tr.reliable == 1].reset_index(drop=True)
arcs_by_trip = {t: io.load_network("train", t)[1] for t in tr.trip_id.unique()}
X, y, groups = tr[FEATURE_COLUMNS].to_numpy(np.float32), tr.y.to_numpy(), tr.trip_id.to_numpy()
splits = list(GroupKFold(n_splits=5).split(X, y, groups))
half = tr.obs.to_numpy() % 2 == 0


def objective(trial: optuna.Trial) -> float:
    params = space(trial, args.model)
    tops, lls, f1s, fit_s = [], [], [], []
    for fi, (tri, vai) in enumerate(splits):
        tri = tri[half[tri]]
        model = make_model(args.model, params)
        t0 = time.time(); model.fit(X[tri], y[tri]); fit_s.append(time.time() - t0)
        p = model.predict_proba(X[vai])[:, 1]
        d = tr.iloc[vai][["trip_id", "obs", "arc_gid", "y", "dist"]].assign(score=p)
        om = observation_metrics(d, arcs_by_trip)
        cm = candidate_metrics(d.y.to_numpy(), p)
        tops.append(om["top1_accuracy"]); lls.append(cm["log_loss"]); f1s.append(cm["cand_macro_f1"])
        trial.report(float(np.mean(tops)), fi)
    trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
    trial.set_user_attr("cv_cand_macro_f1", float(np.mean(f1s)))
    trial.set_user_attr("cv_top1_std", float(np.std(tops)))
    trial.set_user_attr("fit_seconds_total", float(np.sum(fit_s)))
    return float(np.mean(tops))


store_dir = config.ARTIFACT_DIR / "optuna"
store_dir.mkdir(parents=True, exist_ok=True)
study = optuna.create_study(
    study_name=f"pathsnap_{args.model}", direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=10),
    storage=f"sqlite:///{(store_dir / f'{args.model}.db').as_posix()}", load_if_exists=True)
t0 = time.time()
study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)
elapsed = time.time() - t0

out = config.REPORT_DIR / "experiments"
df = study.trials_dataframe()
df.to_csv(out / f"optuna_{args.model}_trials.csv", index=False)
best = {"model": args.model, "best_value_cv_top1": study.best_value, "best_params": study.best_params,
        "best_trial": study.best_trial.number, "n_trials": len(study.trials), "optimisation_seconds_this_run": elapsed,
        "best_user_attrs": study.best_trial.user_attrs, "sampler": "TPE(seed=42, n_startup_trials=10)",
        "objective": "mean GroupKFold(5) observation-level top-1 accuracy (train partition)"}
(out / f"optuna_{args.model}_best.json").write_text(json.dumps(best, indent=2))
print(json.dumps(best, indent=2))

# plots from the actual study
config.FIG_DIR.mkdir(parents=True, exist_ok=True)
vals = df[df.state == "COMPLETE"].value.to_numpy()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(vals)), vals, "o", alpha=.6, label="trial")
ax.plot(np.arange(len(vals)), np.maximum.accumulate(vals), "r-", label="best so far")
ax.set(xlabel="trial", ylabel="CV top-1 accuracy", title=f"Optuna TPE optimisation history ({args.model})"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(config.FIG_DIR / f"optuna_{args.model}_history.png", dpi=120); plt.close(fig)
try:
    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
    ax.set(title=f"Hyperparameter importance (fANOVA, {args.model})", xlabel="importance")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / f"optuna_{args.model}_importance.png", dpi=120); plt.close(fig)
except Exception as e:  # importance needs >1 distinct value per param
    print("importance plot skipped:", e)
params = [c for c in df.columns if c.startswith("params_")]
fig, axs = plt.subplots(2, (len(params) + 1) // 2, figsize=(4 * ((len(params) + 1) // 2), 7))
for a, c in zip(axs.ravel(), params):
    a.scatter(df[c], df.value, c=df.number, cmap="viridis", s=18)
    a.set(xlabel=c.replace("params_", ""), ylabel="CV top-1")
    if c.replace("params_", "") in {"learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "reg_alpha", "min_child_weight", "l2_regularization", "max_leaf_nodes", "min_samples_leaf"}:
        a.set_xscale("log")
fig.suptitle(f"Optuna slice plot ({args.model}); colour = trial number"); fig.tight_layout()
fig.savefig(config.FIG_DIR / f"optuna_{args.model}_slice.png", dpi=110); plt.close(fig)
