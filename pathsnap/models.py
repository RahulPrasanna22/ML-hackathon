"""Model zoo: candidate scorers from different algorithm families.

Every model is a binary classifier P(candidate is the true road | features); the
per-observation argmax gives the classifier-only match and the probabilities become
HMM emission scores. All expose predict_proba.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

SEED = config.RANDOM_SEED

MODEL_SPECS = {
    "logreg": {
        "family": "Linear / probabilistic",
        "params": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    },
    "gaussian_nb": {
        "family": "Probabilistic (generative)",
        "params": {"var_smoothing": 1e-9},
    },
    "random_forest": {
        "family": "Ensemble (bagging)",
        "params": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": 0.5, "n_jobs": -1},
    },
    "extra_trees": {
        "family": "Ensemble (randomised trees)",
        "params": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": None, "n_jobs": -1},
    },
    "hist_gb": {
        "family": "Boosting (histogram GBDT)",
        "params": {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "early_stopping": False},
    },
    "lightgbm": {
        "family": "Boosting (LightGBM)",
        "params": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                   "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, "n_jobs": -1},
    },
    "xgboost": {
        "family": "Boosting (XGBoost)",
        "params": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
                   "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": -1, "eval_metric": "logloss"},
    },
    "mlp": {
        "family": "Neural network (MLP)",
        "params": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "batch_size": 1024, "learning_rate_init": 1e-3,
                   "max_iter": 40, "early_stopping": True, "n_iter_no_change": 5, "validation_fraction": 0.1},
    },
}


def make_model(name: str, params: dict | None = None):
    p = dict(MODEL_SPECS[name]["params"])
    if params:
        p.update(params)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **p))
    if name == "gaussian_nb":
        return GaussianNB(**p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, **p)
    if name == "extra_trees":
        if p.get("max_samples") is None:
            p.pop("max_samples", None)
        return ExtraTreesClassifier(random_state=SEED, **p)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=SEED, **p)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=SEED, **p)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED, **p))
    raise KeyError(name)
