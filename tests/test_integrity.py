"""Data-integrity tests: group separation, partition isolation, reference-free features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pathsnap import config, io
from pathsnap.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CACHE = config.CACHE_DIR / "features_train.parquet"
EXP = config.REPORT_DIR / "experiments"


@pytest.mark.skipif(not TRAIN_CACHE.exists(), reason="training cache not built")
def test_groupkfold_never_shares_trips():
    tr = pd.read_parquet(TRAIN_CACHE, columns=["trip_id", "reliable"])
    tr = tr[tr.reliable == 1].reset_index(drop=True)
    g = tr.trip_id.to_numpy()
    seen_val = set()
    for tri, vai in GroupKFold(n_splits=5).split(np.zeros(len(g)), groups=g):
        a, b = set(g[tri]), set(g[vai])
        assert a.isdisjoint(b)
        seen_val |= b
    assert seen_val == set(g)   # every trip validated exactly once


def test_partitions_are_disjoint_and_cache_has_no_evaluator_data():
    tr, st = set(io.list_trips("train")), set(io.list_trips("student_test"))
    assert tr.isdisjoint(st)
    if TRAIN_CACHE.exists():
        assert set(pd.read_parquet(TRAIN_CACHE, columns=["trip_id"]).trip_id) <= tr
    ev = set(io.list_trips(config.EVAL_PARTITION))
    assert ev.isdisjoint(tr)


def test_training_and_tuning_scripts_never_touch_evaluator_partition():
    for s in ["02_build_dataset.py", "03_train_models.py", "04_bayes_opt.py", "05_finalize_model.py", "06_tune_decoder.py"]:
        src = (ROOT / "scripts" / s).read_text()
        assert "EVAL_PARTITION" not in src and "evaluator_test" not in src, s


def test_bayes_opt_and_decoder_tuning_use_train_only():
    bo = (ROOT / "scripts" / "04_bayes_opt.py").read_text()
    assert "features_student_test" not in bo and "student_test" not in bo
    dec = (ROOT / "scripts" / "06_tune_decoder.py").read_text()
    # student-test is only read AFTER the TRAIN-only selection has been written
    assert dec.index('hmm_params.json").write_text') < dec.index("features_student_test")
    assert "oof_train_final" in dec


MANIFEST = config.ARTIFACT_DIR / "final" / "manifest.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="final bundle not built")
def test_optuna_trials_recorded_for_final_model():
    name = json.loads(MANIFEST.read_text())["model_name"]
    t = pd.read_csv(EXP / f"optuna_{name}_trials.csv")
    best = json.loads((EXP / f"optuna_{name}_best.json").read_text())
    done = t[t.state == "COMPLETE"]
    assert len(done) >= 10 and np.isclose(done.value.max(), best["best_value_cv_top1"])
    assert json.loads(MANIFEST.read_text())["hyperparameters"].items() >= best["best_params"].items()


@pytest.mark.skipif(not MANIFEST.exists(), reason="final bundle not built")
def test_final_model_is_cv_best_not_student_test_best():
    """The final algorithm must be the CV top-1 winner on TRAIN (selection never uses student-test)."""
    cmp = pd.read_csv(EXP / "model_comparison.csv")
    cmp = cmp[cmp.model != "nearest_road"]
    assert json.loads(MANIFEST.read_text())["model_name"] == cmp.loc[cmp.cv_top1_accuracy.idxmax(), "model"]


@pytest.mark.skipif(not TRAIN_CACHE.exists(), reason="training cache not built")
def test_features_identical_without_reference_route(tmp_path, monkeypatch):
    """Recompute features for a train trip whose .route is absent: identical to the cached training features."""
    import shutil
    from pathsnap.features import compute_features
    from pathsnap.network import RoadNetwork
    from pathsnap.preprocess import preprocess_track
    tid = io.list_trips("train")[2]
    src = io.partition_dir("train") / tid
    dst = tmp_path / "x" / tid
    dst.mkdir(parents=True)
    for ext in ["track", "nodes", "arcs"]:
        shutil.copy(src / f"{tid}.{ext}", dst / f"{tid}.{ext}")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    trip = io.load_trip("x", tid)
    df, proj, _ = preprocess_track(trip.track)
    f = compute_features(df, RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y))
    cached = pd.read_parquet(TRAIN_CACHE, filters=[("trip_id", "=", tid)])
    assert len(f) == len(cached)
    assert np.allclose(f[FEATURE_COLUMNS].to_numpy(float), cached[FEATURE_COLUMNS].to_numpy(float), equal_nan=True)
