"""Evaluator-partition safety: test guard, fail-fast configuration, route-free inference.

Uses only synthetic folders and student_test trips. Never reads evaluator data: the
conftest guard raises before any evaluator path could be opened.
"""
import builtins
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import EvaluatorAccessError
from pathsnap import config, io

ROOT = Path(__file__).resolve().parent.parent
REAL_EVAL = ROOT / "evaluator_test"
ST = io.list_trips(config.STUDENT_TEST_PARTITION)
HAVE_BUNDLE = (config.ARTIFACT_DIR / "final" / "manifest.json").exists()


# ------------------------------------------------------------------ 1. test-suite guard
def test_env_and_config_neutralised():
    assert "PATHSNAP_EVAL_DIR" not in os.environ and "PATHSNAP_EVAL_PARTITION" not in os.environ
    assert config.EVAL_DIR_OVERRIDE is None
    assert config.EVAL_PARTITION != "evaluator_test"


def test_guard_blocks_every_evaluator_path(monkeypatch):
    # even if a test re-enables the evaluator configuration, nothing can be read
    monkeypatch.setattr(config, "EVAL_PARTITION", "evaluator_test")
    monkeypatch.setattr(config, "EVAL_DIR_OVERRIDE", str(REAL_EVAL))
    for call in (lambda: io.list_trips("evaluator_test"),
                 lambda: io.load_track("evaluator_test", "00000015"),
                 lambda: io.load_trip("evaluator_test", "00000015"),
                 lambda: io.has_reference_route("evaluator_test", "00000015"),
                 lambda: io.load_reference_route("evaluator_test", "00000015")):
        with pytest.raises(EvaluatorAccessError):
            call()
    # a different partition name pointing into the evaluator folder is blocked as well
    monkeypatch.setattr(config, "DATA_ROOT", ROOT)
    with pytest.raises(EvaluatorAccessError):
        io.load_reference_route("evaluator_test", "00000015")


# ------------------------------------------------------------------ 2. no silent fallback
def _copy_trip_without_route(dst_root: Path, tid: str, exts=("track", "nodes", "arcs")):
    src = io.partition_dir(config.STUDENT_TEST_PARTITION) / tid
    (dst_root / tid).mkdir(parents=True)
    for e in exts:
        shutil.copy(src / f"{tid}.{e}", dst_root / tid / f"{tid}.{e}")


@pytest.mark.parametrize("case", ["missing", "empty", "incomplete"])
def test_explicit_eval_dir_invalid_raises(tmp_path, monkeypatch, case):
    from api.service import EvaluatorConfigError, resolve_eval_partition
    d = tmp_path / "evaldir"
    if case == "empty":
        d.mkdir()
    if case == "incomplete":
        _copy_trip_without_route(d, ST[0], exts=("track", "nodes"))   # .arcs missing
    monkeypatch.setattr(config, "EVAL_DIR_OVERRIDE", str(d))
    with pytest.raises(EvaluatorConfigError):
        resolve_eval_partition()


@pytest.mark.skipif(not ST, reason="dataset not present")
def test_explicit_valid_eval_dir_is_used(tmp_path, monkeypatch):
    from api.service import resolve_eval_partition
    d = tmp_path / "evaldir"
    _copy_trip_without_route(d, ST[0])
    monkeypatch.setattr(config, "EVAL_DIR_OVERRIDE", str(d))
    assert resolve_eval_partition() == (config.EVAL_PARTITION, False)


def test_without_override_fallback_is_reported():
    from api.service import resolve_eval_partition
    assert resolve_eval_partition() == (config.STUDENT_TEST_PARTITION, True)


@pytest.mark.skipif(not HAVE_BUNDLE, reason="final bundle not built")
def test_api_refuses_to_start_with_bad_eval_dir(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api.service import EvaluatorConfigError
    monkeypatch.setattr(config, "EVAL_DIR_OVERRIDE", str(tmp_path / "does_not_exist"))
    with pytest.raises(EvaluatorConfigError):
        with TestClient(app):
            pass


# ------------------------------------------------------------------ 3. route-free inference
class _RouteSpy:
    """Records every file opened through open / pandas.read_csv / numpy.loadtxt."""

    def __init__(self, monkeypatch):
        self.paths = []
        o_open, o_csv, o_loadtxt = builtins.open, pd.read_csv, np.loadtxt

        def rec(p):
            if isinstance(p, (str, os.PathLike)):
                self.paths.append(str(p))

        monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (rec(f), o_open(f, *a, **k))[1])
        monkeypatch.setattr(pd, "read_csv", lambda f, *a, **k: (rec(f), o_csv(f, *a, **k))[1])
        monkeypatch.setattr(np, "loadtxt", lambda f, *a, **k: (rec(f), o_loadtxt(f, *a, **k))[1])

    @property
    def route_files(self):
        return [p for p in self.paths if p.lower().endswith(".route")]


@pytest.mark.skipif(not (HAVE_BUNDLE and ST), reason="final bundle / dataset not present")
def test_inference_never_opens_route_files(monkeypatch):
    from api.service import EvaluationService, PredictionService
    ps = PredictionService()
    tid = min(ST, key=lambda t: (io.partition_dir(config.STUDENT_TEST_PARTITION) / t / f"{t}.track").stat().st_size)
    spy = _RouteSpy(monkeypatch)
    calls = []
    o_load = io.load_reference_route
    monkeypatch.setattr(io, "load_reference_route", lambda *a: (calls.append(a), o_load(*a))[1])
    result, payload = ps.predict(config.STUDENT_TEST_PARTITION, tid)       # prediction only
    assert any(p.endswith(f"{tid}.track") for p in spy.paths)              # spy sees real reads
    assert spy.route_files == [] and calls == []                           # ... but no .route
    assert payload["predicted_route"]["arc_ids"]
    # positive control: the server-side evaluation step DOES read the .route, and the spy catches it
    EvaluationService.score(config.STUDENT_TEST_PARTITION, tid, result)
    assert calls and any(p.endswith(f"{tid}.route") for p in spy.route_files)
