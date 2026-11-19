"""Inference-time trip data validation (io.check_trip_for_inference) and how the API reports it.

Malformed trip files used to reach the pipeline unchecked: they surfaced as opaque 500s
(IndexError / ValueError), or ran silently on corrupted geometry (negative node ids wrap
around in NumPy). These tests use synthetic grids and a synthetic partition folder only;
the API tests swap in a distance-based stub classifier (test-only, not a model) so the real
preprocessing -> candidates -> HMM path runs without the trained bundle.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import grid_network, straight_track
from pathsnap import config, io
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.hmm import HMMParams
from pathsnap.pipeline import MapMatcher


def _trip(track=None, nodes=None, arcs=None):
    g_nodes, g_arcs = grid_network()
    return io.Trip("t1", "p",
                   (straight_track() if track is None else track).reset_index(drop=True),
                   g_nodes if nodes is None else nodes, g_arcs if arcs is None else arcs)


# ------------------------------------------------------------------ 1. pre-flight check
def test_clean_trip_passes_without_warnings():
    assert io.check_trip_for_inference(_trip()) == []


@pytest.mark.parametrize("case, match", [
    ("empty_track", "no GPS observations"),
    ("all_fixes_invalid", "none of the"),
    ("nan_fixes", "none of the"),
    ("empty_nodes", ".nodes file is empty"),
    ("empty_arcs", ".arcs file is empty"),
    ("arc_node_too_large", "outside 0..24"),
    ("arc_node_negative", "outside 0..24"),      # NumPy would silently wrap -1 to the last node
    ("nan_node_used_by_arc", "missing or out-of-range coordinates"),
    ("inf_timestamp", "infinite timestamp"),     # kept by clean_track; inf is not JSON-serialisable
])
def test_unusable_trips_raise_trip_data_error(case, match):
    nodes, arcs = grid_network()
    tr = straight_track()
    bad = {
        "empty_track": dict(track=tr.iloc[:0]),
        "all_fixes_invalid": dict(track=tr.assign(lon=0.0, lat=0.0)),
        "nan_fixes": dict(track=tr.assign(t=np.nan)),
        "empty_nodes": dict(nodes=np.zeros((0, 2)), arcs=np.zeros((0, 2), np.int64)),
        "empty_arcs": dict(arcs=np.zeros((0, 2), np.int64)),
        "arc_node_too_large": dict(arcs=np.r_[arcs, [[0, 999]]]),
        "arc_node_negative": dict(arcs=np.r_[arcs, [[-1, 3]]]),
        "nan_node_used_by_arc": dict(nodes=np.r_[nodes[:-1], [[np.nan, np.nan]]]),
        "inf_timestamp": dict(track=tr.assign(t=np.r_[tr.t.to_numpy()[:-1], np.inf])),
    }[case]
    with pytest.raises(io.TripDataError, match=match) as ei:
        io.check_trip_for_inference(_trip(**bad))
    assert "t1" in str(ei.value) and isinstance(ei.value, ValueError)


def test_unused_bad_node_is_tolerated():
    nodes, arcs = grid_network()
    nodes = np.r_[nodes, [[np.nan, np.nan]]]          # extra node that no arc references
    assert io.check_trip_for_inference(_trip(nodes=nodes, arcs=arcs)) == []


def test_repairable_problems_become_warnings():
    nodes, arcs = grid_network()
    tr = straight_track()
    tr = pd.concat([tr, pd.DataFrame({"lon": [0.0, np.nan], "lat": [0.0, 50.0], "t": [100.0, 101.0]}),
                    tr.iloc[[3]]], ignore_index=True)                       # 2 invalid fixes + 1 duplicate t
    w = io.check_trip_for_inference(_trip(track=tr, arcs=np.r_[arcs, [[4, 4]]]))
    text = " ".join(w)
    assert len(w) == 3
    assert f"2 of {len(tr)} GPS observation(s)" in text
    assert "1 GPS observation(s) with a duplicate timestamp" in text
    assert "1 self-loop arc(s)" in text


def test_single_valid_fix_is_a_warning_not_an_error():
    w = io.check_trip_for_inference(_trip(track=straight_track().iloc[:1]))
    assert any("Only one valid GPS observation" in m for m in w)


# ------------------------------------------------------------------ 2. file parsing
def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text, ncols, dtype, match", [
    ("8.0 50.0 0\n8.1 abc 1\n", 3, np.float64, "non-numeric"),
    ("8.0 50.0\n8.1 50.1\n", 3, np.float64, "expected 3 columns, found 2"),
    ("8.0 50.0 0\n8.1 50.1 1 99\n", 3, np.float64, "unreadable"),            # ragged rows
    ("0 1\n1.5 2\n", 2, np.int64, "integer ids only"),                         # would truncate to 1
    ("0 1\n1\n", 2, np.int64, "integer ids only"),                             # missing field -> NaN
    ("0 1\n1 99999999999999999999999\n", 2, np.int64, "integer ids only"),     # beyond int64
])
def test_malformed_files_raise_trip_data_error(tmp_path, text, ncols, dtype, match):
    p = _write(tmp_path / "x.track", text)
    with pytest.raises(io.TripDataError, match=match):
        io._read_table(p, ncols, dtype)


def test_whitespace_only_file_reads_as_empty(tmp_path):
    assert io._read_table(_write(tmp_path / "x.track", "\n  \n"), 3, np.float64).shape == (0, 3)


# ------------------------------------------------------------------ 3. API behaviour
class _DistanceStub:
    """Test-only scorer: probability decreases with GPS-to-arc distance (feature 'dist')."""

    def predict_proba(self, X):
        p = np.exp(-X[:, FEATURE_COLUMNS.index("dist")] / 10.0)
        return np.c_[1 - p, p]


def _write_trip(root, tid, track, nodes, arcs):
    d = root / "synthetic" / tid
    d.mkdir(parents=True)
    np.savetxt(d / f"{tid}.track", track[["lon", "lat", "t"]].to_numpy(), fmt="%.9f")
    np.savetxt(d / f"{tid}.nodes", nodes, fmt="%.9f")
    np.savetxt(d / f"{tid}.arcs", arcs, fmt="%d")
    return d


@pytest.fixture
def api(tmp_path, monkeypatch):
    from api import main
    from api.service import EvaluationService, PredictionService
    ps = PredictionService.__new__(PredictionService)           # no bundle needed
    ps.matcher = MapMatcher(_DistanceStub(), FEATURE_COLUMNS, HMMParams())
    ps.model_info = {"name": "distance stub (test only)", "version": "test"}
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setitem(main.STATE, "prediction", ps)
    monkeypatch.setitem(main.STATE, "partition", "synthetic")
    monkeypatch.setitem(main.STATE, "evaluation", EvaluationService())
    nodes, arcs = grid_network()
    tr = straight_track()
    _write_trip(tmp_path, "00000001", tr, nodes, arcs)                                   # valid
    _write_trip(tmp_path, "00000002", tr, nodes, np.r_[arcs, [[0, 999]]])                # bad node ref
    noisy = pd.concat([tr, pd.DataFrame({"lon": [0.0], "lat": [0.0], "t": [999.0]})], ignore_index=True)
    _write_trip(tmp_path, "00000003", noisy, nodes, arcs)                                # 1 invalid fix
    _write_trip(tmp_path, "00000004", tr, nodes + 1.0, arcs)                             # network ~100 km away
    d = _write_trip(tmp_path, "00000005", tr, nodes, arcs)
    (d / "00000005.track").write_text("lon lat t\n8.0 50.0 0\n", encoding="utf-8")       # corrupt track
    d = _write_trip(tmp_path, "00000006", tr, nodes, arcs)
    (d / "00000006.arcs").unlink()                                                       # missing .arcs
    _write_trip(tmp_path, "00000007", tr.assign(t=np.r_[tr.t.to_numpy()[:-1], np.inf]), nodes, arcs)  # inf time
    return TestClient(main.app), tmp_path                     # no `with`: lifespan (bundle load) not run


def test_api_valid_synthetic_trip_completes(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000001"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["status"] == "completed" and b["predicted_route"]["arc_ids"]
    assert b["metrics"] == {} and any("No reference route" in w for w in b["warnings"])
    assert not any("GPS observation" in w for w in b["warnings"])


def test_api_bad_network_is_422_with_reason(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000002"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str) and "00000002" in detail and "node ids outside" in detail


def test_api_corrupt_track_is_422(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000005"})
    assert r.status_code == 422 and "non-numeric" in r.json()["detail"]


def test_api_missing_file_is_422_without_server_paths(api):
    client, tmp = api
    r = client.post("/evaluation/run", json={"trip_id": "00000006"})
    detail = r.json()["detail"]
    assert r.status_code == 422 and "00000006.arcs" in detail
    assert str(tmp) not in detail and "\\" not in detail


def test_api_reports_dropped_fixes_as_warning(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000003"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["num_observations"] == len(b["observations"]) + 1     # raw count, invalid fix not drawn
    assert any("1 of" in w and "were ignored" in w for w in b["warnings"])


def test_api_network_not_covering_track_warns_instead_of_crashing(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000004"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["predicted_route"]["arc_ids"] == [] and b["road_network"]["segments"] == []
    assert any("No road segment could be matched" in w for w in b["warnings"])


def test_api_trip_list_survives_one_corrupt_trip(api):
    client, _ = api
    r = client.get("/evaluation/trips")
    assert r.status_code == 200, r.text
    trips = {t["trip_id"]: t for t in r.json()["trips"]}
    assert len(trips) == 7
    assert trips["00000005"] == {"trip_id": "00000005", "num_observations": None, "duration_s": None,
                                 "label": "00000005 (unreadable trip data)"}
    assert trips["00000001"]["num_observations"] == len(straight_track())
    n = len(straight_track())
    assert trips["00000001"]["duration_s"] == n - 1                      # unchanged for valid trips
    assert trips["00000007"]["duration_s"] == n - 2                      # inf timestamp ignored, list still JSON


def test_api_infinite_timestamp_is_422_not_500(api):
    client, _ = api
    r = client.post("/evaluation/run", json={"trip_id": "00000007"})
    assert r.status_code == 422 and "infinite timestamp" in r.json()["detail"]
