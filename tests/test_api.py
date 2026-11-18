"""API tests against the real model bundle and the student-test trips (development fallback partition)."""
import json

import pytest
from fastapi.testclient import TestClient

from pathsnap import config, io

HAVE_BUNDLE = (config.ARTIFACT_DIR / "final" / "manifest.json").exists()
pytestmark = pytest.mark.skipif(not (HAVE_BUNDLE and io.list_trips(config.STUDENT_TEST_PARTITION)),
                                reason="final bundle / dataset not present")


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def run(client):
    trips = client.get("/evaluation/trips").json()["trips"]
    small = min(trips, key=lambda t: t["num_observations"])["trip_id"]
    r = client.post("/evaluation/run", json={"trip_id": small})
    assert r.status_code == 200, r.text
    return small, r


def test_health(client):
    b = client.get("/health").json()
    assert b["status"] == "ok" and b["model_loaded"] is True and b["hmm_enabled"] is True
    assert b["num_evaluation_trips"] >= 1 and isinstance(b["model_name"], str) and b["version"]


def test_trips_ids_only(client):
    b = client.get("/evaluation/trips").json()
    assert len(b["trips"]) == len(io.list_trips(b["partition"]))
    for t in b["trips"]:
        assert set(t) <= {"trip_id", "num_observations", "duration_s", "label"}
        assert isinstance(t["trip_id"], str) and len(t["trip_id"]) == 8  # zero-padded id kept


def test_run_prediction_shape(run):
    tid, r = run
    b = r.json()
    assert b["trip_id"] == tid and b["status"] == "completed" and b["evaluation_id"]
    raw = io.load_track(b["partition"], tid)
    assert b["num_observations"] == len(raw) == len(b["observations"])
    o = b["observations"][0]
    assert abs(o["lon"] - raw.lon.iloc[0]) < 1e-9 and abs(o["lat"] - raw.lat.iloc[0]) < 1e-9  # [lon, lat] order
    pr = b["predicted_route"]
    assert len(pr["arc_ids"]) == len(pr["segments"]) > 0
    for s in pr["segments"]:
        assert len(s["coordinates"]) == 2 and all(len(c) == 2 for c in s["coordinates"])
    # segment geometry equals the raw network geometry of that arc
    nodes, arcs = io.load_network(b["partition"], tid)
    s = pr["segments"][0]
    assert s["coordinates"][0] == nodes[arcs[s["arc_id"]][0]].tolist()
    assert len(b["road_network"]["segments"]) > 0
    assert all(isinstance(k, str) and k.endswith("_ms") for k in b["timing"])
    assert b["hmm"]["applied"] is True


def test_run_metrics_in_unit_range(run):
    b = run[1].json()
    for key in ["top1_accuracy", "macro_f1", "precision", "recall", "route_continuity", "candidate_miss_rate",
                "sequence_accuracy", "network_reachable_continuity"]:
        assert 0.0 <= b["metrics"][key] <= 1.0, key
    assert b["metrics"]["log_loss"] > 0
    assert b["classifier_only_metrics"]["top1_accuracy"] is not None
    assert b["nearest_road_metrics"]["log_loss"] is None


def test_reference_route_never_exposed(run):
    tid, r = run
    b = r.json()
    assert "reference_route" not in b
    text = r.text
    for banned in ["reference_route", "label_arc", "ref_route", "route_pos"]:
        assert banned not in text
    # matched_arc_id values are predictions: they come from the HMM output and are not all equal to the label
    ref = set(io.load_reference_route(b["partition"], tid).tolist())
    pred = {o["matched_arc_id"] for o in b["observations"]}
    assert pred - ref or b["metrics"]["top1_accuracy"] >= 0.99  # sanity: not a copy of the reference


def test_get_stored_and_map(client, run):
    b = run[1].json()
    g = client.get(f"/evaluation/{b['evaluation_id']}")
    assert g.status_code == 200 and g.json() == b
    m = client.get(b["folium_map_url"])
    assert m.status_code == 200 and "leaflet" in m.text.lower()
    assert "Reference route" not in m.text


def test_errors_use_fastapi_detail(client):
    r = client.post("/evaluation/run", json={"trip_id": "99999999"})
    assert r.status_code == 404 and "Unknown trip_id" in r.json()["detail"]
    r = client.post("/evaluation/run", json={"trip_id": "../etc"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)
    assert client.post("/evaluation/run", json={}).status_code == 422
    assert client.post("/evaluation/run", content="not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.get("/evaluation/doesnotexist").status_code == 404
    assert client.get("/evaluation/doesnotexist/map").status_code == 404


def test_cors_preflight(client):
    r = client.options("/evaluation/run", headers={"Origin": "http://192.168.1.35:5173", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers.get("access-control-allow-origin") in ("*", "http://192.168.1.35:5173")


def test_prediction_independent_of_reference(tmp_path, monkeypatch):
    """Copy a trip WITHOUT its .route file: prediction must be identical, metrics absent."""
    import shutil
    from api import service
    tid = io.list_trips(config.STUDENT_TEST_PARTITION)[0]
    src = io.partition_dir(config.STUDENT_TEST_PARTITION) / tid
    dst = tmp_path / "noref" / tid
    dst.mkdir(parents=True)
    for ext in ["track", "nodes", "arcs"]:
        shutil.copy(src / f"{tid}.{ext}", dst / f"{tid}.{ext}")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    ps = service.PredictionService()
    res_noref, p_noref = ps.predict("noref", tid)
    assert service.EvaluationService.score("noref", tid, res_noref) == (None, None, None)
    monkeypatch.undo()
    _, p_ref = ps.predict(config.STUDENT_TEST_PARTITION, tid)
    assert json.dumps(p_noref, sort_keys=True) == json.dumps(p_ref, sort_keys=True)


def test_model_comparison_matches_recorded_train_cv(client):
    """GET /models/comparison serves exactly the recorded TRAIN-CV results; no student/evaluator metrics."""
    import pandas as pd
    r = client.get("/models/comparison")
    assert r.status_code == 200
    b = r.json()
    assert "TRAIN" in b["evaluation"] and b["not_evaluator_results"] is True
    assert len(b["models"]) == 8 and b["baseline"]["key"] == "nearest_road"
    csv = pd.read_csv(config.REPORT_DIR / "experiments" / "model_comparison.csv").set_index("model")
    for m in b["models"]:
        assert m["cv_top1_accuracy"] == pytest.approx(csv.loc[m["key"], "cv_top1_accuracy"])
        assert m["cv_macro_f1"] == pytest.approx(csv.loc[m["key"], "cv_cand_macro_f1"])
    selected = [m["key"] for m in b["models"] if m["selected"]]
    manifest = json.loads((config.ARTIFACT_DIR / "final" / "manifest.json").read_text())
    assert selected == [manifest["model_name"]] == [b["selected_model"]]
    assert not any(k.startswith("st_") for k in json.dumps(b).replace('"', " ").split())
