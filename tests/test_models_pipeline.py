import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pathsnap import config, io
from pathsnap.models import MODEL_SPECS, make_model

ROOT = Path(__file__).resolve().parent.parent
FINAL = config.ARTIFACT_DIR / "final"
HAVE_BUNDLE = (FINAL / "manifest.json").exists()
REAL = io.list_trips(config.STUDENT_TEST_PARTITION)


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_trains_and_outputs_probabilities(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    fast = {"n_estimators": 20} if name in {"random_forest", "extra_trees", "lightgbm", "xgboost"} else {}
    fast = {"max_iter": 20} if name == "hist_gb" else fast
    m = make_model(name, fast)
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (600, 2) and np.allclose(p.sum(1), 1) and ((p >= 0) & (p <= 1)).all()
    assert ((p[:, 1] > 0.5) == y).mean() > 0.8


@pytest.mark.skipif(not (HAVE_BUNDLE and REAL), reason="final bundle / dataset not present")
def test_bundle_reloads_in_fresh_process_and_predicts():
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from pathsnap.pipeline import MapMatcher; from pathsnap import io;"
        "m = MapMatcher.load(); r = m.match_trip(io.load_trip('student_test', '%s'));"
        "import json; print(json.dumps({'n': len(r['df']), 'matched': int((r['hmm_arc_local']>=0).sum()), 'route': len(r['route_local'])}))"
    ) % (ROOT, REAL[0])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["matched"] == res["n"] and res["route"] > 0


@pytest.mark.skipif(not (HAVE_BUNDLE and REAL), reason="final bundle / dataset not present")
def test_inference_is_deterministic_and_route_continuous():
    from pathsnap.pipeline import MapMatcher
    m = MapMatcher.load()
    trip = io.load_trip("student_test", REAL[-1])
    r1, r2 = m.match_trip(trip), m.match_trip(trip)
    assert (r1["hmm_arc_local"] == r2["hmm_arc_local"]).all() and r1["route_local"] == r2["route_local"]
    net, route = r1["net"], r1["route_local"]
    breaks = sum(net.arc_to[a] != net.arc_from[b] for a, b in zip(route[:-1], route[1:]))
    assert breaks == r1["route_unfilled_gaps"]


def test_pipeline_does_not_import_reference_code():
    """Prediction modules must not import the label module or call the reference-route loader."""
    import ast
    for mod in ["pipeline.py", "features.py", "preprocess.py", "network.py", "hmm.py"]:
        tree = ast.parse((ROOT / "pathsnap" / mod).read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                 {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not any("labels" in m for m in imported), mod
        assert "load_reference_route" not in called and "observation_labels" not in called, mod
