"""Phase 10: performance measurements of the final system on the student-test trips.

Measures model-bundle load time, per-trip pipeline stage times, full API response time
(in-process TestClient, i.e. excluding network transfer) and process memory (RSS).
Output: reports/experiments/benchmark.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from pathsnap import config, io

proc = psutil.Process()
rss = lambda: proc.memory_info().rss / 2**20
out = {"rss_start_mb": rss()}

t0 = time.perf_counter()
from pathsnap.pipeline import MapMatcher
m = MapMatcher.load()
out["model_load_ms"] = 1000 * (time.perf_counter() - t0)
out["rss_after_load_mb"] = rss()

trips = io.list_trips(config.STUDENT_TEST_PARTITION)
per = []
peak = rss()
for tid in trips:
    trip = io.load_trip(config.STUDENT_TEST_PARTITION, tid)
    t0 = time.perf_counter()
    r = m.match_trip(trip)
    wall = 1000 * (time.perf_counter() - t0)
    peak = max(peak, rss())
    per.append({"trip_id": tid, "n_obs": len(r["df"]), "n_candidates": len(r["feats"]), "wall_ms": wall,
                **{k.replace("_s", "_ms"): 1000 * v for k, v in r["timings"].items()}})
out["pipeline_per_trip"] = per
out["pipeline_mean_ms"] = float(np.mean([p["wall_ms"] for p in per]))
out["pipeline_max_ms"] = float(np.max([p["wall_ms"] for p in per]))
out["hmm_mean_ms"] = float(np.mean([p["viterbi_ms"] for p in per]))
out["ms_per_observation"] = float(np.sum([p["wall_ms"] for p in per]) / np.sum([p["n_obs"] for p in per]))

from fastapi.testclient import TestClient
from api.main import app
api = []
with TestClient(app) as c:
    for tid in trips:
        t0 = time.perf_counter()
        r = c.post("/evaluation/run", json={"trip_id": tid})
        api.append({"trip_id": tid, "status": r.status_code, "response_ms": 1000 * (time.perf_counter() - t0),
                    "payload_kb": len(r.content) / 1024, "server_inference_ms": r.json()["timing"]["inference_ms"],
                    "server_evaluation_ms": r.json()["timing"]["evaluation_ms"]})
        peak = max(peak, rss())
out["api_per_trip"] = api
out["api_mean_response_ms"] = float(np.mean([a["response_ms"] for a in api]))
out["api_max_response_ms"] = float(np.max([a["response_ms"] for a in api]))
out["api_mean_payload_kb"] = float(np.mean([a["payload_kb"] for a in api]))
out["rss_peak_mb"] = peak
out["cpu"] = {"logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}
(config.REPORT_DIR / "experiments" / "benchmark.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))
