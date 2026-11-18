"""PathSnap Laptop-1 FastAPI backend (implements docs/API_CONTRACT.md).

Run (LAN-accessible):  .venv\\Scripts\\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs:                  http://<laptop1-ip>:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from pathsnap import __version__, config, io

from .schemas import EvaluationResult, HealthResponse, RunRequest, TripListResponse, TripSummary
from .service import EvaluationService, PredictionService, new_evaluation_id, now_iso, resolve_eval_partition

LOG_DIR = config.PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=os.environ.get("PATHSNAP_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")],
)
log = logging.getLogger("pathsnap.api")

STATE: dict = {"started": time.time(), "prediction": None, "evaluation": EvaluationService(), "load_error": None,
               "partition": config.STUDENT_TEST_PARTITION, "fallback": True, "model_load_ms": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        part, fallback = resolve_eval_partition()
    except Exception as e:  # explicit evaluator folder configured but unusable -> refuse to start
        log.critical("REFUSING TO START: %s", e)
        raise
    STATE["partition"], STATE["fallback"] = part, fallback
    if fallback:
        log.warning("Evaluator partition '%s' not found (DATA_ROOT=%s, PATHSNAP_EVAL_DIR=%s) -> serving '%s' "
                    "(development mode).", config.EVAL_PARTITION, config.DATA_ROOT, config.EVAL_DIR_OVERRIDE, part)
    log.info("Evaluation partition '%s': %d trips at %s", part, len(io.list_trips(part)), io.partition_dir(part))
    t0 = time.perf_counter()
    try:
        STATE["prediction"] = PredictionService()
        STATE["model_load_ms"] = 1000 * (time.perf_counter() - t0)
        mi = STATE["prediction"].model_info
        log.info("Model loaded in %.0f ms: %s | HMM params %s", STATE["model_load_ms"], mi["name"],
                 STATE["prediction"].matcher.hmm_params.to_dict())
    except Exception as e:  # keep serving /health so the problem is visible remotely
        STATE["load_error"] = repr(e)
        log.exception("Failed to load model artifacts: %s", e)
    log.info("CORS allowed origins: %s", ORIGINS)
    yield
    log.info("Shutting down")


app = FastAPI(title="PathSnap Map-Matching API (Laptop 1)", version=__version__, lifespan=lifespan,
              description="Candidate generation -> classifier -> HMM/Viterbi -> route reconstruction. "
                          "Reference routes stay server-side; only aggregate metrics are returned.")

ORIGINS = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "-"
    response = await call_next(request)
    log.info("%s %s %s -> %d (%.1f ms)", client, request.method, request.url.path, response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.exception_handler(Exception)
async def unhandled_exc(_: Request, exc: Exception):
    log.error("Internal error: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}: {exc}"})


def _require_model() -> PredictionService:
    ps = STATE["prediction"]
    if ps is None:
        raise HTTPException(503, f"Model not loaded on Laptop 1: {STATE['load_error']}")
    return ps


# ----------------------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    ps = STATE["prediction"]
    return HealthResponse(
        status="ok" if ps is not None else "degraded", model_loaded=ps is not None,
        model_name=ps.model_info["name"] if ps else None, hmm_enabled=ps is not None, version=__version__,
        num_evaluation_trips=len(io.list_trips(STATE["partition"])), evaluation_partition=STATE["partition"],
        evaluation_partition_is_fallback=STATE["fallback"], uptime_s=round(time.time() - STATE["started"], 1))


@app.get("/evaluation/trips", response_model=TripListResponse, tags=["evaluation"])
def list_evaluation_trips():
    """Trip ids plus harmless metadata (observation count, duration). Never routes or labels."""
    part = STATE["partition"]
    trips = []
    for tid in io.list_trips(part):
        tr = io.load_track(part, tid)
        trips.append(TripSummary(trip_id=tid, num_observations=len(tr),
                                 duration_s=float(tr.t.iloc[-1] - tr.t.iloc[0]) if len(tr) else 0.0))
    return TripListResponse(partition=part, trips=trips)


@app.post("/evaluation/run", response_model=EvaluationResult, tags=["evaluation"])
def run_evaluation(req: RunRequest):
    ps = _require_model()
    part = STATE["partition"]
    if req.trip_id not in io.list_trips(part):
        raise HTTPException(404, f"Unknown trip_id '{req.trip_id}': not one of the {part} trips")
    eid = new_evaluation_id()
    log.info("RUN id=%s trip=%s partition=%s", eid, req.trip_id, part)
    # 1) prediction: GPS + road network only
    t0 = time.perf_counter()
    result, payload = ps.predict(part, req.trip_id)
    inference_ms = 1000 * (time.perf_counter() - t0)
    # 2) evaluation: reference route loaded server-side, after prediction
    t1 = time.perf_counter()
    m_hmm, m_clf, m_near = EvaluationService.score(part, req.trip_id, result)
    evaluation_ms = 1000 * (time.perf_counter() - t1)

    tm = result["timings"]
    timing = {"inference_ms": inference_ms,
              "preprocessing_ms": 1000 * tm["preprocess_network_s"],
              "candidate_generation_ms": 1000 * tm["candidates_features_s"],
              "classifier_ms": 1000 * tm["classifier_s"],
              "hmm_ms": 1000 * tm["viterbi_s"],
              "route_reconstruction_ms": 1000 * tm["route_reconstruction_s"],
              "evaluation_ms": evaluation_ms,
              "total_ms": inference_ms + evaluation_ms}
    warnings = []
    if m_hmm is None:
        warnings.append("No reference route on Laptop 1 for this trip: metrics unavailable.")
    if result["route_unfilled_gaps"]:
        warnings.append(f"{result['route_unfilled_gaps']} route gap(s) could not be connected through the road network.")
    hmm_msg = None
    if result["n_breaks"]:
        hmm_msg = f"Viterbi chain restarted {result['n_breaks']} time(s) (long GPS gap or no connected candidates)."
        warnings.append(hmm_msg)
    rec = {"evaluation_id": eid, "trip_id": req.trip_id, "status": "completed", "created_at": now_iso(), "error": None,
           "partition": part, "model": ps.model_info, **payload,
           "metrics": m_hmm or {}, "classifier_only_metrics": m_clf, "nearest_road_metrics": m_near,
           "timing": {k: round(v, 2) for k, v in timing.items()},
           "hmm": {"applied": True, "status": "ok" if not result["n_breaks"] else "ok_with_breaks", "message": hmm_msg},
           "hmm_params": ps.matcher.hmm_params.to_dict(),
           "folium_map_url": f"/evaluation/{eid}/map", "warnings": warnings}
    STATE["evaluation"].save(rec, internal=result)
    log.info("DONE id=%s trip=%s n_obs=%d route_arcs=%d top1=%s macro_f1=%s inference=%.0f ms total=%.0f ms",
             eid, req.trip_id, payload["num_observations"], len(payload["predicted_route"]["arc_ids"]),
             f"{m_hmm['top1_accuracy']:.4f}" if m_hmm else "n/a", f"{m_hmm['macro_f1']:.4f}" if m_hmm else "n/a",
             inference_ms, timing["total_ms"])
    return rec


@app.get("/models/comparison", tags=["models"])
def model_comparison():
    """Recorded TRAIN cross-validation comparison of the 8 candidate classifiers (read-only; no evaluator data)."""
    from .service import load_model_comparison
    try:
        return load_model_comparison()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/evaluation/{evaluation_id}", response_model=EvaluationResult, tags=["evaluation"])
def get_evaluation(evaluation_id: str):
    rec = STATE["evaluation"].get(evaluation_id)
    if rec is None:
        raise HTTPException(404, f"Unknown evaluation_id '{evaluation_id}' (results are kept in memory while the server runs)")
    return rec


@app.get("/evaluation/{evaluation_id}/map", response_class=HTMLResponse, tags=["evaluation"])
def evaluation_map(evaluation_id: str):
    """Folium map of a run: GPS, candidate roads, classifier-only and HMM routes (no reference route)."""
    res = STATE["evaluation"].get_internal(evaluation_id)
    if res is None:
        raise HTTPException(404, f"No map available for evaluation_id '{evaluation_id}' (only the latest runs are kept)")
    from pathsnap.viz import match_map
    rec = STATE["evaluation"].get(evaluation_id)
    return HTMLResponse(match_map(res, ref_route_gid=None, title=f"PathSnap trip {rec['trip_id']}").get_root().render())
