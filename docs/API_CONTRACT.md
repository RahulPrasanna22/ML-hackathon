# PathSnap API contract (Laptop 1 ⇄ Laptop 2)

**Status:** proposed by Laptop 2 (frontend); **confirmed and implemented by Laptop 1** (`api/main.py`). Laptop 1 adds
only optional extra fields, so `frontend/src/api/validate.ts` / `types.ts` need no change. See
[Laptop 1 implementation notes](#laptop-1-implementation-notes) at the end of this document.
Any change agreed here must be mirrored in `frontend/src/api/validate.ts`.

The frontend is tolerant: optional fields may be omitted (the UI shows **N/A**), extra fields are ignored, and unknown numeric entries in `metrics` / `timing` are displayed automatically. Required fields are marked **required**.

## Conventions

- JSON over HTTP, `Content-Type: application/json`.
- **Coordinates are `[longitude, latitude]`**, the same order as the raw `.track` / `.nodes` files (GeoJSON order). Example: `[54.963519, 56.251559]`.
- **Fractions are in `[0, 1]`** (`0.912`, not `91.2`). The UI turns them into percentages.
- **Times are in milliseconds** and keys end in `_ms`.
- Trip IDs are strings (e.g. `"00000000"`). Keep the leading zeros.
- Errors use FastAPI's default shape: `{"detail": "human readable message"}` with a proper HTTP status.

## Ground-truth rules (non-negotiable)

1. `GET /evaluation/trips` returns **trip IDs only** (plus optional harmless metadata). No routes, no labels, no arc IDs.
2. `POST /evaluation/run` receives **only** `{"trip_id": ...}`. The UI never sends or receives evaluator ground truth.
3. Accuracy is computed **server-side** against Laptop 1's internal reference data. Only **aggregate metrics** are returned.
4. `reference_route` is optional and **should be omitted for evaluator trips**. The UI supports drawing it (hidden by default) only for cases where the team deliberately allows it, e.g. student-test demos.
5. `observations[].matched_arc_id` is the **predicted** arc, never the reference arc.

---

## `GET /health`

Used for the connection indicator (polled every 15 s) and the **Test connection** button.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "LightGBM (tuned)",
  "hmm_enabled": true,
  "version": "1.0.0",
  "num_evaluation_trips": 10
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | string | `"ok"` when ready. Anything else is shown verbatim. |
| `model_loaded` | bool | optional |
| `model_name` | string | optional, shown in the overview bar |
| `hmm_enabled` | bool | optional |
| `version` | string | optional |
| `num_evaluation_trips` | int | optional |

## `GET /evaluation/trips`

```json
{
  "partition": "evaluator_test",
  "trips": [
    { "trip_id": "00000003", "num_observations": 1874 },
    { "trip_id": "00000018" }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `trips` | array, **required** | Items are objects with **required** `trip_id` (string). A bare string ID is also accepted. |
| `trips[].num_observations` | int | optional, shown on the trip button |
| `trips[].duration_s` | number | optional |
| `trips[].label` | string | optional display label (defaults to `trip_id`) |
| `partition` | string | optional, shown as a caption |

A bare JSON array (`["00000003", ...]`) is also accepted.

## `POST /evaluation/run`

Request:

```json
{ "trip_id": "00000003" }
```

Response: an **Evaluation result** (below), usually `200` with `status: "completed"`.

Asynchronous processing is also supported. Return `202` (or `200`) with `{"evaluation_id": "...", "trip_id": "...", "status": "queued" | "running"}`, and the UI polls `GET /evaluation/{evaluation_id}` every `VITE_POLL_INTERVAL_MS` until the status is `completed` or `failed`, or `VITE_EVALUATION_TIMEOUT_MS` runs out.

Status codes:

| Code | Meaning in the UI |
|---|---|
| 200 / 202 | Parsed as an evaluation result |
| 404 | "Not found on Laptop 1": unknown `trip_id` |
| 422 | FastAPI validation error. `detail[].msg` is shown. |
| 5xx | "Laptop 1 reported a server error" plus `detail` |

A pipeline failure can also be reported as `200` with `status: "failed"` and an `error` message. The UI treats it as a failure and never as a success.

## `GET /evaluation/{evaluation_id}`

Returns the same **Evaluation result** for an earlier run. Used for polling and for reopening a history item after a page reload. Return `404` if unknown. (Keeping results in memory on Laptop 1 is enough.)

## Optional: `GET /evaluation/{evaluation_id}/map`

If Laptop 1 renders a **Folium** map for the run, serve the HTML here and set `folium_map_url` in the result. The UI then shows a second tab, **Folium map from Laptop 1**, with the HTML in an iframe. Alternatively, inline the HTML as `folium_map_html`.

---

## Evaluation result

```json
{
  "evaluation_id": "7f3c2a91",
  "trip_id": "00000003",
  "status": "completed",
  "created_at": "2026-11-18T10:42:00Z",
  "error": null,
  "model": { "name": "LightGBM (tuned)", "version": "1.0.0" },

  "observations": [
    { "index": 0, "lon": 54.963519, "lat": 56.251559, "t": 1.0, "matched_arc_id": 1203 }
  ],

  "predicted_route": {
    "arc_ids": [1203, 1204, 877],
    "segments": [
      { "arc_id": 1203, "coordinates": [[54.9635, 56.2515], [54.9631, 56.2513]] }
    ]
  },

  "road_network": {
    "segments": [
      { "arc_id": 1203, "coordinates": [[54.9635, 56.2515], [54.9631, 56.2513]] }
    ]
  },

  "metrics": {
    "top1_accuracy": 0.912,
    "macro_f1": 0.874,
    "route_continuity": 0.981,
    "log_loss": 0.231,
    "candidate_miss_rate": 0.012,
    "precision": 0.88,
    "recall": 0.87,
    "sequence_accuracy": 0.64
  },
  "classifier_only_metrics": {
    "top1_accuracy": 0.861,
    "macro_f1": 0.812,
    "route_continuity": 0.702,
    "log_loss": 0.298
  },
  "timing": {
    "inference_ms": 412.5,
    "candidate_generation_ms": 120.2,
    "classifier_ms": 180.1,
    "hmm_ms": 98.4,
    "total_ms": 450.0
  },
  "hmm": { "applied": true, "status": "ok", "message": null },

  "folium_map_url": "/evaluation/7f3c2a91/map",
  "warnings": []
}
```

The numbers above only illustrate the format.

| Field | Type | Notes |
|---|---|---|
| `trip_id` | string, **required** | Must equal the requested trip, or the UI rejects the response. |
| `status` | string | `completed` \| `failed` \| `queued` \| `running`. If omitted: `completed` unless `error` is set. |
| `evaluation_id` | string | Strongly recommended (needed for polling and for reopening from history). |
| `error` | string | Failure reason when `status = "failed"`. |
| `observations` | array | Raw GPS points in trip order: `{index, lon, lat, t, matched_arc_id?}`. `[lon, lat, t]` arrays are also accepted. Drawn as amber markers plus a thin track line. |
| `predicted_route.arc_ids` | int[] | Decoded arc sequence. Its length is shown as "Predicted road segments". |
| `predicted_route.segments` | array | Geometry per predicted arc: `{arc_id, coordinates: [[lon,lat], ...]}` (≥ 2 points). A single polyline `predicted_route.coordinates` is also accepted. |
| `road_network.segments` | array | **Nearby subset only** (e.g. arcs within the trip bounding box plus a small margin). Sending all ~18k arcs works but is slow. |
| `reference_route` | same shape as `predicted_route` | Optional. See the ground-truth rules. Hidden by default. |
| `metrics` | object of number \| null | Known keys get labels and formatting. Unknown numeric keys are listed under "Additional metrics". `null` or a string shows N/A. |
| `classifier_only_metrics` | object | Optional. Enables the "Effect of HMM/Viterbi decoding" chart. |
| `timing` | object of ms | `inference_ms` is the headline. Falls back to `total_ms`. |
| `hmm` | object | `{applied: bool, status: string, message: string}` |
| `num_observations` | int | Optional. Used only if `observations` is omitted. |
| `warnings` | string[] | Shown as a warning notice. |

Accepted metric key aliases (so naming differences don't break the UI): `observation_top1_accuracy`, `top_1_accuracy`, `top1` → `top1_accuracy`; `macro_f1_score`, `f1_macro` → `macro_f1`; `continuity` → `route_continuity`; `logloss` → `log_loss`; `macro_precision` / `macro_recall` → `precision` / `recall`; `viterbi_ms` / `decoding_ms` → `hmm_ms`.

---

## Reference FastAPI sketch for Laptop 1 (adapt freely)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="PathSnap API")

# Laptop 2's UI origin(s). Comma-separated in an env var so no code change is needed on the day.
origins = os.environ.get(
    "PATHSNAP_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    trip_id: str

@app.get("/health")
def health(): ...

@app.get("/evaluation/trips")
def list_trips(): ...           # IDs only, never labels

@app.post("/evaluation/run")
def run(req: RunRequest): ...   # pipeline + server-side scoring, returns Evaluation result

@app.get("/evaluation/{evaluation_id}")
def get_evaluation(evaluation_id: str): ...
```

Run it so Laptop 2 can reach it, and so the evaluator can see the request logs:

```powershell
$env:PATHSNAP_CORS_ORIGINS = "http://192.168.1.35:5173,http://localhost:5173"   # Laptop 2's URL
.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info   # repo root (or run_api.bat)
```

Uvicorn's access log prints one line per request (`"POST /evaluation/run HTTP/1.1" 200`), which is the evidence the guidelines ask to show the evaluator. Logging the received `trip_id` and the inference time in the handler makes it clearer still.

---

## Laptop 1 implementation notes

Implemented in `api/main.py` (routes), `api/service.py` (prediction vs evaluation services) and
`api/schemas.py` (Pydantic models). Start with `run_api.bat` or
`.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info`.

### Confirmed

* All four endpoints plus the optional `GET /evaluation/{evaluation_id}/map` (Folium HTML).
* `POST /evaluation/run` is **synchronous**: it returns `200` with `status: "completed"` (no queued/running states).
* `[lon, lat]` everywhere, fractions in `[0, 1]`, times in ms, zero-padded string trip ids.
* Errors use FastAPI's `{"detail": ...}`: `404` unknown trip / evaluation id, `422` invalid body
  (`trip_id` must match `^[0-9A-Za-z_-]+$`), `503` model not loaded, `500` pipeline failure.
* `reference_route` is **never** sent (not even for student-test trips). Laptop 1's Folium map endpoint also omits it.
* `observations[].index` is the row index in the raw `.track` file; `matched_arc_id` is the HMM prediction.
* `road_network.segments` = arcs whose midpoint is within ~150 m of a GPS fix (typically 1–5 k arcs).
* CORS: default `allow_origins=["*"]` (no credentials) so any Laptop 2 address works without configuration.
  Restrict with `PATHSNAP_CORS_ORIGINS="http://<Laptop-2-IP>:5173,http://<Laptop-2-IP>:4173"` if wanted.

### Additions (optional fields; ignored or auto-displayed by the UI)

| Field | Meaning |
|---|---|
| `health.evaluation_partition`, `health.evaluation_partition_is_fallback`, `health.uptime_s` | which trip folder is served; `true` = evaluator trips not installed yet, serving `student_test` |
| `partition` (result) | partition of the trip |
| `model.algorithm`, `model.family`, `model.created_utc`, `model.selection_metric` | model metadata from `artifacts/final/manifest.json` |
| `classifier_only_route` | `{arc_ids, segments}` of the classifier-only (no HMM) prediction |
| `nearest_road_metrics` | metrics of the nearest-road baseline (same keys as `metrics`, `log_loss` = null) |
| `hmm_params` | decoder parameters used |

### Metric keys returned (`metrics` = classifier + HMM, `classifier_only_metrics`, `nearest_road_metrics`)

All are computed on Laptop 1 after prediction, from the trip's `.route` aligned to the GPS fixes.

| key | definition |
|---|---|
| `top1_accuracy` | fraction of GPS fixes whose predicted arc equals the reference arc aligned to that fix |
| `macro_f1`, `precision`, `recall` | candidate level: each fix's candidates are labelled 1 (= reference arc) or 0; the method selects one candidate per fix; macro average over the two classes |
| `log_loss` | log loss of the classifier probabilities on all candidates (the HMM does not re-calibrate them, so it equals the classifier-only value; `null` for nearest road) |
| `route_continuity` | share of consecutive arc changes in the per-fix sequence that are directly connected (to-node = from-node) |
| `network_reachable_continuity` | share of arc changes reachable through the directed network within 500 m |
| `sequence_accuracy` | 1 − normalised Levenshtein distance between predicted and reference arc sequences |
| `candidate_miss_rate` | share of fixes whose reference arc was not among the candidates |
| `route_precision`, `route_recall`, `route_f1` | length-weighted overlap of predicted and reference route arcs |
| `route_jaccard` | arc-set Jaccard index |
| `edit_distance`, `normalized_edit_distance` | Levenshtein distance between the arc sequences (raw / normalised) |
| `accuracy_undirected` | like `top1_accuracy` but the reverse twin of the correct arc counts as correct |
| `invalid_transitions`, `unreachable_transitions` | counts of the two continuity failures above |
| `mean_deviation_m`, `p95_deviation_m` | distance from matched positions to the reference route |
| `predicted_route_arcs`, `observations_evaluated` | counts |

### Timing keys (`timing`, ms)

`inference_ms` (full prediction, headline), `preprocessing_ms`, `candidate_generation_ms` (candidates + features),
`classifier_ms`, `hmm_ms` (Viterbi), `route_reconstruction_ms`, `evaluation_ms` (server-side scoring), `total_ms`.

### Evaluator trips

Laptop 1 serves the evaluator trips from `<DATA_ROOT>/evaluator_test/<trip_id>/` or from the folder given by
`PATHSNAP_EVAL_DIR`. If neither exists it serves `student_test` and `/health` reports
`evaluation_partition_is_fallback: true`.

### `GET /models/comparison` (added for the dashboard's "Model comparison" section)

Read-only. Serves the recorded TRAIN cross-validation comparison of the 8 candidate classifiers from
`reports/experiments/model_comparison.csv` (5-fold GroupKFold by trip on the 80 TRAIN trips), plus the nearest-road
baseline and the deployed Optuna-tuned variant (`tuned_vs_untuned.csv`). No student-test or evaluator-test metrics are
included (`not_evaluator_results: true`); values that were never recorded are `null`. `404` if the results file is missing.
Fields per model: `key, name, family, cv_top1_accuracy(_std), cv_macro_f1(_std), cv_log_loss, cv_route_continuity,
train_fold_top1_accuracy, fit_time_s_per_fold, fit_time_s_all_train, inference_ms_per_observation, selected`.
Parsed by `parseModelComparison` in `frontend/src/api/validate.ts`.
