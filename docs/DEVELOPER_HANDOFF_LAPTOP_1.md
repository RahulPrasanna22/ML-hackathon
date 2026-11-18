# Handoff: Laptop 1 backend ⇄ Laptop 2 UI

**To:** Raghav (Laptop 1: ML pipeline + FastAPI)
**From:** Rachit (Laptop 2: evaluator UI)

The UI in `frontend/` is finished and tested against mocks. It has **not** yet been run against the real backend. This page is the short version of what the backend needs for integration. The authoritative details are in [`API_CONTRACT.md`](API_CONTRACT.md) and [`TWO_LAPTOP_SETUP.md`](TWO_LAPTOP_SETUP.md).

## 1. Endpoints to implement

| Method | Path | Purpose | Required? |
|---|---|---|---|
| GET | `/health` | Connection indicator, model name | Yes |
| GET | `/evaluation/trips` | The 10 evaluation trip IDs (**IDs only**) | Yes |
| POST | `/evaluation/run` | Run the full pipeline on one trip, score it server-side, return the result | Yes |
| GET | `/evaluation/{evaluation_id}` | Return a stored result (polling and history) | Strongly recommended |
| GET | `/evaluation/{evaluation_id}/map` | Folium HTML for the run (shown in a second tab) | Optional |

## 2. Request and response JSON

`GET /health`

```json
{ "status": "ok", "model_loaded": true, "model_name": "LightGBM (tuned)", "hmm_enabled": true }
```

`GET /evaluation/trips`

```json
{ "partition": "evaluator_test", "trips": [ { "trip_id": "00000003", "num_observations": 1874 } ] }
```

`POST /evaluation/run` with body `{ "trip_id": "00000003" }` returns:

```json
{
  "evaluation_id": "7f3c2a91",
  "trip_id": "00000003",
  "status": "completed",
  "model": { "name": "LightGBM (tuned)", "version": "1.0.0" },
  "observations": [ { "index": 0, "lon": 54.9635, "lat": 56.2515, "t": 1.0, "matched_arc_id": 1203 } ],
  "predicted_route": {
    "arc_ids": [1203, 1204],
    "segments": [ { "arc_id": 1203, "coordinates": [[54.9635, 56.2515], [54.9631, 56.2513]] } ]
  },
  "road_network": { "segments": [ { "arc_id": 1203, "coordinates": [[54.9635, 56.2515], [54.9631, 56.2513]] } ] },
  "metrics": { "top1_accuracy": 0.91, "macro_f1": 0.87, "route_continuity": 0.98, "log_loss": 0.23 },
  "classifier_only_metrics": { "top1_accuracy": 0.86, "macro_f1": 0.81, "route_continuity": 0.70 },
  "timing": { "inference_ms": 412.5, "hmm_ms": 98.4 },
  "hmm": { "applied": true, "status": "ok", "message": null },
  "warnings": []
}
```

The numbers above only illustrate the format. Every field except `trip_id` is optional for the UI: anything missing shows as N/A. The more you send, the more the UI shows.

Errors: use `HTTPException(404, "Unknown trip …")` for unknown trips. For a pipeline failure, return either a 5xx with `detail`, or `200` with `"status": "failed", "error": "…"`.

## 3. Conventions (please follow exactly)

- **Coordinates are `[longitude, latitude]`**, the same order as `.track` and `.nodes`. Don't swap them.
- **Fractions are 0–1** (`0.91`, not `91`).
- **Times are milliseconds**, with keys ending in `_ms`.
- **Trip IDs are strings** with leading zeros (`"00000003"`).
- `trip_id` in the response must equal the requested one. The UI rejects mismatches.

## 4. Route geometry is required

The UI can't draw arc IDs alone; it needs coordinates. Arc `i` has end nodes from row `i` of `.arcs`, and their coordinates come from `.nodes`:

```python
segments = [
    {"arc_id": int(a), "coordinates": [nodes[arcs[a][0]].tolist(), nodes[arcs[a][1]].tolist()]}
    for a in predicted_arc_ids
]
```

- `predicted_route.segments`: one entry per predicted arc, in decoded order.
- `road_network.segments`: **only arcs near the trip** (e.g. both nodes inside the GPS bounding box ± ~200 m). Sending all ~18k arcs works but is slow.
- `observations`: the raw `.track` points (`lon`, `lat`, `t`), plus the **predicted** `matched_arc_id` if available.

## 5. Ground-truth rules

- Never put `.route` contents in `/evaluation/trips` or in the request.
- **Don't send `reference_route` for evaluator trips.** The UI can draw it (hidden by default), but only use that for student-test demos, if at all.
- Return only **aggregate** metrics computed server-side.

## 6. CORS configuration

```python
import os
from fastapi.middleware.cors import CORSMiddleware

origins = os.environ.get("PATHSNAP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

The origin must match the UI address exactly, including the port: `http://<Laptop-2-IP>:5173` for dev, `:4173` for preview. The Connection panel in the UI displays its exact origin.

## 7. Startup on Laptop 1 (PowerShell)

```powershell
# 1. Find Laptop 1's IPv4 (Wi-Fi adapter) and tell Rachit
ipconfig

# 2. Open the port (Administrator PowerShell, once)
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public

# 3. Allow Laptop 2's origin and start on all interfaces
$env:PATHSNAP_CORS_ORIGINS = "http://<Laptop-2-IP>:5173,http://<Laptop-2-IP>:4173,http://localhost:5173"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

`--host 0.0.0.0` is what exposes the API to Laptop 2. `127.0.0.1` would only accept local connections. Keep this terminal visible during the demo: the evaluator must see the `POST /evaluation/run` request logs.

Quick check from Laptop 2:

```powershell
Invoke-RestMethod http://<Laptop-1-IP>:8000/health
Invoke-RestMethod http://<Laptop-1-IP>:8000/evaluation/trips
Invoke-RestMethod -Method Post -ContentType 'application/json' -Body '{"trip_id":"<id>"}' http://<Laptop-1-IP>:8000/evaluation/run
```

## 8. Frontend files: please don't modify without coordinating

- `frontend/src/api/validate.ts` and `frontend/src/api/types.ts`: the executable form of the contract. If the contract changes, update these **and** `docs/API_CONTRACT.md` together, and run `npm test`.
- `frontend/src/api/client.ts`: the only module that calls the backend.
- `frontend/src/test/`: test-only fixtures. Never import them into app code.
- `frontend/.env.example`: template only. Real addresses go in the git-ignored `.env.local`.

If you prefer different field names or paths, tell Rachit or document the change in `API_CONTRACT.md` first. The UI already accepts some aliases (e.g. `observation_top1_accuracy`, `f1_macro`). See the contract for the list.

## 9. Repository notes

- The raw dataset (`Student folder/`) and course PDFs are git-ignored and stay local.
- The deliverables require model weights and configs in the repo, so commit your artifacts in your own folders (e.g. `backend/`, `models/`), within GitHub's 100 MB per-file limit.
- Don't commit `.env` files, virtual environments or evaluator-test data. The root `.gitignore` already covers these.
