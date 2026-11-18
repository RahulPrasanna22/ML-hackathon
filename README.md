# PathSnap: Distributed Map Matching & Route Reconstruction

PathSnap reconstructs the road route a vehicle actually drove from noisy GPS observations. GPS fixes drift onto parallel streets, jump across buildings, drop out and arrive at low sampling rates. PathSnap matches each observation to a road segment of the trip's road network:

1. **Candidate generation:** nearby road segments (arcs) for every GPS point.
2. **Supervised classifier:** scores each candidate. At least five models are compared, and the chosen one is tuned with Bayesian optimisation.
3. **HMM + Viterbi decoding:** turns per-point scores into a connected, realistic route using road-network connectivity.

The system is deployed on two laptops: the model and API on one, the evaluator interface on the other.

## Architecture

```
 ┌──────────────────────── Laptop 2 ────────────────────────┐        ┌──────────────────── Laptop 1 ────────────────────┐
 │  Evaluator UI (frontend/)                                │  HTTP  │  FastAPI backend                                 │
 │  React + TypeScript + Leaflet                            │ ─────► │  GET  /health                                    │
 │  • lists the 10 evaluation trips                         │  JSON  │  GET  /evaluation/trips      (IDs only)          │
 │  • sends the selected trip_id                            │ ◄───── │  POST /evaluation/run        {trip_id}           │
 │  • draws GPS, predicted route, road network              │        │  GET  /evaluation/{id}                           │
 │  • shows metrics, charts, session history                │        │  candidates → classifier → HMM/Viterbi →         │
 │  (no ML, no ground truth, no hardcoded results)          │        │  server-side scoring against reference routes    │
 └──────────────────────────────────────────────────────────┘        └──────────────────────────────────────────────────┘
```

| Laptop | Owner | Responsibilities |
|---|---|---|
| **Laptop 1: model & API** | Raghav | Data processing, feature engineering, model training and comparison, Bayesian hyperparameter optimisation, HMM/Viterbi, exported artifacts, FastAPI backend, server-side evaluation |
| **Laptop 2: user interface** | Rachit | Evaluator dashboard: trip selection, API communication, route map, metrics, charts, history |

Ground-truth labels stay on Laptop 1. The UI only ever sends a trip ID and receives predictions plus aggregate metrics.

## Repository layout

| Path | Contents | Status |
|---|---|---|
| `frontend/` | Laptop 2 evaluator UI (React 19, Vite, TypeScript, Leaflet, Vitest) | Implemented and tested |
| `docs/API_CONTRACT.md` | Proposed REST/JSON contract between the laptops | Proposed, awaiting Laptop 1 confirmation |
| `docs/TWO_LAPTOP_SETUP.md` | Networking, CORS, firewall and demo runbook (PowerShell) | Done |
| `docs/DEVELOPER_HANDOFF_LAPTOP_1.md` | Integration brief for the backend developer | Done |
| *(Laptop 1 folders)* | Data processing, training, HMM, API | To be added by Laptop 1 |

### Dataset (not in the repository)

The hackathon dataset is kept locally and is **not** committed (see `.gitignore`). Place it at:

```
Student folder/Student folder/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          # 80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   # 10 trips
```

| File | Columns (whitespace separated, no header) |
|---|---|
| `.track` | longitude, latitude, timestamp (s), about 1 Hz |
| `.nodes` | longitude, latitude (row index = node id) |
| `.arcs` | from_node_id, to_node_id (row index = arc id) |
| `.route` | ground-truth arc ids in travel order |

The evaluator-testing partition (10 trips) is used for final evaluation only and must never be committed or exposed to the UI.

## Installation (Windows PowerShell)

Prerequisites: Git, Node.js 20+ (tested with 26.7), and Python 3.11+ for Laptop 1.

```powershell
git clone https://github.com/RachitSMane/ml-hackathon-project.git
cd ml-hackathon-project\frontend
npm install
```

## Running the frontend (Laptop 2)

```powershell
cd frontend
Copy-Item .env.example .env.local
notepad .env.local                 # VITE_API_BASE_URL=http://<Laptop-1-IPv4>:8000

npm run dev                        # development, http://localhost:5173
# or
npm run build; npm run preview     # production build, http://<this-laptop>:4173
```

## API configuration

The UI uses the first of these that is set:

1. `?api=http://<ip>:8000` in the page URL
2. The address entered in the UI's **Connection** panel (remembered per browser; no rebuild needed)
3. `VITE_API_BASE_URL` in `frontend/.env.local`

There is no hardcoded localhost default. Laptop 1 must start uvicorn with `--host 0.0.0.0`, open its port in Windows Firewall, and allow the UI origin in CORS. See [`docs/TWO_LAPTOP_SETUP.md`](docs/TWO_LAPTOP_SETUP.md). If CORS can't be fixed in time, there is a proxy fallback (`PATHSNAP_PROXY_TARGET`).

Full variable reference: [`frontend/README.md`](frontend/README.md).

## Testing

```powershell
cd frontend
npm run typecheck
npm test
npm run build
```

Automated tests use clearly separated mocked responses (`frontend/src/test/`). These are never bundled into the app.

## Current implementation status

| Component | Status |
|---|---|
| Evaluator UI (Laptop 2) | Implemented: 53 automated tests passing, production build passing |
| API contract | Proposed in `docs/API_CONTRACT.md`, pending Laptop 1 agreement |
| ML pipeline, HMM/Viterbi, FastAPI backend (Laptop 1) | In progress on Laptop 1, not yet in this repository |
| End-to-end integration with the real backend | **Not yet done.** The UI has only been exercised against test mocks and a local stub. |

## Academic integrity

All results shown by the UI come from the team's own backend. No predictions or metrics are hardcoded, and evaluator ground truth is never sent to the UI. AI assistance was used in developing the frontend and its documentation.
