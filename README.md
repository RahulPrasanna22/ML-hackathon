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
| `docs/API_CONTRACT.md` | REST/JSON contract between the laptops | Proposed by Laptop 2, confirmed and implemented by Laptop 1 (notes at the end) |
| `docs/TWO_LAPTOP_SETUP.md` | Networking, CORS, firewall and demo runbook (PowerShell) | Done |
| `docs/DEVELOPER_HANDOFF_LAPTOP_1.md` | Integration brief for the backend developer | Done |
| `pathsnap/` | Laptop 1 Python package: parsing, validation, GPS cleaning, geodesy, road network, candidates, features, label alignment | Preprocessing done and tested |
| `scripts/01_explore.py`, `scripts/02_build_dataset.py`, `scripts/demo_preprocessing.py` | EDA, dataset build, first-review demo | Done |
| `tests/` | Python tests: parsing, preprocessing, candidates, features, HMM, models, leakage/integrity, API | 46 passing |
| `reports/` | EDA report, statistics, figures, Folium map, preprocessing demo outputs | Done |
| `docs/PREPROCESSING_REVIEW.md` | First-review document (data preprocessing) | Done |
| `pathsnap/models.py`, `hmm.py`, `pipeline.py`, `evaluation.py`, `viz.py` | model zoo, HMM/Viterbi, inference pipeline, server-side evaluation, Folium maps | Done |
| `scripts/03`–`09` | training + CV, Optuna, finalisation, decoder tuning, evaluation, plots, benchmark | Done |
| `api/` | FastAPI backend (`api.main:app`) | Done |
| `artifacts/final/` | final model bundle (model, feature schema, HMM parameters, manifest) | Done |
| `integration/` | contract test (real API JSON → frontend validator), browser E2E test, captured fixtures | Done |
| `reports/RESULTS.md` | model comparison, optimisation, HMM, held-out results, benchmarks, integration | Done |

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
| API contract | Confirmed and implemented; real responses pass the frontend validator (`integration/contract.test.ts`) |
| Data preprocessing and feature engineering (Laptop 1) | Done, see [First Review: Data Preprocessing](#first-review-data-preprocessing) |
| 8 models, Optuna, HMM/Viterbi, FastAPI backend (Laptop 1) | Done. Results in [`reports/RESULTS.md`](reports/RESULTS.md) (student-test, classifier + HMM: accuracy 0.9155, macro-F1 0.9523) |
| End-to-end integration (real model) | Verified locally on one laptop: Edge → UI (dev and production build) → FastAPI → final model; all checks passed (`reports/integration/`). Two-laptop LAN run still to be done on the day |

## First Review: Data Preprocessing

Full write-up with statistics, before and after examples, and what is still pending:
[`docs/PREPROCESSING_REVIEW.md`](docs/PREPROCESSING_REVIEW.md). Dataset exploration: [`reports/EDA.md`](reports/EDA.md).

Pipeline (`pathsnap/`): parse `.track/.nodes/.arcs` → validate → clean GPS (invalid points, ordering, duplicate
timestamps) → local metric projection, speed and heading → crop the road network, build the KD-tree spatial index and the directed graph →
candidate arcs within 50 m (≤ 12 per fix) → 28 label-free features → TRAIN-only labels from the reference route
(monotone alignment, joined after features).

Demo commands (Windows PowerShell, from the repository root, dataset at `Student folder\Student folder\`):

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 1) single-trip walkthrough (~1 s), writes reports\preprocessing_demo\
.venv\Scripts\python scripts\demo_preprocessing.py --data-root "Student folder\Student folder\train" --trip 00000003

# 2) build features + labels for a small TRAIN sample (3 trips, ~3 s; full run: omit --limit)
.venv\Scripts\python scripts\02_build_dataset.py --partitions train --limit 3

# 3) preprocessing tests
.venv\Scripts\python -m pytest tests\test_data_preprocessing.py tests\test_candidates_features.py -v

# 4) (optional, ~3 min) dataset-wide EDA: statistics, figures and a Folium map
.venv\Scripts\python scripts\01_explore.py
```

Visuals: `reports/preprocessing_demo/preprocessing_00000003.png`, `reports/figures/eda_*.png`, and the Folium map
`reports/maps/eda_trip_00000003.html` (open it in a browser).

## Laptop 1: ML pipeline and API

### Pipeline

| step | script | data used | output |
|---|---|---|---|
| EDA | `scripts/01_explore.py` | train, student_test | `reports/EDA.md`, figures, Folium map |
| features + labels | `scripts/02_build_dataset.py` | train, student_test (separately) | `cache/` (git-ignored) |
| 8 models, GroupKFold(5) by trip | `scripts/03_train_models.py` | **train** (CV); student_test only for reporting | `reports/experiments/model_comparison.csv`, `cv_folds.csv` |
| Bayesian optimisation (Optuna TPE) | `scripts/04_bayes_opt.py --model xgboost` | **train only** | `reports/experiments/optuna_xgboost_*`, `artifacts/optuna/*.db` |
| final model + OOF probabilities | `scripts/05_finalize_model.py --model xgboost` | **train**; student_test reported | `artifacts/final/` bundle |
| HMM decoder tuning | `scripts/06_tune_decoder.py` | **train out-of-fold** probabilities only | `artifacts/final/hmm_params.json` |
| held-out confirmation + maps | `scripts/07_evaluate_routes.py --maps` | student_test (confirmation only) | `reports/experiments/route_eval_*`, `reports/maps/match_*.html` |
| plots | `scripts/08_make_plots.py` | saved results | `reports/figures/*.png` |
| benchmark | `scripts/09_benchmark.py` | student_test | `reports/experiments/benchmark.json` |

Everything above: `reproduce_all.bat`. The evaluator partition is never read by any of these scripts.

### Final artifact bundle (`artifacts/final/`)

`model.joblib` (classifier), `feature_schema.json` (28 feature names and order), `hmm_params.json` (decoder),
`manifest.json` (algorithm, tuned hyperparameters, Optuna summary, training data, selection metric, CV and
student-test metrics, preprocessing configuration, library versions). `MapMatcher.load()` reloads it in a fresh
process. Road networks are read per trip at request time.

### Start the backend (Laptop 1)

```powershell
cd "<repo>"
.\setup_windows.bat                                   # once: venv + requirements
# Administrator PowerShell, once:
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
# evaluator trips (when provided): a folder containing <trip_id>\<trip_id>.track/.nodes/.arcs[/.route]
$env:PATHSNAP_EVAL_DIR = "D:\evaluator_test"          # omit -> serves student_test (dev mode, /health says so)
.\run_api.bat                                         # = .venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Start the frontend (Laptop 2) and connect

```powershell
cd frontend; npm install; npm run dev:lan             # http://<laptop2>:5173
# open http://localhost:5173/?api=http://<Laptop-1-IPv4>:8000
```

### Integration checks

```powershell
.venv\Scripts\python -m pytest tests -q                                        # Python: unit, integrity, model, API
cd frontend; npx vitest run --config ../integration/vitest.config.mts; cd ..   # real API JSON through validate.ts
.venv\Scripts\python integration\e2e_browser.py                                # Edge -> UI -> API -> model (needs both servers)
```

## Academic integrity

All results shown by the UI come from the team's own backend. No predictions or metrics are hardcoded, and evaluator ground truth is never sent to the UI. AI assistance was used in developing the frontend and its documentation.
