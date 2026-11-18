# PathSnap evaluator UI (Laptop 2)

The evaluator-facing dashboard for PathSnap: it lists the ten reserved evaluation trips served by the Laptop 1 FastAPI backend, sends the selected trip ID to `POST /evaluation/run`, and visualises the returned route and metrics.

The UI contains **no ML pipeline, no hardcoded predictions and no ground truth**. Every route, coordinate and metric it shows comes from the API response. Missing values are shown as **N/A**.

- API contract: [`../docs/API_CONTRACT.md`](../docs/API_CONTRACT.md)
- Two-laptop networking and demo runbook: [`../docs/TWO_LAPTOP_SETUP.md`](../docs/TWO_LAPTOP_SETUP.md)

## Stack

| Concern | Choice | Why |
|---|---|---|
| App | React 19 + TypeScript + Vite | Typed API layer, fast dev server, static production build |
| Map | Leaflet 1.9 via react-leaflet | Folium generates Leaflet maps, so this is the same engine rendered natively from API JSON, with layer control, zoom/pan, tooltips and fit-to-bounds. If the backend serves a Folium HTML map (`folium_map_url` / `folium_map_html`), it is also shown in a second tab. |
| Charts | Hand-written SVG | No chart dependency. Charts only render for metrics actually returned. |
| Tests | Vitest + Testing Library + jsdom | Mocked fetch and fixtures live only in `src/test/` |

Base map tiles come from OpenStreetMap (no API key; needs internet). Without internet the tiles are blank, but all API layers still draw.

## Prerequisites

- Node.js 20+ (tested with Node 26.7, npm 11.19)

## Install

```powershell
cd frontend
npm install
```

## Configure

```powershell
Copy-Item .env.example .env.local
notepad .env.local      # set VITE_API_BASE_URL=http://<Laptop-1-IPv4>:8000
```

You can also leave it empty and set the address at runtime with the **Connection** button, or open the app with `?api=http://<ip>:8000`.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | *(none)* | Laptop 1 API base URL. With no value, the UI asks for one. There is deliberately no localhost default. |
| `VITE_API_TIMEOUT_MS` | `10000` | Timeout for `/health`, `/evaluation/trips`, `GET /evaluation/{id}` |
| `VITE_EVALUATION_TIMEOUT_MS` | `180000` | Overall limit for one evaluation run, including polling |
| `VITE_POLL_INTERVAL_MS` | `1500` | Poll interval if the backend answers `queued`/`running` |
| `VITE_DEV_FALLBACK_TRIPS` | *(empty)* | **Dev only.** Comma-separated trip IDs shown when the trip list can't be loaded. Always labelled "Dev fallback, not backend data". Ignored in production builds. Runs still go to the backend. |
| `PATHSNAP_PROXY_TARGET` | *(empty)* | Optional. Used by the Vite dev/preview server only. Forwards `/api/*` to Laptop 1 (use with `VITE_API_BASE_URL=/api`) as a CORS-free fallback. |

Resolution order for the API address: `?api=` link parameter → address saved via the Connection panel → `VITE_API_BASE_URL` → ask the user.

`VITE_*` values are compiled into the browser bundle, so never put secrets in them. `.env.local` is git-ignored.

## Run (development)

```powershell
npm run dev          # http://localhost:5173 (this laptop only)
npm run dev:lan      # also reachable from other devices on the network
```

## Production build

```powershell
npm run build        # type-checks, then writes dist/
npm run preview      # serves dist/ on http://0.0.0.0:4173
```

`dist/` is a static site and can be served by any static file server. `VITE_*` values are read at build time, so rebuild after changing `.env.local`. The runtime Connection panel works either way.

## Test

```powershell
npm test             # run once
npm run test:watch   # watch mode
npm run typecheck
```

The tests cover dashboard rendering, the connection panel, trip retrieval and loading/error/fallback states, trip selection, the run request body, duplicate-submission blocking, results rendering, N/A handling for missing metrics, map layers/legend/bounds, invalid and empty geometry, backend failures (5xx, 404, `status: failed`, timeout, network/CORS, contract violations), polling, charts, and session history (including that no geometry or reference data reaches `sessionStorage`).

## Project layout

```
src/
  api/          client.ts (the only module that calls fetch), validate.ts (runtime
                response validation), types.ts, errors.ts
  hooks/        useBackendStatus, useTrips, useEvaluation
  lib/          geo.ts (coordinate validation + Leaflet layers), metrics.ts
                (catalogue + formatting), history.ts (session history)
  components/   Header, ConnectionPanel, OverviewBar, TripSelector, RunStatus,
                MapPanel, RouteMap, ResultsPanel, Charts, HistoryPanel, Notice
  test/         TEST-ONLY fixtures, fetch mock, react-leaflet mock
  config.ts     environment + runtime configuration
```

## Data handling

- Session history lives in `sessionStorage` (cleared when the tab closes). It stores trip ID, status, timestamp and the aggregate metrics only. No geometry, per-observation data or reference routes.
- The saved API address is the only value in `localStorage`.
- Invalid coordinates (non-finite, out of range, `0,0`) are dropped and counted in a visible note. They are never repaired or interpolated.
- A reference route is drawn only if the backend sends `reference_route`, and its layer is off by default. Per the hackathon rules, the backend should not send it for evaluator trips.

## Known notes

- `npm audit` reports a high-severity ReDoS advisory in `picomatch`, which is pulled in by the build tooling (Vite/Vitest). It is not part of the shipped bundle. Re-check with `npm audit` when updating dependencies.
