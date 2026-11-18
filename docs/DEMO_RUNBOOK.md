# PathSnap — Final demo runbook (Laptop 1 = model + API, Laptop 2 = UI)

Verified on Laptop 1 on 2026-11-18. `run_api.bat` started the server, it loaded the final bundle in 1.2 s,
`POST /evaluation/run` returned 200 (trip 00000052, top-1 1.000), and an unknown trip returned 404.

Final bundle (do not touch): `artifacts/final/`. It holds XGBoost (Optuna-tuned, trial 1) plus HMM/Viterbi (TRAIN-tuned:
β 22.28, emission weight 0.215). `model.joblib` sha256 prefix `7ebb199fc9b56a3d`.

> **Use `127.0.0.1` or the Wi-Fi IP, never `localhost`.** On this laptop `localhost` resolves to IPv6 `::1` first, and the
> API listens on IPv4 only, so every `localhost` request waits ~2 s for the fallback (measured: 2 058 ms vs 12 ms).

## 0. Before you start (Laptop 1)

```powershell
cd "C:\Users\Raghav\Downloads\ML Hackathon"
# free ports left over from earlier sessions (no-op if nothing is listening)
Get-NetTCPConnection -LocalPort 8000,5173,4173 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
# your Wi-Fi IPv4 (give this to Laptop 2)
Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" | Select-Object IPAddress
```

### Firewall (Administrator PowerShell, once per laptop)

The Wi-Fi network is classified **Public**, so Windows blocks inbound connections unless a rule allows them.

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
# check:
Get-NetFirewallRule -DisplayName "PathSnap API 8000" | Select-Object DisplayName, Enabled, Action, Profile
```
If Windows shows "Allow python.exe to communicate on these networks" the first time the server starts, tick **both**
Private and Public and click Allow. Clicking Cancel creates a silent *block* rule.

## 1. Launch FastAPI (Laptop 1, normal PowerShell, keep this window visible)

```powershell
cd "C:\Users\Raghav\Downloads\ML Hackathon"
# optional: evaluator trips handed out on the day (folder containing <trip_id>\<trip_id>.track/.nodes/.arcs[/.route])
$env:PATHSNAP_EVAL_DIR = "D:\evaluator_test"
.\run_api.bat
```
Equivalent without the batch file:
`.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info`

Expected console lines: `Model loaded in … ms: XGBoost (tuned) + HMM/Viterbi`, `CORS allowed origins: ['*']`,
`Uvicorn running on http://0.0.0.0:8000`. Without `PATHSNAP_EVAL_DIR` you get a WARNING that it serves `student_test`.

Check on Laptop 1: `Invoke-RestMethod http://127.0.0.1:8000/health`
(expect `status=ok`, `model_name=XGBoost (tuned) + HMM/Viterbi`, `num_evaluation_trips=10`; also check `evaluation_partition`).

## 2. Launch the React frontend (Laptop 2)

```powershell
cd <repo>\frontend
npm install                     # first time only
npm run build; npm run preview  # production build at http://localhost:4173  (or: npm run dev -> :5173)
```
The frontend is unchanged since commit `8f62e2c`, so Laptop 2 needs nothing from Laptop 1's local work.

## 3. Connect Laptop 2 to Laptop 1 over Wi-Fi

1. Both laptops on the **same** Wi-Fi (or the same phone hotspot). Venue Wi-Fi often isolates clients; if step 2 fails
   with the firewall rule in place, switch both laptops to a phone hotspot and re-read Laptop 1's IP.
2. From Laptop 2: `Invoke-RestMethod http://<LAPTOP1-IP>:8000/health` must return `status: ok`.
3. Open `http://localhost:4173/?api=http://<LAPTOP1-IP>:8000` (or click **Connection** in the UI, enter
   `http://<LAPTOP1-IP>:8000`, **Test connection**, **Save and connect**).
4. Header shows **Laptop 1 API · Connected**, and the overview shows 10 evaluation trips.

CORS is open (`*`) by default, so no Laptop 2 IP needs configuring. If a browser ever reports a CORS error,
use the UI's proxy fallback (`docs/TWO_LAPTOP_SETUP.md` §6).

## 4. Live demonstration

1. Laptop 1 console visible next to the UI.
2. Pick a trip → **Run evaluation**. Laptop 1 logs `RUN id=… trip=…`, `DONE … top1=… inference=… ms`,
   `POST /evaluation/run -> 200`.
3. UI shows the GPS points (amber), the predicted route (cyan) and the road network, plus top-1, macro-F1, continuity,
   log loss, inference time, the "Effect of HMM/Viterbi" chart and session history. The **Folium map from Laptop 1** tab shows
   the server-rendered map (no reference route).
4. Good trips to show (student-test, measured): `00000052` (262 fixes, 1.000, ~0.3 s), `00000007` (0.995),
   `00000076` (largest HMM gain, +4.4 pp), `00000056` (largest, 3 958 fixes, ~3–4 s).
   On evaluator trips the numbers will differ, because they are unseen data.

## 5. Backup: one-laptop demo (all on Laptop 1)

Node and `frontend/node_modules` are already installed on Laptop 1.

```powershell
# terminal 1
cd "C:\Users\Raghav\Downloads\ML Hackathon"; .\run_api.bat
# terminal 2
cd "C:\Users\Raghav\Downloads\ML Hackathon\frontend"; npm run build; npm run preview
# browser
start "http://127.0.0.1:4173/?api=http://127.0.0.1:8000"
```
Further fallbacks if the UI itself has a problem:
* Swagger UI: `http://127.0.0.1:8000/docs` → `POST /evaluation/run` → *Try it out* with `{"trip_id": "00000052"}`.
* PowerShell: `Invoke-RestMethod -Method Post -ContentType 'application/json' -Body '{"trip_id":"00000052"}' http://127.0.0.1:8000/evaluation/run`
* Pre-generated evidence (no server needed): `reports/maps/match_student_test_*.html` (offline Folium maps),
  `reports/integration/ui_trip_*.png` (UI screenshots), `reports/RESULTS.md`.
* Self-check before presenting (both servers running): `.venv\Scripts\python integration\e2e_browser.py --ui http://127.0.0.1:4173 --api http://127.0.0.1:8000`
  → must print `E2E PASSED`.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| UI "unreachable", `/health` fails from Laptop 2 | firewall rule (§0), same network, correct IP, try hotspot |
| everything ~2 s slower than expected | you used `localhost`; use `127.0.0.1` / the IP |
| `/health` → `evaluation_partition_is_fallback: true` | `PATHSNAP_EVAL_DIR` not set / wrong folder: stop the server, set it, restart |
| port 8000/4173 in use | §0 "free ports" command |
| `status: degraded` | model failed to load. Read the Laptop 1 console; the bundle must be in `artifacts\final\` |
