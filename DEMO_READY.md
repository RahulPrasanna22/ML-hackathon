# PathSnap — DEMO READY (verified 2026-11-18, 15:15–15:25)

**Status: DEMO READY. Two-laptop connection verified.** Backend + frontend run from the frozen bundle on the official
`evaluator_test` partition, and the browser test passed on all 10 evaluator trips. At 15:29–15:30 Laptop 2
(`10.20.205.106`) connected over Wi-Fi: `/health`, `/evaluation/trips`, CORS preflight, and `POST /evaluation/run` for
00000015 and 00000059 all returned 200 (logged on Laptop 1), with metrics identical to the earlier runs. The current Wi-Fi
works without extra firewall changes. Note: 00000015 took 6.3 s on that run (2.0 s earlier, same results); show 00000059 first.

| item | value |
|---|---|
| Laptop 1 Wi-Fi IPv4 | **`10.8.121.205`** (network "1B-AUDI", Windows category *Public*; updated 17:06, it was 10.20.206.134 on the earlier network). Re-check with `ipconfig` whenever the Wi-Fi changes |
| Backend | `http://10.8.121.205:8000` (local: `http://127.0.0.1:8000`), docs at `/docs` |
| Frontend (production build) | `http://127.0.0.1:4173` on Laptop 1, or `http://localhost:4173` on Laptop 2 |
| Final model | `artifacts/final/`: XGBoost (Optuna-tuned) + TRAIN-tuned HMM/Viterbi, `model.joblib` sha256 `7ebb199fc9b56a3d…` (frozen) |
| Evaluator data | `C:\Users\Raghav\Downloads\ML Hackathon\evaluator_test` (10 trips; git-ignored) |

> Always use **`127.0.0.1` or the IP**, not `localhost`, when talking to the backend *from Laptop 1*.
> `localhost` resolves to IPv6 first, which costs ~2 s per request.

## 1. Start Laptop 1 (backend), PowerShell

```powershell
cd "C:\Users\Raghav\Downloads\ML Hackathon"
# free the demo ports if an old PathSnap server is still running
Get-NetTCPConnection -LocalPort 8000,4173 -State Listen -ErrorAction SilentlyContinue | % { Stop-Process -Id $_.OwningProcess -Force }
$env:PATHSNAP_EVAL_DIR = "C:\Users\Raghav\Downloads\ML Hackathon\evaluator_test"
.\run_api.bat
```
Must print `Evaluation partition 'evaluator_test': 10 trips`, `Model loaded … XGBoost (tuned) + HMM/Viterbi`,
`Uvicorn running on http://0.0.0.0:8000`. If `PATHSNAP_EVAL_DIR` is wrong it prints `CRITICAL | REFUSING TO START` and
exits; it never silently serves student-test.

Verify (second PowerShell on Laptop 1):
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
# status=ok, evaluation_partition=evaluator_test, evaluation_partition_is_fallback=False, num_evaluation_trips=10
```

### Firewall (Administrator PowerShell, once)
Inbound *Allow* rules for `Python311\python.exe` (the process that actually listens) already exist for the Public profile,
so Laptop 2 will probably connect without changes. For certainty, add an explicit port rule:
```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
```

## 2. Start Laptop 2 (frontend), PowerShell

```powershell
cd <repo>\frontend
npm install                      # first time only
npm run build; npm run preview   # serves http://localhost:4173
```
Then open **`http://localhost:4173/?api=http://10.8.121.205:8000`**

## 3. Verify the Laptop 2 → Laptop 1 connection (manual, on Laptop 2)

```powershell
Invoke-RestMethod http://10.8.121.205:8000/health
Invoke-RestMethod http://10.8.121.205:8000/evaluation/trips
```
Expect `status ok` and 10 trips (`00000015 … 00000094`). The UI header must then show **Laptop 1 API · Connected** and
**Evaluation trips: 10**. If it fails, see §7. The network name "LAPTOP-ONLY" suggests the venue Wi-Fi may isolate
devices from each other; in that case put both laptops on a phone hotspot and re-read Laptop 1's IP (`ipconfig`).

## 4. Live demo workflow

1. Laptop 1 console visible (it is the evidence the guidelines require).
2. In the UI, select **`00000059`** first (1 031 fixes, ~0.9 s, top-1 97.6 %), then **Run evaluation**.
3. Point at the Laptop 1 console: `RUN id=… trip=00000059`, `DONE … top1=0.9758 … inference=… ms`, `POST /evaluation/run -> 200`.
4. In the UI: the map (amber GPS, cyan predicted route), metric cards, the **Effect of HMM/Viterbi** chart and the
   **Folium map from Laptop 1** tab.
5. Then **`00000015`** (largest: 6 854 fixes, ~2 s, HMM +10 pp over the classifier) and **`00000087`** (99.5 %).
6. History and the cross-trip chart fill up with each run. Any of the 10 trips works.

## 5. Results

### Official evaluator-test (10 trips, 19 674 fixes). Measured once on 2026-11-18 with the frozen model
`reports/evaluator/evaluator_results.{csv,json}`. Not used for any training, tuning or model choice.

| | nearest road | classifier only | **classifier + HMM** |
|---|---|---|---|
| accuracy (fix-weighted) | 0.5091 | 0.9166 | **0.9596** |
| macro-F1 (fix-weighted) | 0.7140 | 0.9520 | **0.9771** |
| route F1 (trip mean) | 0.6377 | 0.9268 | **0.9829** |
| sequence accuracy (trip mean) | 0.3072 | 0.7116 | **0.8930** |
| network-reachable continuity | 0.9215 | 0.9832 | **1.0000** |
| unreachable transitions (total) | 381 | 67 | **0** |

| trip | fixes | nearest | classifier | **HMM** | HMM macro-F1 | route F1 | seq. acc. | reachable cont. | inference ms | round trip ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 00000015 | 6,854 | 0.4164 | 0.8325 | **0.9329** | 0.9612 | 0.9052 | 0.5903 | 1.000 | 1,438 | 2,004 |
| 00000017 | 1,346 | 0.4562 | 0.9458 | **0.9629** | 0.9814 | 0.9873 | 0.9263 | 1.000 | 1,145 | 1,854 |
| 00000023 | 458 | 0.4039 | 0.9432 | **0.9520** | 0.9718 | 0.9761 | 0.6146 | 1.000 | 261 | 306 |
| 00000030 | 654 | 0.6131 | 0.8333 | **0.8884** | 0.9400 | 0.9751 | 0.9296 | 1.000 | 464 | 703 |
| 00000049 | 2,642 | 0.5700 | 0.9936 | **0.9958** | 0.9974 | 0.9994 | 0.9976 | 1.000 | 880 | 1,289 |
| 00000059 | 1,031 | 0.5170 | 0.9709 | **0.9758** | 0.9870 | 0.9944 | 0.9324 | 1.000 | 618 | 889 |
| 00000067 | 916 | 0.4891 | 0.9683 | **0.9716** | 0.9832 | 1.0000 | 1.0000 | 1.000 | 251 | 310 |
| 00000073 | 1,144 | 0.7281 | 0.9869 | **0.9886** | 0.9935 | 0.9990 | 0.9917 | 1.000 | 727 | 880 |
| 00000087 | 1,399 | 0.7598 | 0.9685 | **0.9950** | 0.9971 | 0.9977 | 0.9978 | 1.000 | 1,080 | 1,662 |
| 00000094 | 3,230 | 0.4889 | 0.9536 | **0.9669** | 0.9812 | 0.9945 | 0.9496 | 1.000 | 1,053 | 1,903 |

The HMM improved accuracy on all 10 trips (+0.2 to +10.0 pp). Exception worth knowing: on 00000015 its *strict*
continuity (81.8 %) is below classifier-only (85.9 %), although accuracy, route F1 and reachable continuity are higher.
Server memory peaked at 666 MB (the API keeps the last 10 runs for the `/map` endpoint).

### Student-test (10 trips, 19 160 fixes). Held-out development confirmation (`reports/RESULTS.md`)
classifier + HMM: accuracy **0.9155**, macro-F1 **0.9523** (classifier only 0.8926 / 0.9391; nearest road 0.5318 / 0.7239).

## 6. Backup: one-laptop demo (everything on Laptop 1)

```powershell
# window 1
cd "C:\Users\Raghav\Downloads\ML Hackathon"; $env:PATHSNAP_EVAL_DIR = "C:\Users\Raghav\Downloads\ML Hackathon\evaluator_test"; .\run_api.bat
# window 2
cd "C:\Users\Raghav\Downloads\ML Hackathon\frontend"; npm run preview
# browser
start "http://127.0.0.1:4173/?api=http://127.0.0.1:8000"
```
More fallbacks: Swagger `http://127.0.0.1:8000/docs` → `POST /evaluation/run` with `{"trip_id":"00000059"}`; screenshots
`reports/integration/ui_trip_000000*.png`; offline maps `reports/maps/match_student_test_*.html`.
Self-check with both servers up: `.venv\Scripts\python integration\e2e_browser.py --ui http://127.0.0.1:4173 --api http://127.0.0.1:8000 --trips 00000059,00000015` → `E2E PASSED`.

## 7. Troubleshooting

| problem | fix |
|---|---|
| Laptop 2: "Laptop 1 is unreachable" | same Wi-Fi? correct IP (`ipconfig` on Laptop 1)? firewall rule (§1)? venue Wi-Fi isolation → phone hotspot |
| CORS error in the browser console | server allows `*` by default; check `PATHSNAP_CORS_ORIGINS` is **not** set on Laptop 1. Fallback: UI proxy (`docs/TWO_LAPTOP_SETUP.md` §6) |
| `port … already in use` / `strictPort` error | §1 "free the demo ports" command (add 5173 if the dev server was used) |
| `CRITICAL REFUSING TO START` | `PATHSNAP_EVAL_DIR` path is wrong: fix it and restart |
| `/health` shows `evaluation_partition_is_fallback: true` | `PATHSNAP_EVAL_DIR` was not set in *that* window. Close the server, set it, restart |
| `status: degraded` / model not loaded | `artifacts\final\` must contain model.joblib, manifest.json, feature_schema.json, hmm_params.json |
| map background grey/blank in the UI | the UI's OSM tiles need internet; routes and GPS still draw. Laptop 1's Folium maps need no tiles. `net::ERR_ABORTED` tile entries in DevTools are normal (Leaflet cancels tiles when re-zooming) |
| everything slow by ~2 s | `localhost` was used. Use `127.0.0.1`/IP |
| `'run_api.bat' is not recognized` | in PowerShell always type `.\run_api.bat` |

### Restart both servers
```powershell
Get-NetTCPConnection -LocalPort 8000,4173,5173 -State Listen -ErrorAction SilentlyContinue | % { Stop-Process -Id $_.OwningProcess -Force }
# then repeat §1 (backend) and §2 / §6 (frontend)
```

## 8. Three-minute presentation script

**(0:00) Problem.** GPS fixes drift onto parallel streets and jump across buildings. PathSnap matches every fix to the road
segment actually driven and rebuilds the route. Laptop 1 runs the model and API; Laptop 2 is the evaluator UI.

**(0:20) Preprocessing.** We parse and validate the track, node and arc files, and sort and deduplicate timestamps. We
project lon/lat into a local metric plane and compute speed and heading. Training labels come from aligning each reference
route to its GPS fixes. They are joined only after the features are computed, and inference never opens a `.route` file
(we test this at runtime).

**(0:45) Candidates and 28 features.** A KD-tree over the road network returns up to 12 arcs within 50 m per fix; the true
arc is missing only ~0.2 % of the time. The 28 features describe distance, heading (most streets are two opposite one-way
arcs), speed, arc geometry, how consistent neighbouring fixes are, and whether network distance matches GPS movement.

**(1:10) Eight models.** Logistic regression, naive Bayes, random forest, extra trees, HistGB, LightGBM, XGBoost and an MLP,
with 5-fold cross-validation grouped by trip on TRAIN only. Nearest road gets 56 %; XGBoost is best at 91.3 % top-1 with the
best-calibrated probabilities.

**(1:30) Bayesian optimisation.** 22 Optuna trials on TRAIN. The gain was within noise. We kept our pre-declared rule and
report it honestly.

**(1:45) HMM/Viterbi.** Candidates are the hidden states. Emissions come from XGBoost; transitions compare network distance
with GPS displacement and penalise U-turns. Viterbi finds the best sequence and shortest paths fill the gaps. The decoder
was tuned only on out-of-fold TRAIN predictions.

**(2:05) Results.** On the held-out student-test trips the HMM gives 91.6 % accuracy and 0.952 macro-F1. On the **official
evaluator trips**, measured once with the frozen model: **96.0 % accuracy, macro-F1 0.977**, against 91.7 % for the classifier
alone and 50.9 % for nearest road. Route F1 is 0.983, there are zero impossible jumps, and every trip improved.

**(2:30) Live demo.** Select trip 00000059 → Run. Laptop 1's console logs the request, the map draws the predicted route,
and the metrics come from server-side scoring. The ground truth never leaves Laptop 1.
