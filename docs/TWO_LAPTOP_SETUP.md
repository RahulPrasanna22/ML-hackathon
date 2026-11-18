# Two-laptop setup and demo runbook

| Laptop | Runs | Default port |
|---|---|---|
| Laptop 1 | FastAPI backend (model, HMM/Viterbi, server-side evaluation) | 8000 |
| Laptop 2 | PathSnap evaluator UI (`frontend/`) | 5173 (dev) / 4173 (preview) |

All commands are Windows PowerShell.

## 1. Same network

Connect both laptops to the same Wi-Fi or phone hotspot. Venue Wi-Fi sometimes blocks device-to-device traffic ("client isolation"). If Laptop 2 cannot reach Laptop 1 even with the firewall open, switch to a phone hotspot.

## 2. Find Laptop 1's IP address (on Laptop 1)

```powershell
ipconfig
```

Use the **IPv4 Address** under the active adapter (`Wireless LAN adapter Wi-Fi`), e.g. `192.168.1.20`. Ignore `vEthernet`/WSL/VirtualBox adapters.

A shorter version:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -like '*Wi-Fi*' } | Select-Object IPAddress
```

Run the same command on Laptop 2 to learn its IP for the CORS setting (step 4).

## 3. Open the firewall port (on Laptop 1, PowerShell as Administrator)

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
```

If Laptop 2 serves the UI to other devices, open 5173 / 4173 on Laptop 2 the same way.

## 4. Start the backend (on Laptop 1)

The backend must listen on all interfaces and allow the UI's origin:

```powershell
$env:PATHSNAP_CORS_ORIGINS = "http://localhost:5173,http://192.168.1.35:5173,http://localhost:4173,http://192.168.1.35:4173"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

(Replace `192.168.1.35` with Laptop 2's IP and `app.main:app` with the real module path.) See `docs/API_CONTRACT.md` for the CORS middleware snippet.

Check from Laptop 2:

```powershell
Invoke-RestMethod http://192.168.1.20:8000/health
Invoke-RestMethod http://192.168.1.20:8000/evaluation/trips
```

## 5. Point the UI at Laptop 1 (on Laptop 2)

Pick one. The UI uses the first one that is set.

1. **In the UI:** click **Connection**, enter `http://192.168.1.20:8000`, click **Test connection**, then **Save and connect**. This is remembered in this browser. No rebuild needed.
2. **Link parameter:** open `http://localhost:5173/?api=http://192.168.1.20:8000`.
3. **Environment file:** create `frontend/.env.local` with `VITE_API_BASE_URL=http://192.168.1.20:8000`, then restart `npm run dev` (or rebuild for `preview`).

## 6. If CORS gets in the way: proxy fallback

Symptoms: `/health` works in PowerShell (`Invoke-RestMethod`), but the UI says *"Could not reach … CORS settings …"*. The browser console shows `blocked by CORS policy`.

Best fix: add the UI's exact origin (scheme + host + port, as shown in the Connection panel) to the backend's CORS list.

Fallback without touching the backend: let Laptop 2's Vite server forward requests. In `frontend/.env.local`:

```ini
VITE_API_BASE_URL=/api
PATHSNAP_PROXY_TARGET=http://192.168.1.20:8000
```

Then restart `npm run dev` (or `npm run build; npm run preview`). The browser then talks only to Laptop 2, which forwards `/api/*` to Laptop 1. Requests still hit Laptop 1's endpoints and appear in its uvicorn logs.

## 7. Demo script for the evaluator

1. Laptop 1: backend running in a visible terminal with `--log-level info`.
2. Laptop 2: `cd frontend; npm run preview` (or `npm run dev`), open the URL in the browser.
3. Header shows **Laptop 1 API · Connected** and the overview shows **Evaluation trips: 10**.
4. The evaluator clicks any trip, then **Run evaluation on <trip>**.
5. Point at Laptop 1's terminal: `POST /evaluation/run … 200` appears.
6. The UI draws the raw GPS observations (amber), the predicted route (cyan) and the road network (slate), and shows Top-1 accuracy, Macro F1, route continuity, log loss, inference time and HMM status. Use the layer control (top-right of the map) to toggle layers.
7. Repeat for other trips. The session history and the cross-trip chart fill in with each run.

## Troubleshooting

| Message in the UI | Likely cause | Fix |
|---|---|---|
| *Not configured* | No API address set | Step 5 |
| *Laptop 1 is unreachable* | Backend down, wrong IP, firewall, different network, or CORS | Steps 2 to 4 and 6. Test with `Invoke-RestMethod` from Laptop 2. |
| *Request timed out* | Backend busy or very slow trip | Raise `VITE_EVALUATION_TIMEOUT_MS`. Check Laptop 1's logs. |
| *Unexpected response from Laptop 1* | JSON doesn't match `docs/API_CONTRACT.md` | The message names the offending field |
| *Not found on Laptop 1* | Unknown trip ID | Check the backend's evaluation-trip list |
| *Laptop 1 reported a failure* | Pipeline returned `status: "failed"` | See the `error` text and Laptop 1's logs |
