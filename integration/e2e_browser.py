"""End-to-end browser test: real Edge -> Laptop 2 UI (Vite) -> Laptop 1 FastAPI -> trained model.

Prerequisites (two terminals):
  1. backend : .venv\\Scripts\\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
  2. frontend: cd frontend; npm run dev        (http://localhost:5173)
Then:        .venv\\Scripts\\python integration\\e2e_browser.py [--trips 00000052,00000007] [--ui URL] [--api URL]

For each trip it clicks the trip button and "Run evaluation", waits for the result and checks:
the route is drawn on the map, the displayed Top-1 / Macro-F1 / segment count equal the JSON the
browser received from the API, the history lists the run, there are no console errors and no
failing map-tile requests. Screenshots + summary go to reports/integration/. Uses the installed
Microsoft Edge via Playwright (channel "msedge"), so no browser download is needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ui", default="http://localhost:5173")
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--trips", default="00000052,00000007")
ap.add_argument("--channel", default="msedge")
args = ap.parse_args()
out = ROOT / "reports" / "integration"
out.mkdir(parents=True, exist_ok=True)
TILE_RE = re.compile(r"/\d+/\d+/\d+(@2x)?\.(png|jpg|jpeg|pbf)")

DRAWN_JS = """() => {
  let drawn = document.querySelectorAll('.leaflet-overlay-pane path').length;
  for (const c of document.querySelectorAll('.leaflet-overlay-pane canvas')) {
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) drawn++;
  }
  return drawn;
}"""


def pct(pattern: str, text: str):
    m = re.search(pattern + r"\s*\n?\s*([\d.]+)\s*%", text)
    return float(m.group(1)) if m else None


results, tiles, failed, console_errors, run_bodies, api_calls = [], [], [], [], {}, []
with sync_playwright() as p:
    browser = p.chromium.launch(channel=args.channel, headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1000})

    def on_response(r):
        if args.api in r.url:
            api_calls.append((r.request.method, r.url.replace(args.api, ""), r.status))
        if r.request.resource_type == "image" and TILE_RE.search(r.url):
            tiles.append((r.status, r.url))
        if r.request.method == "POST" and r.url.endswith("/evaluation/run") and r.status == 200:
            body = r.json()
            run_bodies[body["trip_id"]] = body

    page.on("response", on_response)
    page.on("requestfailed", lambda r: failed.append(r.url))
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(f"{args.ui}/?api={args.api}", wait_until="networkidle")
    page.get_by_role("radiogroup", name="Evaluation trips").wait_for(timeout=30_000)
    n_trip_buttons = page.locator("[data-testid^='trip-']").count()
    header = page.locator("header").inner_text()

    for tid in args.trips.split(","):
        t0 = time.time()
        page.get_by_test_id(f"trip-{tid}").click()
        page.locator("button.btn--run").click()
        page.locator("[data-testid='run-status'].runstatus--done, .runstatus--error, .runstatus--failed").first.wait_for(timeout=180_000)
        ui_s = time.time() - t0
        page.wait_for_timeout(1500)  # let Leaflet finish drawing
        status_text = page.get_by_test_id("run-status").inner_text()
        results_text = page.get_by_test_id("results").inner_text()
        body = run_bodies.get(tid, {})
        seg = re.search(r"Predicted road segments\s*\n?\s*([\d,]+)", results_text)
        shot = out / f"ui_trip_{tid}.png"
        page.screenshot(path=str(shot), full_page=True)
        results.append({
            "trip_id": tid, "ui_seconds": round(ui_s, 2), "status": status_text.strip()[:160],
            "model_in_response": body.get("model", {}).get("name"),
            "drawn_vector_elements_or_pixels": page.evaluate(DRAWN_JS),
            "api_top1_pct": round(100 * body["metrics"]["top1_accuracy"], 1) if body else None,
            "ui_top1_pct": pct(r"Top-1 accuracy", results_text),
            "api_macro_f1_pct": round(100 * body["metrics"]["macro_f1"], 1) if body else None,
            "ui_macro_f1_pct": pct(r"Macro F1-score", results_text),
            "api_route_segments": len(body["predicted_route"]["segments"]) if body else None,
            "ui_route_segments": int(seg.group(1).replace(",", "")) if seg else None,
            "screenshot": str(shot.relative_to(ROOT)),
        })
        print(json.dumps(results[-1]), flush=True)

    history_text = page.get_by_test_id("history").inner_text()
    browser.close()

history_hits = [r["trip_id"] for r in results if r["trip_id"] in history_text]
checks = {
    "trip_buttons_loaded": n_trip_buttons > 0,
    "routes_drawn": all(r["drawn_vector_elements_or_pixels"] > 0 for r in results),
    "top1_matches_api": all(r["ui_top1_pct"] is not None and r["ui_top1_pct"] == r["api_top1_pct"] for r in results),
    "macro_f1_matches_api": all(r["ui_macro_f1_pct"] is not None and r["ui_macro_f1_pct"] == r["api_macro_f1_pct"] for r in results),
    "route_segments_match_api": all(r["ui_route_segments"] == r["api_route_segments"] for r in results),
    "history_lists_all_runs": len(history_hits) == len(results),
    "no_console_errors": not console_errors,
    "no_tile_errors": all(s < 400 for s, _ in tiles),
}
summary = {"ui": args.ui, "api": args.api, "trip_buttons": n_trip_buttons, "header": header.replace("\n", " | ")[:200],
           "runs": results, "checks": checks, "api_calls": api_calls, "history_trips_listed": history_hits,
           "console_errors": console_errors[:10], "tile_requests": len(tiles),
           "tile_status_counts": {str(k): sum(1 for s, _ in tiles if s == k) for k in {s for s, _ in tiles}},
           "failed_requests": failed[:10]}
(out / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(checks, indent=1))
print("tiles:", summary["tile_status_counts"], "| failed requests:", len(failed), "| API calls:", len(api_calls))
ok = all(checks.values())
print("E2E", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
