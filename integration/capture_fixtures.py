"""Capture REAL responses from a running Laptop 1 API into integration/fixtures/ for contract.test.ts.

Usage (API running):  .venv\\Scripts\\python integration\\capture_fixtures.py [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--trips", default="00000052,00000007")
args = ap.parse_args()
out = Path(__file__).resolve().parent / "fixtures"
out.mkdir(exist_ok=True)
c = httpx.Client(base_url=args.api, timeout=300)


def save(name: str, r: httpx.Response, expect: int):
    assert r.status_code == expect, (name, r.status_code, r.text[:300])
    (out / name).write_text(json.dumps(r.json(), indent=1))
    print(f"{name:<22} HTTP {r.status_code}  {len(r.content) / 1024:7.1f} KB")


save("health.json", c.get("/health"), 200)
save("trips.json", c.get("/evaluation/trips"), 200)
first = None
for tid in args.trips.split(","):
    r = c.post("/evaluation/run", json={"trip_id": tid})
    save(f"run_{tid}.json", r, 200)
    first = first or (tid, r.json()["evaluation_id"])
save(f"get_{first[0]}.json", c.get(f"/evaluation/{first[1]}"), 200)
save("err_404.json", c.post("/evaluation/run", json={"trip_id": "99999999"}), 404)
save("err_422.json", c.post("/evaluation/run", json={"trip_id": "../x"}), 422)
