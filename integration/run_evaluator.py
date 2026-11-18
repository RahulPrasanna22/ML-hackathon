"""Official evaluator run through the LIVE API (same path the UI uses).

For each trip: POST /evaluation/run, then record the server-side metrics, timings, server memory
(RSS of the process listening on the API port) and contract checks. The frozen model is never
touched. The ground truth is read only inside the server, after prediction. This script receives
only aggregate metrics and the predicted route, and saves metrics only (no geometry).

Usage (backend running with PATHSNAP_EVAL_DIR set):
    .venv\\Scripts\\python integration\\run_evaluator.py [--trips 00000059,00000015,...] [--api http://127.0.0.1:8000]
Output: reports/evaluator/evaluator_results.{json,csv}
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import httpx
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://127.0.0.1:8000")
ap.add_argument("--trips", default=None, help="comma-separated order (default: all, API order)")
ap.add_argument("--out-dir", default=str(ROOT / "reports" / "evaluator"), help="output folder (existing results elsewhere are not touched)")
ap.add_argument("--csv-name", default="evaluator_results.csv")
ap.add_argument("--json-name", default="evaluator_results.json")
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

c = httpx.Client(base_url=args.api, timeout=600)
health = c.get("/health").json()
assert health["evaluation_partition"] == "evaluator_test" and health["evaluation_partition_is_fallback"] is False, health
trips = [t["trip_id"] for t in c.get("/evaluation/trips").json()["trips"]]
order = args.trips.split(",") if args.trips else trips
port = int(args.api.rsplit(":", 1)[1].split("/")[0])
server = next((psutil.Process(k.pid) for k in psutil.net_connections("tcp")
               if k.laddr and k.laddr.port == port and k.status == psutil.CONN_LISTEN), None)


def rss_mb():
    try:
        return server.memory_info().rss / 2**20 if server else None
    except psutil.Error:
        return None


def contract_problems(b: dict, tid: str) -> list[str]:
    p = []
    if b.get("trip_id") != tid: p.append("trip_id mismatch")
    if b.get("status") != "completed": p.append(f"status={b.get('status')}")
    if "reference_route" in b or "label_arc" in json.dumps(b): p.append("ground truth present in response")
    obs = b.get("observations", [])
    if len(obs) != b.get("num_observations"): p.append("observations count mismatch")
    if any(not (-180 <= o["lon"] <= 180 and -90 <= o["lat"] <= 90) for o in obs): p.append("observation out of range")
    pr = b.get("predicted_route", {})
    if not pr.get("arc_ids") or len(pr["arc_ids"]) != len(pr.get("segments", [])): p.append("route arc/segment mismatch")
    for s in pr.get("segments", []):
        if len(s["coordinates"]) != 2 or not all(math.isfinite(v) and abs(v) <= 180 for xy in s["coordinates"] for v in xy):
            p.append("invalid segment geometry"); break
    for k, v in b.get("metrics", {}).items():
        if v is not None and not math.isfinite(v): p.append(f"non-finite metric {k}")
    for k in ["top1_accuracy", "macro_f1", "precision", "recall", "route_continuity", "sequence_accuracy",
              "candidate_miss_rate", "network_reachable_continuity"]:
        v = b["metrics"].get(k)
        if v is None or not 0 <= v <= 1: p.append(f"{k} not in [0,1]")
    if not all(k.endswith("_ms") for k in b.get("timing", {})): p.append("timing keys not *_ms")
    return p


rows = []
T_START = time.perf_counter()
print(f"server pid={server.pid if server else '?'} rss at start={rss_mb():.0f} MB | trips: {order}")
for tid in order:
    rss0 = rss_mb()
    t0 = time.perf_counter()
    try:
        r = c.post("/evaluation/run", json={"trip_id": tid})
        wall = 1000 * (time.perf_counter() - t0)
        if r.status_code != 200:
            rows.append({"trip_id": tid, "ok": False, "http": r.status_code, "error": r.text[:300], "round_trip_ms": wall})
            print(f"{tid}: HTTP {r.status_code} {r.text[:200]}"); continue
        b = r.json()
    except Exception as e:  # network / timeout
        rows.append({"trip_id": tid, "ok": False, "error": repr(e), "round_trip_ms": 1000 * (time.perf_counter() - t0)})
        print(f"{tid}: FAILED {e!r}"); continue
    m, cm, nm, tm = b["metrics"], b["classifier_only_metrics"], b["nearest_road_metrics"], b["timing"]
    probs = contract_problems(b, tid)
    row = {"trip_id": tid, "ok": not probs, "http": 200, "evaluation_id": b["evaluation_id"],
           "num_observations": b["num_observations"], "predicted_route_arcs": len(b["predicted_route"]["arc_ids"]),
           "response_kb": round(len(r.content) / 1024, 1), "round_trip_ms": round(wall, 1),
           **{f"timing_{k}": v for k, v in tm.items()},
           **{f"hmm_{k}": m[k] for k in m}, **{f"clf_{k}": cm[k] for k in cm}, **{f"nearest_{k}": nm[k] for k in nm},
           "hmm_status": b["hmm"]["status"], "warnings": " | ".join(b["warnings"]), "contract_problems": "; ".join(probs),
           "server_rss_before_mb": rss0, "server_rss_after_mb": rss_mb()}
    rows.append(row)
    print(f"{tid}: n={b['num_observations']:5d} top1={m['top1_accuracy']:.4f} (clf {cm['top1_accuracy']:.4f}, nearest "
          f"{nm['top1_accuracy']:.4f}) macroF1={m['macro_f1']:.4f} routeF1={m['route_f1']:.4f} seqAcc={m['sequence_accuracy']:.4f} "
          f"| infer {tm['inference_ms']:.0f} ms, eval {tm['evaluation_ms']:.0f} ms, round trip {wall:.0f} ms "
          f"| rss {row['server_rss_after_mb']:.0f} MB | contract {'OK' if not probs else probs}", flush=True)

df = pd.DataFrame(rows)
TOTAL_S = time.perf_counter() - T_START
df.to_csv(out / args.csv_name, index=False)
ok = df[df.ok == True]  # noqa: E712
summary = {"partition": "evaluator_test", "model": health["model_name"], "n_trips_requested": len(order),
           "n_trips_ok": int(len(ok)), "total_evaluation_runtime_s": round(TOTAL_S, 2), "failures": df[df.ok != True][["trip_id"] + [c for c in ["http", "error", "contract_problems"] if c in df]].to_dict("records")}  # noqa: E712
if len(ok):
    w = ok.num_observations
    for pre, name in [("hmm", "classifier_hmm"), ("clf", "classifier_only"), ("nearest", "nearest_road")]:
        summary[name] = {"accuracy_fix_weighted": float((ok[f"{pre}_top1_accuracy"] * w).sum() / w.sum()),
                         "macro_f1_fix_weighted": float((ok[f"{pre}_macro_f1"] * w).sum() / w.sum()),
                         **{f"{k}_trip_mean": float(ok[f"{pre}_{k}"].mean()) for k in
                            ["top1_accuracy", "macro_f1", "route_continuity", "network_reachable_continuity",
                             "sequence_accuracy", "route_f1", "normalized_edit_distance", "mean_deviation_m"]}}
    summary["total_observations"] = int(w.sum())
    summary["timing"] = {"inference_ms_mean": float(ok.timing_inference_ms.mean()), "inference_ms_max": float(ok.timing_inference_ms.max()),
                         "evaluation_ms_mean": float(ok.timing_evaluation_ms.mean()), "round_trip_ms_max": float(ok.round_trip_ms.max()),
                         "hmm_ms_mean": float(ok.timing_hmm_ms.mean())}
    summary["server_rss_peak_mb"] = float(df.server_rss_after_mb.max())
(out / args.json_name).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
