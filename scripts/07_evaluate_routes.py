"""Phase 9: end-to-end route evaluation on the STUDENT-TEST partition (and Folium maps).

Runs the exact production pipeline (MapMatcher.load() -> match_trip) on each trip from raw
files, then evaluates separately against the reference route:
  A. nearest-road baseline   B. classifier-only   C. classifier + HMM/Viterbi (+ route filling)

Outputs reports/experiments/route_eval_student_test.csv (+ _summary.json) and
reports/maps/match_<trip>.html.
Usage: python scripts/07_evaluate_routes.py [--partition student_test] [--maps]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pathsnap import config, io
from pathsnap.evaluation import evaluate_prediction
from pathsnap.pipeline import MapMatcher
from pathsnap.viz import match_map

ap = argparse.ArgumentParser()
ap.add_argument("--partition", default=config.STUDENT_TEST_PARTITION, choices=[config.STUDENT_TEST_PARTITION, config.TRAIN_PARTITION])
ap.add_argument("--maps", action="store_true")
args = ap.parse_args()

matcher = MapMatcher.load()
rows = []
for tid in io.list_trips(args.partition):
    trip = io.load_trip(args.partition, tid)
    t0 = time.perf_counter()
    res = matcher.match_trip(trip)                 # prediction: no reference data involved
    wall = time.perf_counter() - t0
    ref = io.load_reference_route(args.partition, tid)  # evaluation only, after prediction
    ev = evaluate_prediction(res, ref, reliable_m=30.0 if args.partition == "train" else None)
    row = {"trip_id": tid, "n_obs": ev["n_obs"], "n_obs_evaluated": ev["n_obs_evaluated"],
           "candidate_miss_rate": ev["candidate_miss_rate"], "latency_s": wall,
           "ms_per_obs": 1000 * wall / ev["n_obs"], **{f"t_{k}": v for k, v in res["timings"].items()}}
    for meth in ["nearest_road", "classifier_only", "classifier_hmm"]:
        row.update({f"{meth}__{k}": v for k, v in ev[meth].items()})
    rows.append(row)
    print(f"{tid}: A={ev['nearest_road']['accuracy']:.4f} B={ev['classifier_only']['accuracy']:.4f} "
          f"C={ev['classifier_hmm']['accuracy']:.4f} | routeF1 B={ev['classifier_only']['route_f1_len']:.3f} "
          f"C={ev['classifier_hmm']['route_f1_len']:.3f} | {wall:.2f}s", flush=True)
    if args.maps:
        config.MAP_DIR.mkdir(parents=True, exist_ok=True)
        c = ev["classifier_hmm"]
        title = f"Trip {tid} ({args.partition}) - HMM acc {c['accuracy']:.3f}, classifier acc {ev['classifier_only']['accuracy']:.3f}"
        match_map(res, ref, title).save(str(config.MAP_DIR / f"match_{args.partition}_{tid}.html"))

df = pd.DataFrame(rows)
out = config.REPORT_DIR / "experiments"
df.to_csv(out / f"route_eval_{args.partition}.csv", index=False)
summary = {"partition": args.partition, "n_trips": len(df),
           "observation_weighted": {}, "trip_mean": {}}
w = df.n_obs_evaluated
for meth in ["nearest_road", "classifier_only", "classifier_hmm"]:
    keys = ["accuracy", "accuracy_undirected", "macro_f1", "continuity", "reachable_continuity", "sequence_accuracy",
            "route_precision_len", "route_recall_len", "route_f1_len", "route_jaccard_arcs", "norm_edit_distance",
            "edit_distance", "mean_deviation_m"]
    summary["observation_weighted"][meth] = {k: float(np.average(df[f"{meth}__{k}"], weights=w)) for k in keys[:3]}
    summary["trip_mean"][meth] = {k: float(df[f"{meth}__{k}"].mean()) for k in keys}
    summary["trip_mean"][meth]["invalid_transitions_total"] = int(df[f"{meth}__invalid_transitions"].sum())
    summary["trip_mean"][meth]["unreachable_transitions_total"] = int(df[f"{meth}__unreachable_transitions"].sum())
summary["latency"] = {"mean_s_per_trip": float(df.latency_s.mean()), "max_s_per_trip": float(df.latency_s.max()),
                      "mean_ms_per_obs": float(df.ms_per_obs.mean())}
summary["candidate_miss_rate_mean"] = float(df.candidate_miss_rate.mean())
summary["hmm_breaks_total"] = int(df["classifier_hmm__n_breaks"].sum())
summary["hmm_route_unfilled_gaps_total"] = int(df["classifier_hmm__route_unfilled_gaps"].sum())
(out / f"route_eval_{args.partition}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
