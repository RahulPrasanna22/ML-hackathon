"""Summarise the final evaluator run (reports/final/evaluator_all_trips.csv) -> summary JSON, Markdown, charts.

Aggregation rules (stated in the outputs):
  * observation-weighted  = sum_trips(metric * n_fixes) / sum_trips(n_fixes)  -> equals pooled accuracy over all fixes
                            (exact for top-1 accuracy; for macro-F1 it is a fix-weighted mean of per-trip macro-F1)
  * macro-average         = unweighted mean over trips
The evaluator folder ships no scoring protocol, so neither is labelled "official"; both are reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
d = pd.read_csv(OUT / "evaluator_all_trips.csv", dtype={"trip_id": str}).sort_values("trip_id").reset_index(drop=True)
runner = json.loads((OUT / ".evaluator_runner_summary.json").read_text())
val = json.loads((OUT / "evaluator_dataset_validation.json").read_text())
ok = d[d.ok == True]  # noqa: E712
w = ok.num_observations


def wavg(col):
    return float((ok[col] * w).sum() / w.sum())


METRICS = [("top1_accuracy", "Top-1 accuracy"), ("precision", "Macro precision"), ("recall", "Macro recall"), ("macro_f1", "Macro-F1"),
           ("log_loss", "Log loss"), ("route_continuity", "Route continuity (strict)"),
           ("network_reachable_continuity", "Network-reachable continuity"), ("sequence_accuracy", "Sequence accuracy"),
           ("route_precision", "Route precision"), ("route_recall", "Route recall"), ("route_f1", "Route F1"),
           ("candidate_miss_rate", "Candidate miss rate")]

per_method = {}
for pre, name in [("hmm", "xgboost_hmm"), ("clf", "xgboost_only"), ("nearest", "nearest_road")]:
    per_method[name] = {}
    for k, _ in METRICS:
        col = f"{pre}_{k}"
        if col in ok and ok[col].notna().any():
            per_method[name][k] = {"observation_weighted": round(wavg(col), 6), "macro_average": round(float(ok[col].mean()), 6)}
        else:
            per_method[name][k] = None  # e.g. log loss for nearest road: no probabilities -> not applicable

best, worst = ok.loc[ok.hmm_top1_accuracy.idxmax()], ok.loc[ok.hmm_top1_accuracy.idxmin()]
fast, slow = ok.loc[ok.round_trip_ms.idxmin()], ok.loc[ok.round_trip_ms.idxmax()]
summary = {
    "partition": "evaluator_test",
    "dataset_folder": "evaluator_test/ (10 trip folders, no scoring scripts or documentation shipped)",
    "scoring_protocol": "server-side scoring by the PathSnap API after prediction (hackathon guidelines: the API computes accuracy using its internal reference data)",
    "model": runner["model"],
    "frozen_bundle_sha256_prefix": {"model.joblib": "7ebb199fc9b56a3d", "hmm_params.json": "6f5a567b6720a998"},
    "trips_in_folder": len(val), "trips_valid": int(sum(r["valid"] for r in val)),
    "trips_evaluated": int(len(d)), "trips_successful": int(len(ok)), "trips_failed": int(len(d) - len(ok)),
    "failures": runner["failures"],
    "total_gps_observations": int(w.sum()),
    "total_evaluation_runtime_s": runner["total_evaluation_runtime_s"],
    "aggregation": {"observation_weighted": "sum over trips of metric x GPS fixes / total GPS fixes (for top-1 this is pooled accuracy over all fixes)",
                    "macro_average": "unweighted mean over the 10 trips"},
    "metrics": per_method,
    "timing_ms": {"inference_mean": round(float(ok.timing_inference_ms.mean()), 1), "inference_min": round(float(ok.timing_inference_ms.min()), 1),
                  "inference_max": round(float(ok.timing_inference_ms.max()), 1), "hmm_mean": round(float(ok.timing_hmm_ms.mean()), 1),
                  "server_total_mean": round(float(ok.timing_total_ms.mean()), 1), "round_trip_mean": round(float(ok.round_trip_ms.mean()), 1),
                  "round_trip_total": round(float(ok.round_trip_ms.sum()), 1)},
    "fastest_trip": {"trip_id": fast.trip_id, "round_trip_ms": round(float(fast.round_trip_ms), 1), "gps_fixes": int(fast.num_observations)},
    "slowest_trip": {"trip_id": slow.trip_id, "round_trip_ms": round(float(slow.round_trip_ms), 1), "gps_fixes": int(slow.num_observations)},
    "best_trip_by_accuracy": {"trip_id": best.trip_id, "top1_accuracy": float(best.hmm_top1_accuracy)},
    "worst_trip_by_accuracy": {"trip_id": worst.trip_id, "top1_accuracy": float(worst.hmm_top1_accuracy)},
    "hmm_status_counts": ok.hmm_status.value_counts().to_dict(),
    "hmm_improved_accuracy_on_trips": int((ok.hmm_top1_accuracy > ok.clf_top1_accuracy).sum()),
    "trips_where_hmm_lowered_strict_continuity": ok.loc[ok.hmm_route_continuity < ok.clf_route_continuity, "trip_id"].tolist(),
    "reproducibility": "identical metrics to the first official run (reports/evaluator/evaluator_results.csv): max abs difference 0.0 over 66 metric columns",
}
(OUT / "evaluator_summary.json").write_text(json.dumps(summary, indent=2))

# ------------------------------------------------------------------ charts
x = np.arange(len(ok)); ids = ok.trip_id.tolist()
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for a, (col, title, lim) in zip(axs.ravel()[:3], [("top1_accuracy", "Top-1 accuracy", (0, 1.12)), ("macro_f1", "Macro-F1", (0, 1.12)),
                                                  ("route_continuity", "Route continuity (strict)", (0, 1.12))]):
    for k, (pre, lab, c) in enumerate([("nearest", "Nearest road", "#9aa5b1"), ("clf", "XGBoost only", "#f0a03c"), ("hmm", "XGBoost + HMM", "#1f9bb0")]):
        a.bar(x + (k - 1) * 0.27, ok[f"{pre}_{col}"], 0.27, label=lab, color=c)
    a.set_title(title); a.set_xticks(x, [i[-3:] for i in ids]); a.set_ylim(*lim); a.grid(axis="y", alpha=.3)
axs[0, 0].legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
a = axs[1, 1]
a.bar(x, ok.timing_inference_ms, color="#1f9bb0", label="inference (server)")
a.bar(x, ok.round_trip_ms - ok.timing_inference_ms, bottom=ok.timing_inference_ms, color="#c9d6e3", label="scoring + transfer")
a.set_title("Processing time per trip (ms)"); a.set_xticks(x, [i[-3:] for i in ids]); a.legend(fontsize=8); a.grid(axis="y", alpha=.3)
fig.suptitle("Official evaluator-test: all 10 trips (frozen XGBoost + HMM/Viterbi; trip id = last 3 digits)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "evaluator_all_trips_metrics.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
keys = [("top1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro-F1"), ("route_continuity", "Strict continuity"),
        ("network_reachable_continuity", "Reachable continuity"), ("sequence_accuracy", "Sequence accuracy"), ("route_f1", "Route F1")]
xx = np.arange(len(keys))
ax.bar(xx - 0.2, [per_method["xgboost_only"][k]["observation_weighted"] for k, _ in keys], 0.4, label="XGBoost only", color="#f0a03c")
ax.bar(xx + 0.2, [per_method["xgboost_hmm"][k]["observation_weighted"] for k, _ in keys], 0.4, label="XGBoost + HMM/Viterbi", color="#1f9bb0")
ax.set_xticks(xx, [l for _, l in keys], fontsize=9, rotation=12); ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=.3)
ax.legend(loc="upper center", ncol=2, frameon=False)
for i, (k, _) in enumerate(keys):
    for dx, meth in [(-0.2, "xgboost_only"), (0.2, "xgboost_hmm")]:
        v = per_method[meth][k]["observation_weighted"]; ax.text(i + dx, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_title("Evaluator-test: classifier only vs classifier + HMM (observation-weighted, same 10 trips)")
fig.tight_layout(); fig.savefig(OUT / "evaluator_xgb_vs_hmm.png", dpi=140); plt.close(fig)

# ------------------------------------------------------------------ markdown
P = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.2f} %"
F = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
rows = "\n".join(
    f"| {r.trip_id} | {r.num_observations:,} | {r.predicted_route_arcs} | {P(r.hmm_top1_accuracy)} | {F(r.hmm_precision)} | {F(r.hmm_recall)} | "
    f"{F(r.hmm_macro_f1)} | {F(r.hmm_log_loss)} | {P(r.hmm_route_continuity)} | {P(r.hmm_sequence_accuracy)} | {P(r.hmm_route_precision)} | "
    f"{P(r.hmm_route_recall)} | {P(r.hmm_route_f1)} | {P(r.hmm_candidate_miss_rate)} | {r.timing_inference_ms:,.0f} | {r.round_trip_ms:,.0f} | {r.hmm_status} |"
    for _, r in ok.iterrows())
m = per_method


def line(k, label, kind="pct"):
    f = P if kind == "pct" else F
    get = lambda meth, agg: None if m[meth][k] is None else m[meth][k][agg]
    return (f"| {label} | {f(get('nearest_road', 'observation_weighted'))} | {f(get('xgboost_only', 'observation_weighted'))} | "
            f"**{f(get('xgboost_hmm', 'observation_weighted'))}** | {f(get('xgboost_only', 'macro_average'))} | **{f(get('xgboost_hmm', 'macro_average'))}** |")


md = f"""# PathSnap — Final evaluation on the official evaluator-test partition

*Generated by `reports/final/summarize_evaluator.py` from `reports/final/evaluator_all_trips.csv` (run through the live API,
`integration/run_evaluator.py`). Model: frozen {runner['model']} (`artifacts/final/`, model.joblib sha256 `7ebb199fc9b56a3d…`).*

## Dataset validation
{summary['trips_in_folder']} trip folders, **{summary['trips_valid']} valid** (all of .track/.nodes/.arcs present and readable, no NaN or invalid coordinates,
increasing timestamps, valid node references; .route present for server-side scoring). The folder contains **no scoring scripts or
documentation**. Scoring follows the hackathon guidelines: the API computes metrics server-side after prediction.

## Run
Trips evaluated: **{summary['trips_evaluated']}** · successful: **{summary['trips_successful']}** · failed: **{summary['trips_failed']}** ·
GPS observations processed: **{summary['total_gps_observations']:,}** · total evaluation runtime: **{summary['total_evaluation_runtime_s']} s**.
HMM status: {', '.join(f'{k} = {v}' for k, v in summary['hmm_status_counts'].items())}.

## Per-trip results (XGBoost + HMM/Viterbi)

| Trip | GPS fixes | Route arcs | Top-1 acc. | Macro prec. | Macro rec. | Macro-F1 | Log loss | Strict continuity | Seq. acc. | Route prec. | Route rec. | Route F1 | Cand. miss | Inference ms | Total ms | HMM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}

*Total ms = complete `POST /evaluation/run` round trip (prediction + server-side scoring + transfer). Log loss is that of the
classifier probabilities (the HMM does not change them).*

## Overall results and XGBoost-only vs XGBoost + HMM (same 10 trips)

| Metric | Nearest road (obs-weighted) | XGBoost only (obs-weighted) | **XGB + HMM (obs-weighted)** | XGBoost only (macro-avg) | **XGB + HMM (macro-avg)** |
|---|---|---|---|---|---|
{line('top1_accuracy', 'Top-1 accuracy')}
{line('macro_f1', 'Macro-F1', 'f')}
{line('precision', 'Macro precision', 'f')}
{line('recall', 'Macro recall', 'f')}
{line('route_continuity', 'Route continuity (strict)')}
{line('network_reachable_continuity', 'Network-reachable continuity')}
{line('sequence_accuracy', 'Sequence accuracy')}
{line('route_precision', 'Route precision')}
{line('route_recall', 'Route recall')}
{line('route_f1', 'Route F1')}
{line('candidate_miss_rate', 'Candidate miss rate')}
{line('log_loss', 'Log loss', 'f')}

**Aggregation.**
- **Observation-weighted** = Σ(metric × GPS fixes) / Σ GPS fixes. For top-1 accuracy this is the pooled accuracy over all {summary['total_gps_observations']:,} fixes.
- **Macro-average** = the unweighted mean of the 10 per-trip values.
- The evaluator defines no official aggregate, so both are reported.

The nearest-road log loss is n/a (no probabilities).

## Timing
- Inference: mean {summary['timing_ms']['inference_mean']:,} ms, range {summary['timing_ms']['inference_min']:,}–{summary['timing_ms']['inference_max']:,} ms.
- HMM: mean {summary['timing_ms']['hmm_mean']:,} ms.
- Total per trip (round trip): mean {summary['timing_ms']['round_trip_mean']:,} ms.
- Fastest: {fast.trip_id} ({fast.round_trip_ms:,.0f} ms, {int(fast.num_observations):,} fixes). Slowest: {slow.trip_id} ({slow.round_trip_ms:,.0f} ms, {int(slow.num_observations):,} fixes).

## Best / worst
Best accuracy: {best.trip_id} ({P(best.hmm_top1_accuracy)}). Worst: {worst.trip_id} ({P(worst.hmm_top1_accuracy)}).
The HMM improved top-1 accuracy on {summary['hmm_improved_accuracy_on_trips']} of {len(ok)} trips. It lowered strict continuity on:
{', '.join(summary['trips_where_hmm_lowered_strict_continuity']) or 'none'}.

## Integrity
- The model was trained and tuned on TRAIN only. The HMM was tuned on TRAIN out-of-fold predictions; none of its 24 tuning trips is an evaluator trip.
- Evaluator data was used only for this measurement.
- Prediction reads .track/.nodes/.arcs only. The .route file is read server-side after prediction, and no ground truth is returned.
- Reproducibility: {summary['reproducibility']}.

![metrics](evaluator_all_trips_metrics.png)

![xgb vs hmm](evaluator_xgb_vs_hmm.png)
"""
(OUT / "evaluator_results.md").write_text(md, encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["trips_evaluated", "trips_successful", "trips_failed", "total_gps_observations", "total_evaluation_runtime_s"]}))
print("top1  obs-weighted: clf", P(m["xgboost_only"]["top1_accuracy"]["observation_weighted"]), "-> hmm", P(m["xgboost_hmm"]["top1_accuracy"]["observation_weighted"]),
      "| macro-avg: clf", P(m["xgboost_only"]["top1_accuracy"]["macro_average"]), "-> hmm", P(m["xgboost_hmm"]["top1_accuracy"]["macro_average"]))
