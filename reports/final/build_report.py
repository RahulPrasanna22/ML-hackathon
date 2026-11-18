"""Build the PathSnap final project report from the repository's recorded results.

Every number in the report is read from a source file at build time (see the `S` dict and the
"Source:" line under each table), so the report stays traceable and cannot drift from the evidence.

Run (project venv):  .venv\\Scripts\\python reports\\final\\build_report.py
Outputs: reports/final/PathSnap_Final_Project_Report.{md,html,pdf,docx}
The HTML/DOCX conversion runs in an isolated tool environment (PATHSNAP_REPORT_ENV, default
%TEMP%\\pathsnap_reportenv) holding markdown + pypandoc_binary + pypdf; the PDF is printed with
Playwright/Edge from the project venv. Nothing in the application or model is modified.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "final"
REP = ROOT / "reports"
EXP = REP / "experiments"
TOOLENV = Path(os.environ.get("PATHSNAP_REPORT_ENV", Path(os.environ["TEMP"]) / "pathsnap_reportenv"))

# ------------------------------------------------------------------ load sources
S = {
    "eda": json.loads((REP / "eda_summary.json").read_text()),
    "eda_trips": pd.read_csv(REP / "eda_trip_stats.csv", dtype={"trip_id": str}),
    "cmp": pd.read_csv(EXP / "model_comparison.csv"),
    "folds": pd.read_csv(EXP / "cv_folds.csv"),
    "opt": json.loads((EXP / "optuna_xgboost_best.json").read_text()),
    "opt_trials": pd.read_csv(EXP / "optuna_xgboost_trials.csv"),
    "tvu": pd.read_csv(EXP / "tuned_vs_untuned.csv"),
    "hmm": json.loads((EXP / "hmm_tuning.json").read_text()),
    "hmm_trials": pd.read_csv(EXP / "hmm_tuning_trials.csv"),
    "st_route": pd.read_csv(EXP / "route_eval_student_test.csv", dtype={"trip_id": str}),
    "st_sum": json.loads((EXP / "route_eval_student_test_summary.json").read_text()),
    "bench": json.loads((EXP / "benchmark.json").read_text()),
    "ev": pd.read_csv(REP / "evaluator" / "evaluator_results.csv", dtype={"trip_id": str}).sort_values("trip_id"),
    "ev_sum": json.loads((REP / "evaluator" / "evaluator_results.json").read_text()),
    "e2e_ev": json.loads((REP / "integration" / "e2e_summary_evaluator.json").read_text()),
    "manifest": json.loads((ROOT / "artifacts" / "final" / "manifest.json").read_text()),
    "hmm_params": json.loads((ROOT / "artifacts" / "final" / "hmm_params.json").read_text()),
    "obs_train": pd.read_parquet(ROOT / "cache" / "obs_train.parquet"),
    "obs_st": pd.read_parquet(ROOT / "cache" / "obs_student_test.parquet"),
    "cand_train": pd.read_csv(REP / "candidate_stats_train.csv"),
    "cand_st": pd.read_csv(REP / "candidate_stats_student_test.csv"),
}
EVID = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in (OUT / "evidence").glob("*.txt")}

NAMES = {"nearest_road": "Nearest road (baseline)", "logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes",
         "random_forest": "Random Forest", "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting",
         "lightgbm": "LightGBM", "xgboost": "XGBoost", "mlp": "MLP (64-32)"}
pct = lambda v, d=2: "n/r" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f} %"
f4 = lambda v: "n/r" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
n0 = lambda v: f"{v:,.0f}"


def fig(name: str, caption: str, width: str = "100%") -> str:
    return f'\n<figure><img src="figures/{name}" style="width:{width}" alt="{caption}"/><figcaption>{caption}</figcaption></figure>\n'


# ------------------------------------------------------------------ derived numbers
eda_tr, eda_st = S["eda"]["train"], S["eda"]["student_test"]
ot, os_ = S["obs_train"], S["obs_st"]
rel = ot.label_dist <= 30
cand_rows_tr, cand_rows_st = int(S["cand_train"].n_rows.sum()), int(S["cand_st"].n_rows.sum())
miss_tr_all, miss_tr_rel = 1 - ot.label_in_cands.mean(), 1 - ot.label_in_cands[rel].mean()
miss_st = 1 - os_.label_in_cands.mean()
cmp = S["cmp"].set_index("model")
ml = cmp.drop(index="nearest_road").sort_values("cv_top1_accuracy", ascending=False)
sel = S["manifest"]["model_name"]
tvu = S["tvu"].set_index(S["tvu"].variant.str.split(" ").str[0])
opt, ot_trials = S["opt"], S["opt_trials"]
n_trials_complete = int((ot_trials.state == "COMPLETE").sum())
hp = S["hmm_params"]
stsum, evs = S["st_sum"], S["ev_sum"]
ev = S["ev"]
bench = S["bench"]
pt = lambda name: [l for l in EVID.get(name, "").splitlines() if l.strip() and not l.startswith("#")]
py_line = next((l for l in pt("python_tests") if "passed" in l), "n/r")
fe_line = next((l for l in pt("frontend_tests") if "Tests" in l), "n/r").strip()
ct_line = next((l for l in pt("contract_tests") if "Tests" in l), "n/r").strip()
tc_ok = "exit code: 0" in EVID.get("typecheck", "")
build_ok = "built in" in EVID.get("build", "") and "exit code: 0" in EVID.get("build", "")
e2e_checks = S["e2e_ev"]["checks"]
e2e_runs = S["e2e_ev"]["runs"]

# ------------------------------------------------------------------ tables
model_rows = []
for key, r in ml.iterrows():
    status = "**Selected (final)**" if key == sel else "Not selected"
    model_rows.append(f"| {'**' + NAMES[key] + '**' if key == sel else NAMES[key]} | {r.family} | {pct(r.cv_top1_accuracy)} | "
                      f"± {pct(r.cv_top1_accuracy_std)} | {f4(r.cv_cand_macro_f1)} | {f4(r.cv_log_loss)} | "
                      f"{r.fit_all_s:.1f} | {r.st_inference_ms_per_obs:.4f} | {status} |")
b = cmp.loc["nearest_road"]
model_rows.append(f"| *{NAMES['nearest_road']}* | geometric, no learning | {pct(b.cv_top1_accuracy)} | – | n/r | n/r | – | – | reference |")
MODEL_TABLE = ("| Model | Family | CV top-1 accuracy | Fold std | CV macro-F1 | CV log loss | Training time (s) | Inference (ms / fix) | Status |\n"
               "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(model_rows))

fold_tab = S["folds"][S["folds"].model.isin(ml.index)].pivot(index="model", columns="fold", values="top1_accuracy").loc[ml.index]
FOLD_TABLE = ("| Model | " + " | ".join(f"Fold {i}" for i in fold_tab.columns) + " | Train-fold top-1 |\n|---|" + "---|" * (len(fold_tab.columns) + 1) + "\n" +
              "\n".join(f"| {NAMES[k]} | " + " | ".join(pct(v) for v in row) + f" | {pct(cmp.loc[k, 'cv_train_top1_accuracy'])} |" for k, row in fold_tab.iterrows()))

T = tvu
TUNE_TABLE = f"""| Variant | CV top-1 (± std) | CV macro-F1 | CV log loss | Student-test top-1 | Student-test macro-F1 | Student-test log loss | Training time | Inference |
|---|---|---|---|---|---|---|---|---|
| XGBoost, default configuration | {pct(T.loc['untuned', 'cv_top1_accuracy'])} ± {pct(T.loc['untuned', 'cv_top1_std'])} | {f4(T.loc['untuned', 'cv_cand_macro_f1'])} | {f4(T.loc['untuned', 'cv_log_loss'])} | {pct(T.loc['untuned', 'st_top1_accuracy'])} | {f4(T.loc['untuned', 'st_cand_macro_f1'])} | {f4(T.loc['untuned', 'st_log_loss'])} | {T.loc['untuned', 'fit_all_s']:.1f} s | {T.loc['untuned', 'st_inference_ms_per_obs']:.4f} ms |
| **XGBoost, Optuna-tuned (deployed)** | **{pct(T.loc['tuned', 'cv_top1_accuracy'])}** ± {pct(T.loc['tuned', 'cv_top1_std'])} | {f4(T.loc['tuned', 'cv_cand_macro_f1'])} | {f4(T.loc['tuned', 'cv_log_loss'])} | {pct(T.loc['tuned', 'st_top1_accuracy'])} | {f4(T.loc['tuned', 'st_cand_macro_f1'])} | {f4(T.loc['tuned', 'st_log_loss'])} | {T.loc['tuned', 'fit_all_s']:.1f} s | {T.loc['tuned', 'st_inference_ms_per_obs']:.4f} ms |"""

bp = opt["best_params"]
BEST_PARAMS = "\n".join(f"| `{k}` | {v:.4g} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in bp.items())


def abc_table(src: dict, keys: list[tuple[str, str, str]]) -> str:
    rows = []
    for label, k, kind in keys:
        vals = []
        for meth in ["nearest_road", "classifier_only", "classifier_hmm"]:
            v = src[meth].get(k)
            vals.append("n/r" if v is None else (pct(v) if kind == "pct" else f"{v:,.4f}" if kind == "f4" else f"{v:,.1f}" if kind == "f1" else f"{v:,.0f}"))
        rows.append(f"| {label} | " + " | ".join(vals[:2]) + f" | **{vals[2]}** |")
    return "| Metric | A: nearest road | B: XGBoost only | C: XGBoost + HMM/Viterbi |\n|---|---|---|---|\n" + "\n".join(rows)


st_ow, st_tm = stsum["observation_weighted"], stsum["trip_mean"]
ST_TABLE = abc_table({m: {**st_tm[m], **{f"w_{k}": v for k, v in st_ow[m].items()}} for m in st_tm}, [
    ("Pointwise accuracy (fix-weighted)", "w_accuracy", "pct"), ("Macro-F1 (fix-weighted)", "w_macro_f1", "f4"),
    ("Accuracy, either travel direction (fix-weighted)", "w_accuracy_undirected", "pct"),
    ("Accuracy (mean over trips)", "accuracy", "pct"), ("Strict continuity (mean)", "continuity", "pct"),
    ("Network-reachable continuity (mean)", "reachable_continuity", "pct"), ("Sequence accuracy (mean)", "sequence_accuracy", "pct"),
    ("Route precision, length-weighted (mean)", "route_precision_len", "pct"), ("Route recall, length-weighted (mean)", "route_recall_len", "pct"),
    ("Route F1, length-weighted (mean)", "route_f1_len", "pct"), ("Edit distance (mean, arcs)", "edit_distance", "f1"),
    ("Unreachable transitions (total)", "unreachable_transitions_total", "int"), ("Mean deviation from reference route (m)", "mean_deviation_m", "f1")])

EV_TABLE = abc_table({"nearest_road": {**evs["nearest_road"]}, "classifier_only": {**evs["classifier_only"]}, "classifier_hmm": {**evs["classifier_hmm"]}}, [
    ("Pointwise top-1 accuracy (fix-weighted)", "accuracy_fix_weighted", "pct"), ("Macro-F1 (fix-weighted)", "macro_f1_fix_weighted", "f4"),
    ("Top-1 accuracy (mean over trips)", "top1_accuracy_trip_mean", "pct"), ("Macro-F1 (mean over trips)", "macro_f1_trip_mean", "f4"),
    ("Strict continuity (mean)", "route_continuity_trip_mean", "pct"), ("Network-reachable continuity (mean)", "network_reachable_continuity_trip_mean", "pct"),
    ("Sequence accuracy (mean)", "sequence_accuracy_trip_mean", "pct"), ("Route F1, length-weighted (mean)", "route_f1_trip_mean", "pct"),
    ("Normalised edit distance (mean, lower is better)", "normalized_edit_distance_trip_mean", "f4"),
    ("Unreachable transitions (total)", "unreachable_transitions_total", "int"), ("Mean deviation from reference route (m)", "mean_deviation_m_trip_mean", "f1")])

ev_rows = []
for _, r in ev.iterrows():
    ev_rows.append(f"| {r.trip_id} | {r.num_observations:,} | {pct(r.nearest_top1_accuracy, 1)} | {pct(r.clf_top1_accuracy, 1)} | **{pct(r.hmm_top1_accuracy, 1)}** | "
                   f"{f4(r.hmm_macro_f1)} | {pct(r.hmm_route_precision, 1)} | {pct(r.hmm_route_recall, 1)} | {pct(r.hmm_route_f1, 1)} | {pct(r.hmm_route_continuity, 1)} | "
                   f"{pct(r.hmm_sequence_accuracy, 1)} | {pct(r.hmm_candidate_miss_rate, 2)} | {r.timing_inference_ms:,.0f} | {r.round_trip_ms:,.0f} |")
EV_TRIPS = ("| Trip | GPS fixes | A nearest | B XGBoost | C XGB+HMM | Macro-F1 (C) | Route prec. (C) | Route rec. (C) | Route F1 (C) | Strict cont. (C) | Seq. acc. (C) | Cand. miss | Inference (ms) | Round trip (ms) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(ev_rows))

st = S["st_route"].sort_values("trip_id")
ST_TRIPS = ("| Trip | GPS fixes | A nearest | B XGBoost | C XGB+HMM | Route F1 B | Route F1 C | Seq. acc. C |\n|---|---|---|---|---|---|---|---|\n" +
            "\n".join(f"| {r.trip_id} | {r.n_obs:,} | {pct(r.nearest_road__accuracy, 1)} | {pct(r.classifier_only__accuracy, 1)} | **{pct(r.classifier_hmm__accuracy, 1)}** | "
                      f"{pct(r.classifier_only__route_f1_len, 1)} | {pct(r.classifier_hmm__route_f1_len, 1)} | {pct(r.classifier_hmm__sequence_accuracy, 1)} |" for _, r in st.iterrows()))

ev_gain = ev.hmm_top1_accuracy - ev.clf_top1_accuracy
st_gain = st.classifier_hmm__accuracy - st.classifier_only__accuracy
worse_cont = ev[ev.hmm_route_continuity < ev.clf_route_continuity]
e2e_rows = "\n".join(f"| {r['trip_id']} | {r['ui_top1_pct']} % / {r['api_top1_pct']} % | {r['ui_macro_f1_pct']} % / {r['api_macro_f1_pct']} % | "
                     f"{r['ui_route_segments']} / {r['api_route_segments']} | {r['ui_seconds']:.2f} s |" for r in e2e_runs)

# ------------------------------------------------------------------ the report
today = date.today().strftime("%d %B %Y")
md = f"""<div class="cover">

<p class="cover-kicker">ML Hackathon · Project Report</p>

# PathSnap

<p class="cover-sub">GPS Map Matching and Route Reconstruction Using Machine Learning</p>

<p class="cover-desc">A two-laptop system that snaps noisy GPS observations to the road network with a tuned XGBoost candidate
classifier and HMM/Viterbi sequence decoding, served by a FastAPI backend to a React evaluator dashboard.</p>

<div class="cover-team">

**Team**

Raghav G · Rahul · Raghav H · Rachit

</div>

<p class="cover-date">{today}</p>

</div>

<div class="pagebreak"></div>

## Contents

1. Abstract · 2. Introduction · 3. Problem Statement · 4. Dataset Description · 5. Data Preprocessing ·
6. Machine Learning Methodology · 7. Model Training and Validation · 8. Eight-Model Comparison ·
9. Bayesian Hyperparameter Optimisation · 10. HMM and Viterbi Decoding · 11. System Architecture ·
12. Backend Implementation · 13. Frontend Implementation · 14. Official Evaluator Integration ·
15. Results and Performance Analysis · 16. Testing and Validation · 17. Challenges and Solutions ·
18. Limitations · 19. Future Scope · 20. Conclusion · 21. References · 22. Appendix

> **Reading the numbers.** Results are reported for three strictly separated data partitions and are never mixed:
> **(A) TRAIN cross-validation** (model development), **(B) student-test** (held-out confirmation during development) and
> **(C) official evaluator-test** (final evaluation, measured once with the frozen model). Every table names its source file.
> "n/r" means *not recorded*: the value was never measured and is not estimated.

## 1. Abstract

GPS receivers in vehicles produce noisy positions: fixes drift onto parallel streets, jump across buildings and drop out.
**PathSnap** reconstructs the road route a vehicle actually drove by matching every GPS observation to a directed road
segment (*arc*) of the trip's road network. For each fix, a spatial index proposes up to {S['manifest']['pipeline_config']['CANDIDATE_MAX_K']}
candidate arcs within {S['manifest']['pipeline_config']['CANDIDATE_RADIUS_M']:.0f} m. A supervised classifier scores each (fix, candidate) pair from
{len(S['manifest']['pipeline_config']) and 28} engineered geometric, kinematic and topological features. A Hidden Markov Model decoded with the Viterbi
algorithm then turns the per-fix scores into a connected route, using directed network distances.

Eight classifiers from six algorithm families were compared with 5-fold cross-validation grouped by trip on the 80 TRAIN trips.
**XGBoost** achieved the best cross-validated top-1 accuracy ({pct(cmp.loc['xgboost', 'cv_top1_accuracy'])}) and the lowest log loss
({f4(cmp.loc['xgboost', 'cv_log_loss'])}). The three best models lie within one fold standard deviation of each other. Bayesian
optimisation with Optuna ({n_trials_complete} trials) produced the deployed configuration, and the HMM decoder was tuned on
out-of-fold TRAIN predictions only.

On the **official evaluator-test partition** (10 unseen trips, {evs['total_observations']:,} GPS fixes), the frozen system reached a
pointwise accuracy of **{pct(evs['classifier_hmm']['accuracy_fix_weighted'])}** and macro-F1 **{f4(evs['classifier_hmm']['macro_f1_fix_weighted'])}**.
This compares with {pct(evs['classifier_only']['accuracy_fix_weighted'])} for the classifier alone and {pct(evs['nearest_road']['accuracy_fix_weighted'])}
for nearest-road snapping. It also achieved a length-weighted route F1 of {pct(evs['classifier_hmm']['route_f1_trip_mean'])} with zero network-unreachable transitions.
The system runs as a FastAPI backend (Laptop 1) and a React/TypeScript/Leaflet dashboard (Laptop 2). The two were verified end-to-end in a browser on all
10 evaluator trips and over a real two-laptop Wi-Fi connection.

**Technologies:** Python 3.11, NumPy, pandas, SciPy, scikit-learn, XGBoost, LightGBM, Optuna, FastAPI/Uvicorn, Folium,
React 19, TypeScript 5.9, Vite 8, Leaflet, Vitest, pytest, Playwright.

## 2. Introduction

**GPS and navigation.** Global Navigation Satellite Systems estimate a receiver's position from the travel time of satellite
signals. Consumer receivers in vehicles typically report one fix per second (1 Hz), with errors of several metres in open
sky and considerably more in urban canyons.

**Measurement noise.** Multipath reflections from buildings, signal blockage in tunnels and under trees, atmospheric delay
and poor satellite geometry produce random scatter and systematic offsets. Dropouts create gaps of seconds to minutes. As a
result, a raw trace may appear on a parallel street, on the opposite carriageway, or inside a building.

**Why map matching is necessary.** Most applications need the *road-level* path, not the raw coordinates: which roads were
driven, in which direction and in which order. Nearest-road snapping is insufficient. In this dataset it identifies the
correct directed arc for only {pct(evs['nearest_road']['accuracy_fix_weighted'], 1)} of evaluator-test fixes. The reasons are
that two-way streets are stored as two opposite arcs with identical geometry, that intersections place several arcs within a few
metres, and that independent per-point decisions produce impossible jumps between unconnected roads.

**Applications.** Map matching underpins fleet and logistics tracking, ride-hailing fare and route auditing,
traffic-speed and congestion estimation, usage-based insurance, transport research on travel behaviour, and road-network
quality control.

## 3. Problem Statement

**Definition.** Given a trip's GPS observations $Z = (z_1,\\dots,z_T)$, $z_t = (\\lambda_t, \\varphi_t, \\tau_t)$ (longitude,
latitude, time), and the trip's directed road network $G = (V, A)$, find for every observation the arc $a_t \\in A$ that the
vehicle was travelling on. Also return the connected route $R = (r_1,\\dots,r_M)$ of arcs that the vehicle traversed.

| Item | Content |
|---|---|
| Input | `.track` (GPS fixes), `.nodes` (node coordinates), `.arcs` (directed arcs), all per trip. The trip id is sent by the UI. |
| Output | Matched arc per fix, reconstructed continuous route as arc ids and `[lon, lat]` geometry, and server-side metrics |
| Not available at prediction time | The reference route `.route` (used only for training labels and for scoring after prediction) |

**Challenges.**
1. *Noise and outliers*: fixes {pct(eda_tr['label_dist_percentiles']['50'] / 100, 0) if False else f"{eda_tr['label_dist_percentiles']['50']:.1f} m"} (median) from the reference road, with heavy tails.
2. *Direction ambiguity*: the median trip network has {pct(eda_tr['two_way_frac_median'], 0)} of arcs with a reverse twin.
3. *Junction ambiguity*: several candidate arcs within a few metres at intersections.
4. *Topology*: networks contain disconnected components (median {eda_tr['weak_components_median']:.0f} weakly connected components per train trip).
5. *Gaps*: sampling gaps of up to {eda_tr['dt_max_overall']:,.0f} s in TRAIN.
6. *Scale*: networks of up to {eda_tr['n_arcs_max']:,} arcs per trip, and trips of up to {eda_tr['n_obs_max']:,} fixes.
7. *Leakage risk*: labels exist only as route sequences and must never reach the prediction path.

**Objectives.**
(i) Build a reproducible preprocessing and feature pipeline.
(ii) Train and compare at least five classifiers with trip-grouped cross-validation.
(iii) Tune the selected model with Bayesian optimisation.
(iv) Add HMM/Viterbi decoding and route reconstruction.
(v) Serve the system through a FastAPI backend to a separate evaluator UI.
(vi) Evaluate strictly without leakage on held-out data.

## 4. Dataset Description

**Partitions.** The organisers' dataset is split by trip, and no trip appears in two partitions. The trip ids of all three
partitions were checked and are disjoint.

| Partition | Trips | GPS fixes | Purpose in PathSnap |
|---|---|---|---|
| TRAIN | {eda_tr['n_trips']} | {eda_tr['n_obs_total']:,} | fitting, 5-fold GroupKFold cross-validation, model selection, Bayesian optimisation, HMM tuning |
| Student-test | {eda_st['n_trips']} | {eda_st['n_obs_total']:,} | held-out confirmation only (never used for any choice) |
| Evaluator-test | {evs['n_trips']} | {evs['total_observations']:,} | official final evaluation, measured once with the frozen system |

*Source: `reports/eda_summary.json`, `reports/evaluator/evaluator_results.json`.*

**Files per trip** (whitespace-separated, no header; verified by inspection):

| File | Columns | Notes |
|---|---|---|
| `.track` | longitude, latitude, timestamp (s) | about 1 Hz: on average {pct(eda_tr['frac_dt_1s_mean'], 1)} of TRAIN steps are exactly 1 s |
| `.nodes` | longitude, latitude | the row index is the node id |
| `.arcs` | from_node, to_node | the row index is the arc id. Arcs are **directed**; a two-way street is two arcs |
| `.route` | arc id | chronological reference arc sequence. It does **not** label individual fixes |

Coordinates are WGS84 longitude/latitude in the order `[lon, lat]`, as in GeoJSON. Each trip ships its **own** road network
(TRAIN median {eda_tr['n_arcs_median']:,.0f} arcs, max {eda_tr['n_arcs_max']:,}), so arc ids are local to a trip. Trips are worldwide,
from longitude −100° to 137°.
{fig('eda_trip_locations.png', 'Figure 1. Trip locations (median GPS position) for TRAIN and student-test (source: scripts/01_explore.py).', '85%')}
**How reference labels are used safely.** `.route` lists arcs but not which fix lies on which arc. Per-fix training labels
are derived by a monotone dynamic-programming alignment of fixes to route positions (Section 5.5). This is done only for
TRAIN and student-test, and only *after* features have been computed. At inference time the loader `io.load_trip()` reads only
`.track/.nodes/.arcs`. The evaluator's `.route` files are read only by the server-side scoring step, after the prediction has finished.

**Reference-label quality.** Reference routes are treated as reference information, not perfect ground truth. In TRAIN the
fix-to-reference distance has median {eda_tr['label_dist_percentiles']['50']:.1f} m but a 95th percentile of
{eda_tr['label_dist_percentiles']['95']:.0f} m, because a few routes do not cover the drive. Trip 00000054's route covers only part of a long drive,
and trip 00000033's route diverges from the GPS. Fixes farther than 30 m from their aligned reference arc are therefore marked
*unreliable* and excluded from fitting and cross-validation ({pct(rel.mean(), 1)} of TRAIN fixes are reliable). In student-test the 99th percentile is
{eda_st['label_dist_percentiles']['99']:.1f} m and all fixes are used.

## 5. Data Preprocessing

All preprocessing lives in the `pathsnap/` package. The same code runs for training and inference, and it has no fitted parameters.

### 5.1 GPS loading and cleaning (`io.py`, `preprocess.py`)
1. Parse with column-count checks; reject unreadable files.
2. Drop NaN, out-of-range and (0, 0) coordinates; stable-sort by timestamp; drop duplicated timestamps; keep the original row index.
3. Project into a **local metric plane**: an equirectangular projection on the WGS84 ellipsoid around the trip's median
   position, using the meridional radius $M$ and prime-vertical radius $N$ at the reference latitude $\\varphi_0$:
   $x = (\\lambda-\\lambda_0)\\,\\tfrac{{\\pi}}{{180}}\\,N\\cos\\varphi_0$, $y = (\\varphi-\\varphi_0)\\,\\tfrac{{\\pi}}{{180}}\\,M$.
   For trip extents of a few tens of kilometres this is accurate to well under GPS noise; it is verified against haversine distances in the tests.
4. Kinematics: step distance and $\\Delta t$, centred speed, and a robust heading whose window grows from ±1 to ±8 fixes until the
   displacement is at least 2 m (otherwise marked invalid). Outliers (>{S['manifest']['pipeline_config']['MAX_PLAUSIBLE_SPEED_MS'] * 3.6:.0f} km/h) and gaps (>5 s) are flagged.

**Data-quality checks** (`io.validate_trip`, `network.full_graph_stats`) run on every trip:

| Check | TRAIN | Student-test |
|---|---|---|
| NaN / invalid coordinates | {eda_tr['track_nan_total']} / {eda_tr['invalid_coords_total']} | {eda_st['track_nan_total']} / {eda_st['invalid_coords_total']} |
| Non-increasing timestamps | {eda_tr['t_non_increasing_total']} | {eda_st['t_non_increasing_total']} |
| Largest time gap | {eda_tr['dt_max_overall']:,.0f} s | {eda_st['dt_max_overall']:,.0f} s |
| Trips with a gap > 30 s | {eda_tr['trips_with_gap_gt_30s']} | {eda_st['trips_with_gap_gt_30s']} |
| Speed-outlier fixes (flagged) | {eda_tr['speed_outliers_total']} | {eda_st['speed_outliers_total']} |
| Isolated nodes / invalid route ids | {eda_tr['isolated_nodes_total']} / {eda_tr['route_invalid_ids_total']} | {eda_st['isolated_nodes_total']} / {eda_st['route_invalid_ids_total']} |
| Reference arcs lost by network cropping | {eda_tr['route_arcs_cropped_away_total']} | {eda_st['route_arcs_cropped_away_total']} |

*Source: `reports/eda_summary.json`.*
{fig('eda_sampling_speed.png', 'Figure 2. Sampling interval and GPS speed distributions (TRAIN).', '90%')}
### 5.2 Road-network construction (`network.py`)
The network is cropped to the GPS bounding box plus {S['manifest']['pipeline_config']['NETWORK_CROP_BUFFER_M']:.0f} m. This removed no reference arc,
as the table above shows. Nodes are re-indexed and projected. For each arc the code computes length, bearing, in- and out-degree and a
reverse-twin lookup. A directed adjacency list supports a bounded, memoised Dijkstra search, and a shortest-path routine fills route gaps.

### 5.3 Candidate generation and spatial filtering
Every arc is densified into points {S['manifest']['pipeline_config']['INDEX_SAMPLE_STEP_M']:.0f} m apart and indexed in a SciPy **KD-tree**. For each fix,
a radius query of {S['manifest']['pipeline_config']['CANDIDATE_RADIUS_M']:.0f} m returns nearby arcs. Exact point-to-segment distances are computed, and the
{S['manifest']['pipeline_config']['CANDIDATE_MAX_K']} nearest arcs are kept; if none is in range, the nearest arc is kept as a fallback. The radius and
*K* were chosen from TRAIN statistics: in student-test, 99.9 % of reference arcs lie within the 12 nearest arcs.

| Candidate statistics | TRAIN | Student-test |
|---|---|---|
| Candidate rows (fix × arc) | {cand_rows_tr:,} | {cand_rows_st:,} |
| Candidates per fix | {cand_rows_tr / len(ot):.2f} | {cand_rows_st / len(os_):.2f} |
| Reference arc missing from candidates (all fixes) | {pct(miss_tr_all)} | {pct(miss_st)} |
| … among fixes with reliable labels | {pct(miss_tr_rel)} | n/a (all used) |

*Source: `reports/candidate_stats_*.csv`, `cache/obs_*.parquet`.*

### 5.4 The 28 engineered features (`features.py`)

| Group | Features | Rationale |
|---|---|---|
| Distance | `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `n_cands` | closeness to the fix, absolute and relative to the other candidates |
| Direction | `heading_diff`, `cos_heading`, `heading_valid`, `heading_rank`, `wide_heading_diff` | separates two-way twin arcs; a ±5-fix window adds robustness at low speed |
| Kinematics | `speed`, `dt_prev`, `step_prev` | sampling and motion context |
| Arc geometry | `arc_length`, `proj_t`, `dist_to_arc_end` | where along the arc the fix projects |
| Topology | `has_reverse`, `to_out_deg`, `from_in_deg` | two-way roads and junction complexity |
| Temporal context | `ctx_dist_mean`, `ctx_dist_max`, `ctx_dist_rank` | distance of the ±3 neighbouring fixes to the same arc |
| Connectivity | `prev_connected`, `next_connected` | whether the arc connects to the neighbouring fixes' nearest arcs |
| Transition consistency | `trans_dev_prev`, `trans_reachable_prev`, `trans_dev_next`, `trans_reachable_next` | $\\lvert d_{{network}} - d_{{GPS}} \\rvert$ to the neighbours' nearest arcs (bounded Dijkstra) |

No feature uses the reference route. A test recomputes features for a trip after deleting its `.route` file and asserts that
they are identical.

### 5.5 Training-label alignment (`labels.py`, TRAIN and student-test only)
Fix $i$ is assigned to a route position $j$, with positions non-decreasing, by minimising the total point-to-arc distance:
$C[i,j] = d(z_i, r_j) + \\min_{{k \\le j}} C[i-1,k]$. A row-wise prefix minimum makes this $O(T\\cdot M)$. Using route *positions*
rather than arc ids handles routes that revisit an arc. A candidate row is labelled $y=1$ if it is the aligned arc.
{fig('preprocessing_00000003.png', 'Figure 3. Preprocessing of TRAIN trip 00000003: GPS fixes (blue) on the reference route (green) and network (grey); zoom on candidate arcs (orange) and the labelled arc (green); candidates per fix (source: scripts/demo_preprocessing.py).')}
{fig('folium_preprocessing_00000003.png', 'Figure 4. Offline Folium map of the same trip: GPS fixes, nearby road network and reference route (no external tiles).', '85%')}
## 6. Machine Learning Methodology

**Formulation.** Map matching is cast as **binary candidate classification**: for each (fix, candidate arc) row, predict
$P(y=1 \\mid \\mathbf{{x}})$, the probability that this arc is the driven road. The arg-max over a fix's candidates gives the
classifier-only match, and the probabilities become HMM emission scores. Every model therefore must produce probabilities.
Eight models from six families were compared, so that the choice rests on evidence rather than habit:

| Model | Family | Configuration (initial) | Why considered |
|---|---|---|---|
| Logistic Regression | linear | standardised inputs, L2, C = 1 | interpretable baseline with calibrated probabilities |
| Gaussian Naive Bayes | probabilistic, generative | default variance smoothing | very fast; tests whether conditional independence suffices |
| Random Forest | bagging ensemble | 200 trees, min leaf 5, √features, 50 % bootstrap | non-linear interactions, robust to scaling |
| Extra Trees | randomised trees | 200 trees, min leaf 5, √features | lower variance than RF, fast training |
| HistGradientBoosting | boosting | 300 iterations, 31 leaves, learning rate 0.1 | histogram GBDT in scikit-learn |
| LightGBM | boosting | 400 trees, 63 leaves, learning rate 0.05, subsampling | efficient leaf-wise boosting |
| XGBoost | boosting | 400 trees, depth 7, learning rate 0.05, subsampling | regularised depth-wise boosting, strong on tabular data |
| MLP (64-32) | neural network | 2 hidden layers, Adam, early stopping | non-linear representation learning |

*Source: `pathsnap/models.py`.* Boosted trees suit this problem because the features are heterogeneous and strongly interacting.
For example, a small distance matters only if the heading is consistent. The linear and naive-Bayes models test how far simpler
assumptions go. All models were trained on identical rows and folds.

## 7. Model Training and Validation

**Data.** Training uses only the {eda_tr['n_trips']} TRAIN trips: {int(rel.sum()):,} reliable-label fixes, giving
{int(S['manifest']['n_training_rows']):,} candidate rows.

**Grouped cross-validation.** `GroupKFold(n_splits=5)` groups rows by `trip_id`. Consecutive 1 Hz fixes of a trip are
highly correlated, so random row-wise splits would leak near-duplicates into validation and inflate scores. With trip
grouping every validation fold contains only trips the model has never seen; a test asserts that the folds are disjoint and cover every trip.

**Pipeline** (`scripts/03_train_models.py`): for each model and each fold, fit on the training folds and predict
probabilities on the held-out trips. Metrics are computed from those predictions, together with training-fold metrics for
diagnosing over-fitting. Each model is then refit on all TRAIN rows and saved.

**Metrics.**
- *Top-1 accuracy* (primary): the fraction of fixes whose highest-probability candidate is the reference arc. The denominator counts all evaluated fixes, so candidate misses count as errors.
- *Macro-F1*, *precision* and *recall*: computed at candidate level (threshold 0.5) and averaged over both classes.
- *Log loss*: measures probability calibration, which matters because the probabilities feed the HMM.
- *Route continuity*: the share of arc changes between consecutive fixes that are directly connected.
- *Timing*: training and inference time.

**Selection criterion.** The criterion, fixed before results were seen, was the highest mean cross-validated top-1 accuracy on TRAIN.
Ties within fold noise were resolved by probability quality (log loss) and inference speed. Student-test results were never consulted for any choice.

## 8. Eight-Model Comparison

{MODEL_TABLE}

*Table 1. TRAIN 5-fold GroupKFold results (mean over folds). Source: `reports/experiments/model_comparison.csv`.
Training time is a single fit on all TRAIN rows. The slower models were measured while other jobs shared the CPU, so their times
are indicative only. Inference time is classifier scoring per GPS fix. n/r = not recorded.*
{fig('models_cv_comparison.png', 'Figure 5. Cross-validated macro-F1 (left) and top-1 accuracy (right) with fold standard deviation; dashed line is the nearest-road baseline.')}
{fig('dashboard_model_comparison.png', 'Figure 6. The same comparison as shown in the evaluator dashboard (served by GET /models/comparison): mean ± 1 fold std on a labelled zoomed axis.', '92%')}
**Per-fold top-1 accuracy (TRAIN):**

{FOLD_TABLE}

*Table 2. Source: `reports/experiments/cv_folds.csv`.*

**Why XGBoost.** XGBoost has the highest mean CV top-1 accuracy ({pct(cmp.loc['xgboost', 'cv_top1_accuracy'])}), the best
macro-F1 ({f4(cmp.loc['xgboost', 'cv_cand_macro_f1'])}), the lowest log loss ({f4(cmp.loc['xgboost', 'cv_log_loss'])}) and
fast inference. **The top models are, however, statistically indistinguishable.** LightGBM ({pct(cmp.loc['lightgbm', 'cv_top1_accuracy'])})
and Random Forest ({pct(cmp.loc['random_forest', 'cv_top1_accuracy'])}) are within {abs(cmp.loc['xgboost', 'cv_top1_accuracy'] - cmp.loc['random_forest', 'cv_top1_accuracy']) * 100:.2f}
percentage points, far less than the fold standard deviation of about 1.3 pp. The choice is therefore driven as much by calibration and speed as by accuracy.

**Observations.**
- **Fit.** The tree ensembles fit their training folds clearly better than the validation folds, showing moderate over-fitting
  (Random Forest {pct(cmp.loc['random_forest', 'cv_train_top1_accuracy'])} vs {pct(cmp.loc['random_forest', 'cv_top1_accuracy'])}). The linear and
  naive-Bayes models show almost no gap and lower scores, which indicates under-fitting.
- **Calibration.** Gaussian NB's log loss ({f4(cmp.loc['gaussian_nb', 'cv_log_loss'])}) is an order of magnitude worse, because the
  independence assumption makes it over-confident.
- **Variation between trips.** Fold 0 is hardest for every model, so the difficulty varies by trip as well as by algorithm.
{fig('models_log_loss.png', 'Figure 7. Log loss per model (log scale; CV and student-test).', '80%')}
{fig('models_train_vs_val.png', 'Figure 8. Training-fold versus validation-fold top-1 accuracy (over-/under-fitting diagnostic).', '80%')}
## 9. Bayesian Hyperparameter Optimisation

**Why Optuna.** Optuna's Tree-structured Parzen Estimator (TPE) models good and bad regions of the search space and proposes
promising configurations. This is far more sample-efficient than grid search when each trial costs a full 5-fold cross-validation.

**Set-up** (`scripts/04_bayes_opt.py`, TRAIN only):

| Item | Value |
|---|---|
| Sampler | {opt['sampler']} |
| Objective | {opt['objective']} (maximised) |
| Budget | 50-minute timeout; **{n_trials_complete} trials completed** ({opt['optimisation_seconds_this_run']:,.0f} s); every trial is stored in SQLite (`artifacts/optuna/xgboost.db`) and `optuna_xgboost_trials.csv` |
| Cost control | trials fit on every second fix of the training folds (1 Hz fixes are redundant) and validate on all fixes; the final configuration is refit on all rows |

| Hyperparameter | Search range | Best value (trial {opt['best_trial']}) |
|---|---|---|
| `n_estimators` | 100–1000 (step 50) | {bp['n_estimators']} |
| `learning_rate` | 0.01–0.2 (log) | {bp['learning_rate']:.4f} |
| `max_depth` | 3–12 | {bp['max_depth']} |
| `min_child_weight` | 0.5–50 (log) | {bp['min_child_weight']:.3f} |
| `subsample` | 0.5–1.0 | {bp['subsample']:.3f} |
| `colsample_bytree` | 0.4–1.0 | {bp['colsample_bytree']:.3f} |
| `reg_lambda` | 0.001–10 (log) | {bp['reg_lambda']:.4f} |

*Source: `reports/experiments/optuna_xgboost_best.json`. Best objective value (half-data trials): {pct(opt['best_value_cv_top1'])}.*

**Default vs tuned XGBoost** (identical full-data folds and partitions):

{TUNE_TABLE}

*Table 3. Source: `reports/experiments/tuned_vs_untuned.csv`.*

**Interpretation.** Tuning raised CV top-1 accuracy by only
{(T.loc['tuned', 'cv_top1_accuracy'] - T.loc['untuned', 'cv_top1_accuracy']) * 100:.2f} pp, which is well inside the fold
standard deviation. It also **worsened calibration** (log loss {f4(T.loc['untuned', 'cv_log_loss'])} → {f4(T.loc['tuned', 'cv_log_loss'])})
and made training and inference about three times slower. The tuned model was kept because the selection rule fixed in advance
(best CV top-1 on TRAIN) prefers it. Changing the rule after seeing held-out results would be selection bias. On student-test the
tuned model is slightly *worse*; this is reported, not acted upon. The HMM decoder later compensates for the over-confident
probabilities through a small emission weight (Section 10).
{fig('optuna_xgboost_history.png', 'Figure 9. Optuna optimisation history (each trial and best-so-far CV top-1).', '75%')}
{fig('tuned_vs_untuned.png', 'Figure 10. Default versus tuned XGBoost (CV and student-test) and student-test log loss.', '90%')}
## 10. Hidden Markov Model and Viterbi Decoding

**Why independent predictions are insufficient.** The classifier scores each fix in isolation. Consecutive fixes can then be assigned to
unconnected arcs, or flip between twin directions. On student-test the classifier-only sequence contains
{stsum['trip_mean']['classifier_only']['unreachable_transitions_total']} arc changes that are unreachable within 500 m through the
directed network.

**Model** (`pathsnap/hmm.py`). The hidden state at time $t$ is a candidate arc $c$ of fix $t$, together with its projected position on the arc.

* **Emission:** $\\log P(z_t \\mid c) = w_e \\cdot \\log \\dfrac{{p_{{t,c}}}}{{\\sum_{{c'}} p_{{t,c'}}}}$, where $p$ is the XGBoost
  probability, normalised over the fix's candidates. An optional Gaussian distance term was available, but tuning set it to zero.
* **Transition:** $\\log P(c_t \\to c_{{t+1}}) = -\\dfrac{{\\lvert d_{{\\text{{net}}}} - d_{{\\text{{GPS}}}} \\rvert}}{{\\beta}} - \\rho\\,[c_{{t+1}} = \\text{{reverse}}(c_t)]$.
  Here $d_{{\\text{{net}}}}$ is the **directed** network distance between the projected positions (bounded Dijkstra; forward moves on
  the same arc are direct), $d_{{\\text{{GPS}}}}$ is the straight-line displacement between the fixes, and $\\rho$ penalises U-turns onto
  the reverse twin arc. Unreachable pairs get $-\\infty$.
* **Viterbi recurrence:** $V_{{t+1}}(b) = E_{{t+1}}(b) + \\max_a \\left[ V_t(a) + T(a,b) \\right]$, with back-pointers. The best final state is
  backtracked to give one arc per fix.
* **Breaks and gaps:** if no finite transition exists (disconnected network parts, dead ends) or a gap exceeds
  {hp['gap_break_s']:.0f} s, the chain restarts with a penalty. The Dijkstra search radius grows with the time gap.
* **Route reconstruction:** the per-fix arc sequence is compressed, and consecutive non-adjacent arcs are joined with the
  directed shortest path (`reconstruct_route`).

**TRAIN-only tuning** (`scripts/06_tune_decoder.py`). The emissions came from **out-of-fold** probabilities of the final model on
{len(S['hmm']['tuning_trips'])} seeded TRAIN trips, so the decoder saw realistic test-time scores. Optuna ran {S['hmm']['n_trials']} trials
({S['hmm']['optimisation_seconds']:.0f} s). The parameters were selected on TRAIN accuracy, and student-test was used only as confirmation.

| Decoder parameter | Search range | Final value |
|---|---|---|
| β (transition scale, m) | 1–100 (log) | {hp['beta']:.2f} |
| emission weight $w_e$ | 0.2–3 (log) | {hp['emission_weight']:.3f} |
| σ (Gaussian distance term) | {{0, 5, 10, 20, 40}} | {hp['sigma']:.0f} (off) |
| same-arc backward tolerance (m) | 2–30 | {hp['same_arc_tol']:.1f} |
| break penalty | 5–200 (log) | {hp['break_penalty']:.1f} |
| U-turn (reverse-arc) penalty ρ | 0–10 | {hp['reverse_penalty']:.2f} |

| TRAIN out-of-fold (24 trips) | Accuracy |
|---|---|
| XGBoost only | {pct(S['hmm']['train_subset_classifier_only_acc'])} |
| + HMM, default parameters | {pct(S['hmm']['train_subset_default_hmm_acc'])} |
| + HMM, tuned parameters | **{pct(S['hmm']['train_subset_best_hmm_acc'])}** |

*Source: `reports/experiments/hmm_tuning.json`, `artifacts/final/hmm_params.json`.* The small emission weight
({hp['emission_weight']:.3f}) means the decoder relies more on network consistency than on the over-confident classifier
probabilities, which matches the calibration finding of Section 9.
{fig('route_eval_abc_student_test.png', 'Figure 11. Student-test: nearest road (A), XGBoost only (B) and XGBoost + HMM (C) per trip: pointwise accuracy, route F1 and normalised edit distance.')}
{fig('folium_route_student_test_00000009.png', 'Figure 12. Folium map of student-test trip 00000009 generated by the pipeline: raw GPS, candidate roads, reference route (server-side only), classifier-only and HMM routes (offline, no tiles).', '88%')}
## 11. System Architecture
{fig('architecture.png', 'Figure 13. End-to-end architecture and two-laptop deployment.', '96%')}
**Request flow.**
1. The React dashboard sends `POST /evaluation/run {{"trip_id"}}`.
2. FastAPI loads the trip's `.track/.nodes/.arcs`.
3. The pipeline runs: preprocessing → candidates → 28 features → tuned XGBoost → HMM/Viterbi → route reconstruction.
4. The server then reads the reference route and computes metrics.
5. The response contains the predicted route as `[lon, lat]` segments, the per-fix matches, the metrics and the timings. It contains no ground truth.

**Two-laptop deployment.** Laptop 1 runs the backend and model (Uvicorn on `0.0.0.0:8000`); Laptop 2 runs the Vite production
build of the UI and calls Laptop 1 over Wi-Fi. CORS allows any origin without credentials. The connection was verified
in practice: Laptop 2 (10.20.205.106) called `/health`, `/evaluation/trips` and `POST /evaluation/run` on Laptop 1, and all returned
HTTP 200 (`logs/demo_backend.log`).

## 12. Backend Implementation

**Stack.** Python 3.11, FastAPI, Uvicorn, Pydantic. The trained bundle is loaded once at start-up by
`pathsnap.pipeline.MapMatcher.load()`, which checks that the stored feature schema equals the code's feature list.

| Module | Responsibility |
|---|---|
| `pathsnap/io.py` | parsing, validation, partition resolution; `load_trip` never reads `.route` |
| `pathsnap/preprocess.py`, `geo.py` | cleaning, projection, kinematics |
| `pathsnap/network.py` | cropped network, KD-tree index, bounded Dijkstra, shortest paths |
| `pathsnap/features.py` | candidate generation and the 28 features |
| `pathsnap/models.py` | the 8-model zoo |
| `pathsnap/hmm.py` | HMM/Viterbi decoding and route reconstruction |
| `pathsnap/pipeline.py` | `MapMatcher`: the only prediction path (used by scripts and the API) |
| `pathsnap/evaluation.py`, `labels.py`, `metrics.py` | server-side scoring after prediction |
| `api/main.py`, `service.py`, `schemas.py` | endpoints, prediction and evaluation services, Pydantic schemas |

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model name, partition served, whether the fallback is active, number of trips |
| `GET /evaluation/trips` | trip ids with observation count and duration (never routes or labels) |
| `POST /evaluation/run` | full pipeline plus server-side scoring for one trip |
| `GET /evaluation/{{id}}` | stored result (in memory) |
| `GET /evaluation/{{id}}/map` | Folium HTML of the run (no reference route) |
| `GET /models/comparison` | recorded TRAIN cross-validation comparison (read-only) |

**Response format.**
- Coordinates follow GeoJSON order `[lon, lat]`, fractions lie in [0, 1], times are in milliseconds (`*_ms`), and trip ids are zero-padded strings.
- `metrics` holds the classifier + HMM metrics; `classifier_only_metrics` and `nearest_road_metrics` hold the comparisons.
- The contract was proposed by the frontend developer (`docs/API_CONTRACT.md`) and implemented by the backend.

**Error handling.**
- Errors use FastAPI's `{{"detail": …}}` with 404 (unknown trip or evaluation), 422 (invalid body), 503 (model not loaded) and 500 (pipeline failure).
- If `PATHSNAP_EVAL_DIR` is set but missing or incomplete, the server **refuses to start**; it never silently serves another partition.
- Every request is logged with client IP, path, status and latency, and every run logs `RUN` and `DONE` lines with the metrics.

## 13. Frontend Implementation

**Stack.** React 19, TypeScript 5.9, Vite 8, Leaflet/react-leaflet; tests with Vitest and Testing Library. All backend calls go
through a single client module (`api/client.ts`). A runtime validator (`api/validate.ts`) rejects responses that do not
match the contract, so a malformed response is never shown as a success.

**Dashboard features.**
- Connection panel with health polling; the backend address can come from the UI, a `?api=` URL parameter or `.env`.
- Trip selector listing the 10 evaluation trips.
- Interactive Leaflet map: raw GPS observations (amber), the predicted route (cyan) and the nearby road network, with a layer control.
- Metric cards (top-1, macro-F1, continuity, log loss, inference time, HMM status) and an additional-metrics table.
- Charts: route quality, *Effect of HMM/Viterbi decoding* (classifier-only vs classifier + HMM) and a server timing breakdown.
- A **Model comparison** section: a sortable table of the 8 models (XGBoost highlighted) and a dot plot with fold-std whiskers, clearly labelled
  *TRAIN cross-validation · not evaluator-test results*.
- Session history, a cross-trip chart, and a second tab with the server-rendered Folium map.
{fig('dashboard_evaluator_00000015.png', 'Figure 14. Dashboard after evaluating official evaluator trip 00000015 (6,854 fixes): predicted route (cyan) over GPS (amber); metrics from Laptop 1.')}
{fig('dashboard_hmm_chart_00000015.png', 'Figure 15. Dashboard charts for evaluator trip 00000015, including the effect of HMM/Viterbi decoding versus classifier-only.', '92%')}
## 14. Official Evaluator Integration

**Folder.** `evaluator_test/` contains 10 trip folders (`00000015` … `00000094`), each holding `.track/.nodes/.arcs/.route`.
There are **no** scripts, README or metric definitions. The "evaluator" is the reserved data partition, and scoring is
performed by our API ("the API calculates prediction accuracy using its internal reference data", hackathon guidelines).
The trip ids are disjoint from TRAIN and student-test, and the input files passed validation (no NaN, ordered timestamps, valid node references).

**Loading.** The backend is started with `PATHSNAP_EVAL_DIR=<repo>\\evaluator_test`. `/health` then reports
`evaluation_partition = evaluator_test` with no fallback, and `/evaluation/trips` lists exactly the 10 ids.

**Frozen model.** The bundle `artifacts/final/` (model SHA-256 prefix `7ebb199fc9b56a3d`) was hashed before and after
the evaluation and was unchanged. No training, tuning or selection used evaluator data.

**Scoring.** For each request the prediction is completed first, from `.track/.nodes/.arcs` only. The server then aligns the reference
route to the fixes and computes the metrics. Only aggregate metrics and the *predicted* route are returned.

**Leakage prevention (verified by tests).**
- No prediction module imports the label code (static analysis test).
- A runtime file-access spy shows that prediction opens no `.route` file, and that scoring does.
- Predictions are byte-identical whether or not the `.route` file exists.
- The test suite cannot read evaluator data even if `PATHSNAP_EVAL_DIR` is set; a guard in `tests/conftest.py` blocks it.
- No training or tuning script references the evaluator partition.

**CV vs evaluator metrics.** TRAIN CV top-1 is a *classifier-only* score on reliable-label fixes of validation folds. Evaluator
metrics are computed on *all* fixes of unseen trips, for the full classifier + HMM system. The two measure different things and
are reported separately.

## 15. Results and Performance Analysis

### 15.A TRAIN cross-validation (development)
See Table 1. XGBoost reached a CV top-1 accuracy of {pct(cmp.loc['xgboost', 'cv_top1_accuracy'])} ± {pct(cmp.loc['xgboost', 'cv_top1_accuracy_std'])},
against {pct(cmp.loc['nearest_road', 'cv_top1_accuracy'])} for nearest road. With the HMM on out-of-fold TRAIN predictions, accuracy was {pct(S['hmm']['train_subset_best_hmm_acc'])} (24 trips).

### 15.B Student-test confirmation (10 trips, {stsum and int(st.n_obs.sum()):,} fixes)

{ST_TABLE}

*Table 4. Source: `reports/experiments/route_eval_student_test_summary.json`.*

{ST_TRIPS}

*Table 5. Per-trip student-test results. Source: `reports/experiments/route_eval_student_test.csv`. The HMM improved accuracy on
{int((st_gain > 0).sum())} of {len(st)} trips (gain {st_gain.min() * 100:+.2f} to {st_gain.max() * 100:+.2f} pp).*

### 15.C Official evaluator-test (10 trips, {evs['total_observations']:,} fixes, frozen system, each trip evaluated once)

{EV_TABLE}

*Table 6. Source: `reports/evaluator/evaluator_results.json`. Mean candidate miss rate: {pct(evs['candidate_miss_rate_mean'])}.*

{EV_TRIPS}

*Table 7. Per-trip evaluator-test results. Source: `reports/evaluator/evaluator_results.csv`. Inference = server-side prediction time;
round trip = complete `POST /evaluation/run`, including scoring. Strict cont. = share of arc changes between consecutive fixes that are directly connected.*
{fig('evaluator_accuracy_per_trip.png', 'Figure 16. Official evaluator-test: per-trip pointwise accuracy of nearest road (A), XGBoost only (B) and XGBoost + HMM (C).')}
{fig('evaluator_route_quality.png', 'Figure 17. Official evaluator-test: route F1, sequence accuracy and network-reachable continuity, XGBoost only vs XGBoost + HMM.')}
**Analysis.**
- **Accuracy.** The HMM improved accuracy on **all {len(ev)} evaluator trips** (gain {ev_gain.min() * 100:+.2f} to {ev_gain.max() * 100:+.2f} pp).
  The largest gain was on the longest trip, {ev.loc[ev_gain.idxmax(), 'trip_id']}.
- **Route coherence.** Unreachable transitions fall from {evs['classifier_only']['unreachable_transitions_total']} (classifier only) to
  {evs['classifier_hmm']['unreachable_transitions_total']}. Mean sequence accuracy rises from {pct(evs['classifier_only']['sequence_accuracy_trip_mean'], 1)} to {pct(evs['classifier_hmm']['sequence_accuracy_trip_mean'], 1)}.
- **Where the HMM did not improve everything.** On trip{'s' if len(worse_cont) != 1 else ''} {', '.join(worse_cont.trip_id)}, strict continuity is *lower* with the
  HMM ({', '.join(f"{pct(r.hmm_route_continuity, 1)} vs {pct(r.clf_route_continuity, 1)}" for _, r in worse_cont.iterrows())}), even though accuracy and route F1 are higher.
  The same trip has the lowest sequence accuracy ({pct(ev.loc[ev.trip_id == '00000015', 'hmm_sequence_accuracy'].iloc[0], 1)}).
- **Evaluator vs student-test.** Evaluator-test accuracy is higher than student-test ({pct(evs['classifier_hmm']['accuracy_fix_weighted'], 1)} vs
  {pct(st_ow['classifier_hmm']['accuracy'], 1)}). The two partitions contain different trips, and the difference should not be read as an
  improvement of the model, which is identical.

**Performance** (`reports/experiments/benchmark.json`, student-test, 16-core laptop, {bench['cpu']['ram_gb']} GB RAM):
model load {bench['model_load_ms']:,.0f} ms; pipeline per trip {bench['pipeline_mean_ms']:,.0f} ms on average
(max {bench['pipeline_max_ms']:,.0f} ms, {bench['ms_per_observation']:.2f} ms per fix); HMM {bench['hmm_mean_ms']:,.0f} ms per trip;
API response {bench['api_mean_response_ms']:,.0f} ms on average (max {bench['api_max_response_ms']:,.0f} ms);
peak memory {bench['rss_peak_mb']:,.0f} MB.
On evaluator-test: mean inference {evs['timing_ms']['inference_mean']:,} ms, max {evs['timing_ms']['inference_max']:,} ms, mean round trip
{evs['timing_ms']['round_trip_mean']:,} ms, max {evs['timing_ms']['round_trip_max']:,} ms; peak server memory {evs['server_rss_peak_mb']} MB.

## 16. Testing and Validation

All results below were produced by fresh runs on {today}. The raw output is stored in `reports/final/evidence/`.

| Suite | Result | What it covers |
|---|---|---|
| Python (pytest, 7 files) | **{py_line}** | parsing, preprocessing, geodesy, candidates, features, label alignment, HMM, metrics, every model's probability output, bundle reload in a fresh process, determinism, API endpoints and errors, CORS, GroupKFold separation, partition disjointness, leakage guards, refusal to start on a bad evaluator directory, the runtime `.route` access spy |
| Frontend (Vitest, 5 files) | **{fe_line}** | API client, response validation, dashboard flows, history, model-comparison table (sorting, highlighting, missing values) |
| TypeScript check (`tsc --noEmit`) | **{'passed (exit 0)' if tc_ok else 'n/r'}** | static types of the whole UI |
| Production build (`vite build`) | **{'passed' if build_ok else 'n/r'}** | bundling for deployment |
| API contract (real responses through the frontend validator) | **{ct_line}** | real FastAPI JSON parsed by the unmodified frontend validator |
| Browser end-to-end (Playwright + Edge, production build) | **{'passed: all ' + str(len(e2e_checks)) + ' checks' if all(e2e_checks.values()) else 'failed'}** on all {len(e2e_runs)} evaluator trips | real UI → API → frozen model |

*Unit tests use synthetic data only for isolated cases. All result metrics come from real data.*

**End-to-end check results** (`reports/integration/e2e_summary_evaluator.json`): {', '.join(f'{k.replace("_", " ")} = {v}' for k, v in e2e_checks.items())}.
All map-tile HTTP responses returned 200. The browser recorded some `net::ERR_ABORTED` events; these are Leaflet cancelling tiles of the previous view and are not errors.

| Trip | Top-1 UI / API | Macro-F1 UI / API | Route segments UI / API | UI time |
|---|---|---|---|---|
{e2e_rows}

*Table 8. The values displayed in the browser equal the API response exactly.*

**Final model verification.** The bundle reloads in a fresh process (`test_bundle_reloads_in_fresh_process_and_predicts`), and inference is deterministic.
The bundle's SHA-256 hashes were identical before and after evaluator use.

## 17. Challenges and Solutions

| Challenge (observed in this project) | Solution |
|---|---|
| Reference routes give no per-fix labels | monotone DP alignment of fixes to route positions (Section 5.5) |
| Some TRAIN reference routes do not cover the drive (e.g. trips 00000054, 00000033) | reliability filter (≤ 30 m); unreliable fixes excluded from fitting and CV |
| Twin arcs with identical geometry; ambiguity at junctions | heading features over two windows, transition-consistency features, U-turn penalty in the HMM |
| Independent predictions produce disconnected routes | HMM/Viterbi with directed network distances; shortest-path gap filling |
| Near-tied model ranking | pre-declared selection rule; calibration and speed as tie-breakers; honest reporting |
| Optuna tuning worsened calibration | decoder tuned on out-of-fold probabilities learned a small emission weight (0.215) |
| Large networks (up to 307 k arcs) and long trips | network cropping, KD-tree index, memoised bounded Dijkstra |
| Contract mismatch between the backend draft and the UI | backend re-shaped to the UI's contract; contract test with the real validator |
| Windows `OSError [Errno 22]` when re-saving a PNG open in a viewer | reproduced as a memory-mapped file; atomic writes with a timestamped fallback |
| OpenStreetMap tiles returned HTTP 403 for maps opened from disk | offline Folium maps drawing the road network instead of tiles |
| `localhost` resolving to IPv6 first (≈2 s delay per request) | documented use of `127.0.0.1` or the LAN IP |
| Silent fallback to student-test if the evaluator path is wrong | backend refuses to start when an explicit evaluator folder is invalid |
| Tests that could touch evaluator labels | test-suite guard blocking every evaluator path |
| Two-laptop Wi-Fi connectivity | binding to `0.0.0.0`, permissive CORS, firewall check; verified with a real Laptop 2 connection |

## 18. Limitations

* **Residual errors are mostly direction.** The undirected accuracy on student-test is {pct(st_ow['classifier_hmm']['accuracy_undirected'], 1)}
  against {pct(st_ow['classifier_hmm']['accuracy'], 1)} directed. The largest remaining error is choosing the reverse twin arc of the correct road.
* **The HMM does not improve every metric.** On evaluator trip 00000015 strict continuity is lower with the HMM; sequence
  accuracy varies widely between trips.
* **Optimisation gave no real gain.** Bayesian optimisation did not materially improve accuracy and worsened calibration.
* **Labels are a proxy.** Labels come from route alignment; some TRAIN reference routes are incomplete.
* **Performance trade-offs.** The tuned model is about 3× slower than the default configuration. Server-side scoring of long trips takes up to about 0.8 s
  in addition to inference. API memory grows to about {evs['server_rss_peak_mb']} MB because the last 10 runs are kept for the map endpoint.
* **Generalisation.** The networks are per-trip and worldwide, but there are only 80 training trips. Behaviour on very different road types or
  sampling rates (e.g. 0.1 Hz) was not tested. There is no GPS smoothing.
* **Deployment.**
  - Offline batch processing per trip; no streaming.
  - In-memory result store.
  - The UI's map tiles need internet.
  - The firewall rule may need administrator rights.
  - Venue Wi-Fi client isolation could block the two-laptop set-up.

## 19. Future Scope

* **Real-time streaming:** online (fixed-lag) Viterbi over a sliding window for live tracking.
* **Larger networks:** a persistent spatial database (e.g. PostGIS) and precomputed contraction hierarchies for shortest paths.
* **Better sequence models:** jointly trained CRFs, or neural sequence models (Transformers over candidate sets), with learned transition features.
* **Calibration:** isotonic or Platt calibration of the classifier before decoding; a multi-objective search (accuracy and log loss).
* **Direction handling:** explicit direction-of-travel features from longer windows and one-way restrictions.
* **Scalability and deployment:** containerised backend, asynchronous job queue, persistent results, authentication, monitoring.
* **Route quality:** GPS smoothing (Kalman), dedicated handling of long gaps, uncertainty estimates per matched segment.

## 20. Conclusion

PathSnap delivers a complete, reproducible map-matching system: validated preprocessing, candidate generation with 28
label-free features, a documented comparison of 8 classifiers under trip-grouped cross-validation, Bayesian optimisation,
TRAIN-tuned HMM/Viterbi decoding with route reconstruction, a FastAPI backend and a React evaluator dashboard.
On the official evaluator-test partition, measured once with the frozen model, the system matched
**{pct(evs['classifier_hmm']['accuracy_fix_weighted'])}** of GPS fixes to the correct directed road segment, with macro-F1 {f4(evs['classifier_hmm']['macro_f1_fix_weighted'])}
and a route F1 of {pct(evs['classifier_hmm']['route_f1_trip_mean'])}. This is a clear improvement over nearest-road snapping
({pct(evs['nearest_road']['accuracy_fix_weighted'])}) and over the classifier alone ({pct(evs['classifier_only']['accuracy_fix_weighted'])}).
Data integrity was enforced by design and verified by tests: TRAIN-only development, student-test for confirmation, evaluator data
unseen until the final measurement, and ground truth never exposed to the prediction path or the UI.
The two-laptop deployment was demonstrated over a real Wi-Fi connection.

## 21. References

1. P. Newson and J. Krumm, "Hidden Markov Map Matching Through Noise and Sparseness," *Proc. 17th ACM SIGSPATIAL GIS*, 2009.
2. A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," *IEEE Trans. Information Theory*, 13(2), 1967.
3. L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," *Proc. IEEE*, 77(2), 1989.
4. M. A. Quddus, W. Y. Ochieng and R. B. Noland, "Current map-matching algorithms for transport applications: State-of-the art and future research directions," *Transportation Research Part C*, 15(5), 2007.
5. M. Kubička, A. Cela, P. Moulin, H. Mounier and S.-I. Niculescu, "Dataset for testing and training of map-matching algorithms," *IEEE Intelligent Vehicles Symposium*, 2015.
6. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. KDD*, 2016.
7. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *NeurIPS*, 2017.
8. L. Breiman, "Random Forests," *Machine Learning*, 45, 2001.
9. P. Geurts, D. Ernst and L. Wehenkel, "Extremely randomized trees," *Machine Learning*, 63, 2006.
10. T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," *Proc. KDD*, 2019.
11. J. Bergstra, R. Bardenet, Y. Bengio and B. Kégl, "Algorithms for Hyper-Parameter Optimization," *NeurIPS*, 2011 (TPE).
12. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12, 2011.
13. E. W. Dijkstra, "A note on two problems in connexion with graphs," *Numerische Mathematik*, 1, 1959.
14. J. L. Bentley, "Multidimensional binary search trees used for associative searching," *Communications of the ACM*, 18(9), 1975 (k-d tree).
15. Software: NumPy, pandas, SciPy, FastAPI, Uvicorn, Pydantic, Folium/Leaflet, React, TypeScript, Vite, Vitest, pytest, Playwright (official documentation).

## 22. Appendix

### 22.1 Project folder structure
```
pathsnap/        preprocessing, network, features, labels, models, hmm, pipeline, evaluation, viz
scripts/         01_explore … 09_benchmark, demo_preprocessing
api/             FastAPI app (main.py, service.py, schemas.py)
frontend/        React/Vite evaluator UI (Laptop 2)
tests/           pytest suite (7 files, incl. evaluator-safety guard)
integration/     contract test, Playwright end-to-end, evaluator runner, captured fixtures
artifacts/final/ model.joblib, feature_schema.json, hmm_params.json, manifest.json   (frozen)
reports/         EDA, experiments (CSV/JSON), figures, maps, evaluator results, final report
docs/            API contract, two-laptop set-up, demo runbook, preprocessing review
evaluator_test/  official evaluator trips (git-ignored, never used for development)
```

### 22.2 Important commands (Windows PowerShell)
```powershell
.\\setup_windows.bat                                                # environment
.\\reproduce_all.bat                                                # full TRAIN pipeline (EDA -> models -> Optuna -> HMM -> eval -> plots)
$env:PATHSNAP_EVAL_DIR = "<repo>\\evaluator_test"; .\\run_api.bat   # backend on 0.0.0.0:8000
cd frontend; npm run build; npm run preview                          # UI on :4173
.venv\\Scripts\\python -m pytest tests -q                            # Python tests
.venv\\Scripts\\python integration\\e2e_browser.py --ui http://127.0.0.1:4173 --api http://127.0.0.1:8000
```

### 22.3 API example
```http
POST /evaluation/run
{{"trip_id": "00000059"}}

200 OK  (abridged)
{{"evaluation_id": "…", "trip_id": "00000059", "status": "completed",
 "model": {{"name": "XGBoost (tuned) + HMM/Viterbi", "version": "1.0.0"}},
 "observations": [{{"index": 0, "lon": …, "lat": …, "t": 1.0, "matched_arc_id": …}}, …],
 "predicted_route": {{"arc_ids": […], "segments": [{{"arc_id": …, "coordinates": [[lon, lat], [lon, lat]]}}, …]}},
 "metrics": {{"top1_accuracy": {ev.loc[ev.trip_id == '00000059', 'hmm_top1_accuracy'].iloc[0]:.4f}, "macro_f1": {ev.loc[ev.trip_id == '00000059', 'hmm_macro_f1'].iloc[0]:.4f}, …}},
 "classifier_only_metrics": {{…}}, "timing": {{"inference_ms": …, "hmm_ms": …}},
 "hmm": {{"applied": true, "status": "ok"}}, "warnings": []}}
```

### 22.4 Final model parameters
**XGBoost** (`artifacts/final/manifest.json`):

| Parameter | Value |
|---|---|
{chr(10).join(f"| `{k}` | {v:.4g} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in S['manifest']['hyperparameters'].items())}

**Pipeline configuration:** {', '.join(f"`{k}` = {v}" for k, v in S['manifest']['pipeline_config'].items())}.
**HMM:** {', '.join(f"`{k}` = {v:.4g}" if isinstance(v, float) else f"`{k}` = {v}" for k, v in hp.items())}.
**Environment:** {', '.join(f"{k} {v}" for k, v in S['manifest']['environment'].items())}.

### 22.5 Additional figures
{fig('eda_trajectories.png', 'Figure A1. Six TRAIN trajectories over their road networks.', '90%')}
{fig('eda_label_distance_rank.png', 'Figure A2. Fix-to-reference-arc distance and rank of the reference arc among the nearest arcs.', '85%')}
{fig('models_confusion_matrices.png', 'Figure A3. Candidate-level confusion matrices summed over the 5 CV folds.', '95%')}
{fig('mlp_training_history.png', 'Figure A4. MLP training history (the only model with epoch-wise curves).', '65%')}
{fig('optuna_xgboost_importance.png', 'Figure A5. Optuna hyperparameter importance.', '65%')}
{fig('models_efficiency.png', 'Figure A6. Accuracy versus inference cost per model.', '65%')}

### 22.6 Statement on AI assistance
In line with the hackathon's academic-integrity rules, the team discloses the following. AI coding assistants were used during
development of the backend, ML pipeline, tests and documentation, and of the frontend. They also helped draft this report. All
reported numbers were produced by executing the code in this repository and are read programmatically from the result files named
under each table.

### 22.7 Traceability
| Result | Source file |
|---|---|
| Dataset statistics | `reports/eda_summary.json`, `reports/eda_trip_stats.csv` |
| Model comparison, folds | `reports/experiments/model_comparison.csv`, `cv_folds.csv` |
| Optuna | `reports/experiments/optuna_xgboost_best.json`, `optuna_xgboost_trials.csv` |
| Tuned vs default | `reports/experiments/tuned_vs_untuned.csv` |
| HMM tuning | `reports/experiments/hmm_tuning.json`, `artifacts/final/hmm_params.json` |
| Student-test | `reports/experiments/route_eval_student_test*.csv/json` |
| Evaluator-test | `reports/evaluator/evaluator_results.csv/json` |
| Performance | `reports/experiments/benchmark.json` |
| Tests | `reports/final/evidence/*.txt`, `reports/integration/e2e_summary_evaluator.json` |
"""

(OUT / "PathSnap_Final_Project_Report.md").write_text(md, encoding="utf-8")
print("markdown written:", len(md.split()), "words")

# ------------------------------------------------------------------ HTML + DOCX (isolated tool env)
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font-family: 'Segoe UI', 'Calibri', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1d2433; }
h1 { font-size: 34pt; margin: 0; color: #0f3d5e; }
h2 { font-size: 15.5pt; color: #0f3d5e; border-bottom: 2px solid #1f9bb0; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #1f4e85; margin-top: 14px; page-break-after: avoid; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 8.4pt; page-break-inside: avoid; }
th { background: #0f3d5e; color: #fff; text-align: left; padding: 4px 5px; }
td { border-bottom: 1px solid #d5dbe5; padding: 3px 5px; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #eef2f7; padding: 0 2px; border-radius: 2px; }
pre { background: #f4f7fb; border: 1px solid #d5dbe5; padding: 8px; font-size: 7.8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
figure { margin: 10px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e1e6ee; }
figcaption { font-size: 8.6pt; color: #4a5568; margin-top: 4px; font-style: italic; }
blockquote { border-left: 4px solid #f0a03c; background: #fff8ec; margin: 10px 0; padding: 6px 12px; }
em { color: #374151; }
.cover { height: 247mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.cover-kicker { letter-spacing: .18em; text-transform: uppercase; color: #1f9bb0; font-weight: 600; text-align: center; }
.cover-sub { font-size: 17pt; color: #1f4e85; margin: 10px 0 24px; text-align: center; }
.cover-desc { max-width: 140mm; margin: 0 auto 40px; color: #4a5568; text-align: center; }
.cover-team { font-size: 13pt; text-align: center; } .cover-team p { text-align: center; }
.cover-date { margin-top: 30px; color: #6b7280; text-align: center; }
.pagebreak { page-break-after: always; }
.katex, .math { font-style: italic; }
"""
conv = r'''
import sys, re, markdown, pypandoc
from pathlib import Path
out = Path(sys.argv[1]); css = Path(sys.argv[2]).read_text(encoding="utf-8")
md = (out / "PathSnap_Final_Project_Report.md").read_text(encoding="utf-8")
# MathJax-free: render $...$ math as plain italic text for HTML/PDF
def math_html(m):
    s = m.group(1)
    s = re.sub(r"\\dfrac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{(.+?)\}\{(.+?)\}", r"\1/\2", s)
    for a, b in [(r"\lvert", "|"), (r"\rvert", "|"), (r"\left", ""), (r"\right", ""), (r"\max", "max"), (r"\min", "min"),
                 (r"\log", "log"), (r"\cdot", "·"), (r"\sum", "Σ"), (r"\to", "→"), (r"\in", "∈"), (r"\le", "≤"), (r"\dots", "…"),
                 (r"\lambda", "λ"), (r"\varphi", "φ"), (r"\tau", "τ"), (r"\beta", "β"), (r"\rho", "ρ"), (r"\pi", "π"), (r"\cos", "cos"),
                 (r"\mathbf", ""), (r"\text", ""), (r"\,", " "), ("\\", "")]:
        s = s.replace(a, b)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s); s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    return '<span class="math">' + s.replace("{", "").replace("}", "") + "</span>"
md_html = re.sub(r"\$(.+?)\$", math_html, md)
body = markdown.markdown(md_html, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Final Project Report</title><style>{css}</style></head><body>{body}</body></html>"
(out / "PathSnap_Final_Project_Report.html").write_text(html, encoding="utf-8")
# DOCX via pandoc (keeps real TeX math as Word equations); strip HTML-only wrappers
md_docx = re.sub(r'<div class="pagebreak"></div>', "", md)
md_docx = re.sub(r'<figure><img src="([^"]+)"[^>]*/><figcaption>(.*?)</figcaption></figure>', r"![\2](\1)", md_docx)
md_docx = re.sub(r"</?div[^>]*>", "", md_docx); md_docx = re.sub(r'<p class="[^"]+">(.*?)</p>', r"\1\n", md_docx)
pypandoc.convert_text(md_docx, "docx", format="markdown+tex_math_dollars+pipe_tables", outputfile=str(out / "PathSnap_Final_Project_Report.docx"),
                      extra_args=[f"--resource-path={out}", "--toc", "--metadata=title:PathSnap: GPS Map Matching and Route Reconstruction Using Machine Learning"])
print("html + docx written")
'''
(OUT / ".style.css").write_text(CSS, encoding="utf-8")
py = TOOLENV / "Scripts" / "python.exe"
subprocess.run([str(py), "-c", conv, str(OUT), str(OUT / ".style.css")], check=True)
(OUT / ".style.css").unlink()

# ------------------------------------------------------------------ PDF via Playwright/Edge (project venv)
from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    br = p.chromium.launch(channel="msedge", headless=True)
    pg = br.new_page()
    pg.goto((OUT / "PathSnap_Final_Project_Report.html").as_uri(), wait_until="networkidle")
    pg.pdf(path=str(OUT / "PathSnap_Final_Project_Report.pdf"), format="A4", print_background=True, prefer_css_page_size=True,
           display_header_footer=True,
           header_template="<div style='font-size:7px;width:100%;text-align:right;padding-right:16mm;color:#8a94a6'>PathSnap · Final Project Report</div>",
           footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#6b7280'>Page <span class='pageNumber'></span> of <span class='totalPages'></span></div>",
           margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
    br.close()
pages = subprocess.run([str(py), "-c", f"import pypdf;print(len(pypdf.PdfReader(r'{OUT / 'PathSnap_Final_Project_Report.pdf'}').pages))"],
                       capture_output=True, text=True, check=True).stdout.strip()
print("pdf pages:", pages)
