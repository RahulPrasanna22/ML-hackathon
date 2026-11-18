# PathSnap — Laptop 1 results (2026-11-18)

Every number below is copied from a file produced by the scripts in this repository:
`reports/experiments/model_comparison.csv`, `optuna_xgboost_best.json`, `tuned_vs_untuned.csv`,
`hmm_tuning.json`, `route_eval_student_test*.{csv,json}`, `benchmark.json`, `reports/integration/e2e_summary*.json`.

## Data usage (leakage control)

| stage | train (80 trips) | student_test (10) | evaluator_test |
|---|---|---|---|
| feature/label build | yes | yes (separately) | never read |
| model fitting, 5-fold GroupKFold by trip | **yes** | no | no |
| algorithm selection (CV top-1) | **yes** | reported only | no |
| Optuna (22 trials) | **yes** | no | no |
| HMM decoder tuning (30 trials) | **yes**: out-of-fold probabilities, 24 trips | confirmation only | no |
| final held-out confirmation | – | **yes** | – |

Training rows are restricted to fixes whose aligned reference label lies within 30 m (87.5 % of train fixes;
see `reports/EDA.md`). Student-test metrics use all fixes.

## 1. Eight-model comparison (candidate classifiers, TRAIN 5-fold GroupKFold)

| model | family | CV top-1 | ± fold std | CV macro-F1 | CV log loss | CV continuity | train-fold top-1 | fit (all rows) | inference ms/fix | size MB | student-test top-1 | ST macro-F1 | ST log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest road (baseline) | geometric | 0.5609 | – | – | – | 0.4137 | – | – | – | – | 0.5318 | – | – |
| Logistic Regression | linear | 0.8878 | 0.0117 | 0.9212 | 0.1074 | 0.7816 | 0.8893 | 2.3 s | 0.001 | <0.01 | 0.8668 | 0.9136 | 0.1037 |
| Gaussian NB | probabilistic | 0.8559 | 0.0152 | 0.8543 | 0.7037 | 0.7817 | 0.8573 | 0.3 s | 0.002 | <0.01 | 0.8338 | 0.8498 | 0.7063 |
| Random Forest | bagging | 0.9110 | 0.0126 | 0.9458 | 0.0699 | 0.8198 | 0.9516 | 79.9 s | 0.020 | 61.3 | 0.9005 | 0.9408 | 0.0682 |
| Extra Trees | randomised trees | 0.9078 | 0.0121 | 0.9430 | 0.0743 | 0.8212 | 0.9467 | 52.9 s | 0.033 | 210.7 | 0.8966 | 0.9367 | 0.0724 |
| HistGradientBoosting | boosting | 0.9070 | 0.0123 | 0.9440 | 0.0821 | 0.7829 | 0.9428 | 13.9 s | 0.019 | 0.5 | 0.8866 | 0.9348 | 0.0965 |
| LightGBM | boosting | 0.9120 | 0.0137 | 0.9466 | 0.0692 | 0.8196 | 0.9555 | 44.0 s | 0.113 | 1.2 | 0.8982 | 0.9391 | 0.0720 |
| **XGBoost** | boosting | **0.9127** | 0.0135 | **0.9468** | **0.0681** | **0.8223** | 0.9520 | 19.2 s | 0.007 | 1.0 | 0.8982 | 0.9393 | 0.0704 |
| MLP (64-32) | neural network | 0.9075 | 0.0124 | 0.9411 | 0.0804 | 0.8089 | 0.9310 | 102.9 s | 0.002 | 0.05 | 0.8945 | 0.9351 | 0.0752 |

*Top-1* = fraction of GPS fixes whose highest-scoring candidate is the reference arc. *Macro-F1* is computed at candidate
level (threshold 0.5) over the classes {not reference, reference}. *Continuity* = share of arc changes in the per-fix
argmax sequence that are directly connected. The fit times of the slower models were measured while other jobs shared
the CPU, so treat them as indicative. Random Forest and Extra Trees weights are git-ignored (> 50 MB); regenerate them with `scripts/03_train_models.py`.

**Selection (TRAIN only):** XGBoost has the best CV top-1, macro-F1, log loss and continuity. LightGBM and Random Forest
are within one fold standard deviation on top-1, so the tie is broken by the better-calibrated probabilities
(important for the HMM emissions) and faster inference. Tree ensembles fit training folds clearly better than validation
folds (RF 0.952 vs 0.911): moderate overfitting. Linear and naive Bayes models underfit (train ≈ validation, both lower).

## 2. Bayesian optimisation (Optuna TPE, XGBoost)

* Objective: mean GroupKFold(5) observation top-1 on TRAIN (trials fit on every 2nd fix of the training folds for budget,
  and validate on all fixes). Sampler TPE(seed 42, 10 random start-up trials). 22 trials completed in 3 155 s (50 min budget).
* Search space: n_estimators 100–1000, learning_rate 0.01–0.2 (log), max_depth 3–12, min_child_weight 0.5–50 (log),
  subsample 0.5–1, colsample_bytree 0.4–1, reg_lambda 1e-3–10 (log).
* Best (trial 1): n_estimators 900, learning_rate 0.0605, max_depth 10, min_child_weight 0.550, subsample 0.985,
  colsample_bytree 0.899, reg_lambda 0.0071.

Tuned vs untuned on identical full-data folds (`tuned_vs_untuned.csv`):

| | CV top-1 | CV macro-F1 | CV log loss | student-test top-1 | ST macro-F1 | ST log loss | fit | inference ms/fix |
|---|---|---|---|---|---|---|---|---|
| untuned | 0.9127 ± 0.0135 | 0.9468 | 0.0681 | 0.8982 | 0.9393 | 0.0704 | 19 s | 0.007 |
| tuned (final) | 0.9135 ± 0.0150 | 0.9468 | 0.0856 | 0.8926 | 0.9367 | 0.0912 | 63 s | 0.025 |

The tuned model was kept under the pre-declared rule (best CV top-1 on TRAIN). Its gain (+0.07 pp) is far smaller than
the fold standard deviation. It is less well calibrated (higher log loss) and slower, and on the held-out student-test
it is slightly *worse*. Honest conclusion: hyperparameter optimisation did not materially improve this model. The
student-test figures were not used to change the choice.

## 3. HMM / Viterbi decoder (tuned on TRAIN out-of-fold probabilities)

Hidden states = candidate arcs (with projected position) per fix. Emission = `w_e · log(p / Σp)` of the classifier.
Transition = `−|d_network − d_gps| / β`. The network distance comes from a bounded directed Dijkstra; unreachable
transitions get −∞; a switch to the reverse twin arc pays a U-turn penalty. Chain restarts (with a penalty) handle
gaps > 600 s and dead ends. The route is reconstructed by shortest-path filling between consecutive matched arcs.

30 Optuna trials over 24 seeded train trips (199 s). Final parameters (`artifacts/final/hmm_params.json`): β = 22.28,
emission_weight = 0.215, sigma = 0, same_arc_tol = 26.7 m, break_penalty = 26.8, reverse_penalty = 7.35.
The low emission weight means the decoder discounts the over-confident tuned probabilities.

| (TRAIN, out-of-fold, 24 trips) | accuracy |
|---|---|
| classifier only | 0.9042 |
| HMM, default parameters | 0.9206 |
| HMM, tuned | **0.9349** |

## 4. Held-out student-test confirmation (10 trips, 19 160 fixes, final bundle, full pipeline from raw files)

| | A nearest road | B classifier only | C classifier + HMM |
|---|---|---|---|
| accuracy (fix-weighted) | 0.5318 | 0.8926 | **0.9155** |
| macro-F1 (fix-weighted) | 0.7239 | 0.9391 | **0.9523** |
| accuracy, either direction of the road | 0.8912 | 0.9496 | **0.9687** |
| accuracy (trip mean) | 0.5620 | 0.9134 | **0.9353** |
| strict continuity (trip mean) | 0.4474 | 0.8384 | **0.8975** |
| network-reachable continuity | 0.8797 | 0.9781 | **0.9988** |
| sequence accuracy (1 − norm. edit distance) | 0.3226 | 0.6956 | **0.9239** |
| route F1, length-weighted | 0.6442 | 0.9546 | **0.9860** |
| edit distance (trip mean) | 423.4 | 147.8 | **25.5** |
| unreachable transitions (total) | 709 | 146 | **3** |
| mean deviation from reference route | 1.13 m | 0.43 m | **0.27 m** |

Candidate miss rate 0.17 %. HMM chain breaks: 0. Unfillable route gaps: 3 (over 10 trips).
The HMM improved accuracy on all 10 trips, by +0.40 pp (00000040) up to +4.37 pp (00000076). Its largest effect is on route
coherence: edit distance falls ~6× and unreachable jumps fall from 146 to 3. On the hardest trip, 00000077, directed accuracy
is 0.752 but 0.967 when the reverse twin of the correct arc counts. So ~21.5 % of its fixes are on the right road
segment with the wrong travel direction (route F1 is still 0.980). Why the direction is lost there has not been investigated further.

## 5. Performance (`benchmark.json`, 16-core laptop, 15.8 GB RAM)

| measure | value |
|---|---|
| model bundle load (in benchmark process) | 2 057 ms (cold, fresh process incl. imports ≈ 3.0 s; API startup log 489 ms) |
| pipeline per trip, mean / max | 594 ms / 1 112 ms (0.31 ms per GPS fix) |
| HMM/Viterbi per trip, mean | 235 ms |
| API response, mean / max (in-process, incl. server-side evaluation) | 1 121 ms / 3 586 ms (trip 00000056: 1 317 ms prediction + 2 242 ms evaluation) |
| API payload, mean | 410 KB |
| process memory: after model load / peak | 313 MB / 616 MB |

## 6. Integration (real servers, real model)

* `integration/contract.test.ts`: real API JSON parsed by Laptop 2's unmodified `validate.ts`: **6/6 passed**.
* `integration/e2e_browser.py` (Edge → UI → FastAPI → final model):
  * dev server, trips 00000052 / 00000007 / 00000077: **all 8 checks passed**
  * production build (`vite preview`), trips 00000000 / 00000056 / 00000091: **all 8 checks passed**

  The checks: real trips listed, route drawn, displayed top-1 / macro-F1 / segment count equal the API JSON, history
  lists each run, 0 console errors, 0 tile HTTP errors (50 and 53 tiles, all 200). The only `requestfailed` events
  were `net::ERR_ABORTED` client cancellations: React StrictMode double-mount in dev, and Leaflet cancelling tiles of
  the previous view.
* Screenshots: `reports/integration/ui_trip_*.png`.

## 7. Limitations

* Evaluator trips were not available; the API serves `student_test` until `PATHSNAP_EVAL_DIR` is set. `/health` reports this.
* Reference-route labels are a proxy (monotone alignment); some train trips have reference routes that do not cover the drive.
* Hyperparameter optimisation gave no meaningful gain and worse calibration (see §2).
* Server-side evaluation of very long trips takes up to ~2.2 s (reachable-continuity path searches); prediction itself takes ≤ 1.3 s.
* No GPS smoothing. The dominant residual error is choosing the reverse-direction twin arc (e.g. trip 00000077: 0.752 directed vs 0.967 undirected accuracy).
* The Laptop 2 UI loads OpenStreetMap tiles from the internet (its own design); without internet, routes still draw but
  the background stays blank. Laptop 1's Folium maps need no tiles.
