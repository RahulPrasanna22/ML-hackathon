# PathSnap — First Review: GPS Data Preprocessing (Laptop 1)

Everything in this document was produced by code in this repository and executed on 2026-11-18.
The numbers come from `reports/eda_summary.json`, `reports/eda_trip_stats.csv`, `reports/candidate_stats_*.csv`
and `reports/preprocessing_demo/summary_00000003.json`.

## 1. Problem statement

GPS fixes drift off the road: onto parallel streets, across buildings, through gaps. PathSnap matches each
GPS observation to the road segment (arc) the vehicle actually drove on, and then reconstructs the route.
Preprocessing turns the raw per-trip files into a clean, metric, validated representation:
**one row per (GPS fix, nearby candidate arc)** with engineered features. Supervised models then score
these rows, and an HMM/Viterbi decoder connects the scores into a route.

## 2. Raw dataset structure (verified by inspecting the files)

```
Student folder/Student folder/            (local only, git-ignored)
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
```
The evaluator-test partition is not distributed and is never accessed by the preprocessing code.

| file | content (whitespace separated, no header) | train trip 00000003 |
|---|---|---|
| `.track` | lon, lat, t (s) — ~1 Hz | 821 fixes, 25.6 KB |
| `.nodes` | lon, lat; row index = node id | 9 138 nodes |
| `.arcs` | from_node, to_node; row index = arc id; **directed** | 18 028 arcs |
| `.route` | chronological reference arc ids | 79 arcs |

Findings that shaped the pipeline:
* **Every trip has its own road network** (train median 19.5 k arcs, max 307 k). Arc ids are local to a trip.
* `.route` has **no per-fix labels** (821 fixes vs 79 route arcs). Labels must be derived (step 7).
* Trips are worldwide (lon −100…137, lat −35…62). Distances therefore need a location-independent metric projection.
* 95 % of arcs (median) have a reverse twin with identical geometry. Only direction tells them apart, so heading matters.

## 3. Pipeline, step by step (module → what the code does)

| # | step | code | operations actually performed |
|---|---|---|---|
| 1 | Parse | `pathsnap/io.py` | `load_track` / `load_network` with column-count checks; `load_trip` reads **only** `.track/.nodes/.arcs`; `.route` is read only by `load_reference_route` (labels and evaluation) |
| 2 | Validate | `io.validate_trip`, `network.full_graph_stats` | NaN, out-of-range and (0,0) coordinates; non-increasing / duplicate timestamps; gaps > 5 s and > 30 s; invalid node refs, self-loops, duplicate arcs; route id validity and connectivity; weak and strong components, isolated nodes, two-way share |
| 3 | Clean GPS | `preprocess.clean_track` | drop NaN, out-of-range and (0,0) fixes; stable sort by time; drop duplicate timestamps (keep first); keep `orig_idx` |
| 4 | Project + kinematics | `geo.LocalProjection`, `preprocess.preprocess_track` | WGS84 local equirectangular projection around the trip median (metres); step distance, dt, speed; robust centred heading (window grows until displacement ≥ 2 m, else marked invalid); flags for speed outliers (> 252 km/h) and gaps (> 5 s) |
| 5 | Road network | `network.RoadNetwork` | crop to GPS bbox + 1 km; reindex nodes; arc length and bearing; in/out degree; reverse-twin lookup; directed adjacency for bounded Dijkstra; KD-tree spatial index over arcs densified every 10 m |
| 6 | Candidates | `features.generate_candidates` | KD-tree radius query (50 m) → exact point-to-segment distance → keep the 12 nearest; fallback to the nearest arcs if none are within the radius |
| 7 | Features | `features.compute_features` | 28 features per (fix, candidate): distance and ranks, heading difference (±1 and ±5 fix windows), speed, arc geometry, projection position, topology (degree, reverse twin), context distances over ±3 fixes, connectivity to the neighbouring fixes' nearest arcs, network-distance vs GPS-displacement consistency (bounded Dijkstra), dt and step. No label information is involved |
| 8 | Labels (TRAIN only) | `labels.py`, `dataset.build_trip` | monotone dynamic-programming alignment of fixes to route positions (minimises total fix-to-arc distance; positions never go back); label joined **after** feature computation; `reliable` flag if the fix is within 30 m of its aligned reference arc |
| 9 | Dataset build | `dataset.build_partition`, `scripts/02_build_dataset.py` | per-trip parallel build → `cache/features_<partition>.parquet`, `cache/obs_<partition>.parquet`, `reports/candidate_stats_<partition>.csv` |

Partition separation: `train` and `student_test` are built separately, and their trip ids are disjoint (checked in
tests). The build script refuses any other partition. Metric projection, crop and candidate parameters are fixed
constants (`pathsnap/config.py`) chosen from TRAIN statistics. Preprocessing has no fitted parameters, so the same
code runs identically at training and inference time.

## 4. Input → output example (train trip 00000003)

Raw `.track` (first 3 rows, `reports/preprocessing_demo/sample_raw_track_00000003.csv`):
```
lon,lat,t
37.201398,56.007471,1.0
37.201322,56.007449,2.0
37.201328,56.007424,3.0
```
After cleaning, projection and kinematics (`sample_clean_track_00000003.csv`):
```
orig_idx,lon,lat,t,x,y,dt_prev,step_prev,speed,heading,heading_valid,speed_outlier,gap_prev
0,37.2014,56.0075,1.0,1408.0487,2647.3676,,,5.3389,242.6902,True,False,False
1,37.2013,56.0074,2.0,1403.3048,2644.9181,1.0,5.3389,3.4087,219.8602,True,False,False
2,37.2013,56.0074,3.0,1403.6793,2642.1346,1.0,2.8086,6.4097,115.7388,True,False,False
```
Candidates and features for fix 0 (selected columns of `sample_candidate_features_00000003.csv`; `y` = training label):
```
 obs  arc_gid    dist  dist_rank  heading_diff  cos_heading  ctx_dist_mean  trans_dev_next  y
   0     9853  1.1754          0       71.4461       0.3182         2.0529        500.0000  0
   0     9852  5.3395          1       91.5700      -0.0274         7.1840        500.0000  0
   0       12  7.1980          2       54.8560       0.5756         5.0569          2.2657  1
   0     5525  7.1980          3      125.1440      -0.5756         5.0569         61.1405  0
   ...  (12 candidates in total)
```
The reference arc (12) is only the **third nearest**. Its reverse twin 5525 has the same distance but the opposite
heading. Arc 12 is also the only candidate whose network distance to the next fix matches the GPS displacement
(`trans_dev_next` 2.3 m). The engineered features capture exactly this.

## 5. Before vs after (whole dataset)

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| raw GPS fixes | 208 417 | 19 160 |
| fixes dropped by cleaning (NaN / invalid / duplicate t) | 0 | 0 |
| speed-outlier fixes flagged (kept, handled downstream) | 149 | 1 |
| trips with a gap > 30 s | 28 (largest 153 064 s) | 2 (largest 210 s) |
| candidate rows (fix × arc) | 1 360 924 | 146 913 |
| candidates per fix | 6.53 | 7.67 |
| reference arc missing from candidates (all fixes) | 11.33 % | 0.21 % |
| ... restricted to reliable labels (≤ 30 m) | **0.18 %** | 0.04 % |
| fixes with reliable labels | 87.5 % | 99.2 % |
| full build time (16 cores) | 13.7 s | 1.5 s |

The train miss rate on *all* fixes is high because of a few trips whose reference route does not cover the drive
(00000054: the route covers ~14 % of a 169 km drive; 00000033: the route diverges from the GPS). These are
reference-label problems, not GPS problems. Unreliable fixes are excluded from fitting and cross-validation
(see `reports/EDA.md` §4).

Distance from each fix to its aligned reference arc:

| percentile | 50 | 75 | 90 | 95 | 99 |
|---|---|---|---|---|---|
| train | 5.7 m | 10.9 m | 89 m | 142 m | 206 m |
| student_test | 5.3 m | 9.0 m | 13.5 m | 16.9 m | 28.1 m |

## 6. Commands actually executed

```powershell
# environment (once)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# exploratory analysis of train + student_test (≈3 min): stats, figures, Folium map
.venv\Scripts\python scripts\01_explore.py

# full preprocessing / feature build for both partitions (≈15 s on 16 cores)
.venv\Scripts\python scripts\02_build_dataset.py
#   [train] trips=80 obs=208417 candidate rows=1360924 rows/obs=6.53 positives=184811 candidate miss rate (all)=0.1133 (reliable labels)=0.0018 reliable obs=0.8749 time=13.7s
#   [student_test] trips=10 obs=19160 candidate rows=146913 rows/obs=7.67 positives=19119 candidate miss rate (all)=0.0021 (reliable labels)=0.0004 reliable obs=0.9922 time=1.5s

# small representative sample (first 3 train trips; writes cache/sample_* only)
.venv\Scripts\python scripts\02_build_dataset.py --partitions train --limit 3
#   [train] trips=3 obs=9379 candidate rows=79963 rows/obs=8.53 positives=8624 candidate miss rate (all)=0.0805 (reliable labels)=0.0005 reliable obs=0.8918 time=3.2s

# single-trip review demo (≈1 s)
.venv\Scripts\python scripts\demo_preprocessing.py --data-root "Student folder\Student folder\train" --trip 00000003

# preprocessing tests
.venv\Scripts\python -m pytest tests\test_data_preprocessing.py tests\test_candidates_features.py -v
```

Demo output summary (trip 00000003):
```
trip 00000003: 821 raw fixes -> 821 clean fixes -> 9155 candidate rows x 28 features
network 18028 arcs -> 17890 after crop; candidate miss rate 0.24%
validation: 0 NaN, 0 invalid coords, 0 non-increasing t, 0 invalid node refs, 6 duplicate arcs, route connected 100 %
labels: median fix->reference distance 4.5 m, 100 % reliable, 819 positive rows
timings: load 12 ms, preprocess 7 ms, network 122 ms, candidates 31 ms, features 107 ms
```

## 7. How to reproduce the demo

1. Place the dataset at `Student folder\Student folder\` (or pass any path with `--data-root`).
2. `setup_windows.bat` (creates `.venv` and installs `requirements.txt`).
3. `.venv\Scripts\python scripts\demo_preprocessing.py --data-root "Student folder\Student folder\train" --trip 00000003`
   (any train trip id works; it defaults to the first trip).
4. Outputs land in `reports/preprocessing_demo/`:
   * `preprocessing_<trip>.png`: GPS vs reference route vs network; zoom showing candidate arcs and the labelled arc; candidates-per-fix histogram
   * `sample_raw_track_<trip>.csv`, `sample_clean_track_<trip>.csv`, `sample_candidate_features_<trip>.csv`: first 50 fixes, before and after
   * `summary_<trip>.json`: all statistics printed by the demo
   * `map_<trip>.html`: Folium map (GPS fixes, nearby road network, reference route) **without external map
     tiles**. OpenStreetMap's tile servers answer HTTP 403 for maps opened from disk, so the road network is drawn
     as the background instead. Set `PATHSNAP_MAP_TILES=OpenStreetMap` to add tiles when served over HTTP.

   Note (Windows): if a previous output (e.g. the PNG) is open in an image viewer or editor preview, Windows
   refuses to overwrite the memory-mapped file (`OSError: [Errno 22] Invalid argument` from `savefig`). The
   demo writes through a temp file and, in that case, saves `<name>_<timestamp>.png` and prints a note instead
   of crashing. This was reproduced deliberately and fixed on 2026-11-18.
5. Dataset-level visuals: `reports/figures/eda_*.png` and the Folium map `reports/maps/eda_trip_00000003.html`
   (GPS fixes, cropped road network, reference route, layer control). Open it in a browser.

No model training, Bayesian optimisation or GPU is needed for any of the above.

## 8. Completed vs pending

| item | status |
|---|---|
| Parsing of `.track/.nodes/.arcs/.route` with schema checks | ✅ done, tested |
| Coordinate / timestamp / network / route validation | ✅ done, tested |
| Invalid-point removal, time ordering, duplicate-timestamp removal | ✅ done, tested (the provided data contains none of these defects) |
| Geodesic handling (local metric projection, verified against haversine) | ✅ done, tested |
| Speed / heading computation, outlier and gap **flags** | ✅ done. Outliers are flagged, not removed; downstream models/decoder handle them |
| Network crop, spatial index, directed graph, bounded Dijkstra, shortest paths | ✅ done, tested |
| Candidate generation with radius + top-K + fallback | ✅ done, tested |
| 28-feature extraction, label-free | ✅ done, tested (a test asserts identical features with the `.route` file removed) |
| TRAIN label alignment + reliability flag | ✅ done, tested |
| Partition-separated dataset build (train / student_test) | ✅ done |
| GPS smoothing / Kalman filtering | ❌ not implemented (deliberate: raw fixes plus context features are passed to the HMM) |
| Repair of reference routes that do not cover the drive | ❌ not attempted; such fixes are flagged unreliable and excluded |
| Model training, Bayesian optimisation, HMM tuning, API | ✅ completed after the first review, see `reports/RESULTS.md` |
