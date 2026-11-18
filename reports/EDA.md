# PathSnap — Dataset Exploration (Phase 2)

All numbers below were produced by `scripts/01_explore.py` (raw values in `reports/eda_trip_stats.csv`
and `reports/eda_summary.json`). Partitions analysed: **train** (80 trips) and **student_test** (10 trips).
The evaluator-testing partition is not distributed to students and was never accessed.

## 1. Organisation and verified schemas

`<DATA_ROOT>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}` — every one of the 90 trips has all four files.

| File | Verified content |
|---|---|
| `.track` | 3 whitespace columns, no header: lon, lat, t (s, integer-valued floats starting at 1) |
| `.nodes` | 2 columns lon, lat; row index = node_id |
| `.arcs` | 2 int columns from_node, to_node; row index = arc_id; **directed** (a two-way street = two arcs) |
| `.route` | 1 int column; chronological reference arc_ids; consecutive arcs connected in 99.9 % of steps |

Key relationships discovered:
* **Each trip ships its own road network** (median 19.5 k arcs in train, up to 307 k). Arc ids are only meaningful within their trip.
* The `.route` does **not** label individual GPS fixes (e.g. trip 00000001: 1 315 fixes vs 525 route arcs).
  Per-fix labels must be derived → monotone DP alignment of fixes to route positions (`pathsnap/labels.py`).
* Trips are worldwide (lon −100 … 137, lat −35 … 62): the geodesy must be location independent.

## 2. Inventory

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS fixes total | 208 417 | 19 160 |
| fixes / trip (min / median / max) | 361 / 1 529 / 19 313 | 262 / 1 420 / 3 958 |
| median duration | 0.48 h | 0.5 h |
| median GPS path length | 24.0 km | 21.0 km |
| median reference-route length | 14.3 km (280 arcs) | 14.8 km (250 arcs) |
| nodes / trip (median, max) | 9 608 / 155 003 | 6 597 / 26 635 |
| arcs / trip (median, max) | 19 527 / 307 392 | 13 145 / 56 027 |
| raw size | 85.6 MB | 4.2 MB |

## 3. Data quality

| Check | train | student_test |
|---|---|---|
| NaN values / invalid coordinates / (0,0) fixes | 0 / 0 / 0 | 0 / 0 / 0 |
| non-increasing or duplicated timestamps | 0 | 0 |
| fraction of 1 s steps (mean over trips) | 93.8 % | 98.2 % |
| largest gap | 153 064 s (≈42 h, trip 00000090) | 210 s |
| trips with a gap > 30 s | 28 | 2 |
| speed outliers (> 252 km/h step) | 149 | 1 |
| isolated nodes | 0 | 0 |
| weakly connected components / trip (median) | 20 | 11 |
| smallest share of nodes in the largest weak component | 69 % | 87 % |
| share of arcs with a reverse twin (median) | 95 % | 93 % |
| invalid route arc ids | 0 | 0 |

Missing values: none in any file. The relevant "missingness" is temporal (GPS gaps; vehicle parked
for hours in two train trips) and spatial (GPS fixes off the network).

## 4. GPS vs reference route (label quality)

Distance of each fix to its aligned reference arc:

| percentile | 50 | 75 | 90 | 95 | 99 |
|---|---|---|---|---|---|
| train | 5.7 m | 10.9 m | 89 m | 142 m | 206 m |
| student_test | 5.3 m | 9.0 m | 13.5 m | 16.9 m | 28.1 m |

The heavy train tail comes from a few trips whose reference route does **not** cover the drive:
* **00000054** — the route covers ~14 % of the GPS path (86 % of fixes clamp to the last route arc; 169 km path vs 31 km route).
* **00000033** — the route diverges from the GPS path (median offset 122 m while the nearest road is 7 m away).
* **00000002, 00000066, 00000043** — partial divergence segments.

Decision (train partition only): a fix's label is considered **reliable if it is within 30 m** of its
aligned reference arc (the student-test 99th percentile is 28 m). 87.5 % of train fixes are reliable;
only these are used for fitting and cross-validation. This follows the guideline not to treat the
reference as error-free ground truth.

## 5. Decisions taken from the train EDA

| Decision | Evidence |
|---|---|
| Local equirectangular projection on the WGS84 ellipsoid around the trip median | extents ≤ ~40 km → distortion ≪ GPS noise; verified against haversine in tests |
| Candidate radius 50 m | 99.86 % of student-test fixes lie within 50 m of their reference arc |
| Keep ≤ 12 nearest candidates | reference arc within the 12 nearest arcs for 99.9 % of student-test fixes |
| Heading is a key feature | 95 % of streets are two-way → twin arcs with identical geometry; only direction separates them |
| HMM must handle breaks | 20 weak components/trip, gaps up to 42 h |
| Crop each network to trajectory bbox + 1 km | networks up to 307 k arcs; crop never removed a reference arc (0 cropped) |

## 6. Figures and maps

* `figures/eda_trip_sizes.png` — fixes, network size and path length per trip
* `figures/eda_sampling_speed.png` — sampling-interval and speed distributions (train)
* `figures/eda_label_distance_rank.png` — fix-to-reference distance and reference-arc rank among nearest arcs
* `figures/eda_trip_locations.png` — worldwide trip locations
* `figures/eda_trajectories.png` — six train trajectories over their road networks
* `figures/eda_network_connectivity.png` — network size and weak components per trip
* `maps/eda_trip_00000003.html` — Folium: GPS fixes, cropped road network, reference route (train trip)
