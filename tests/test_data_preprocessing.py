import numpy as np
import pandas as pd
import pytest

from pathsnap import config, io
from pathsnap.geo import LocalProjection, angle_diff, bearing_xy, haversine_m, point_segment_distance
from pathsnap.labels import align_track_to_route
from pathsnap.preprocess import clean_track, preprocess_track

from conftest import straight_track

REAL = io.list_trips(config.STUDENT_TEST_PARTITION)


# ---------------------------------------------------------------- parsing (real files)
@pytest.mark.skipif(not REAL, reason="dataset not present")
def test_parse_real_trip_schema():
    tid = REAL[0]
    trip = io.load_trip(config.STUDENT_TEST_PARTITION, tid)
    assert list(trip.track.columns) == ["lon", "lat", "t"]
    assert trip.nodes.shape[1] == 2 and trip.arcs.shape[1] == 2
    assert trip.arcs.dtype == np.int64
    assert trip.arcs.max() < len(trip.nodes) and trip.arcs.min() >= 0
    route = io.load_reference_route(config.STUDENT_TEST_PARTITION, tid)
    assert route.ndim == 1 and route.max() < len(trip.arcs)
    rep = io.validate_trip(trip, route)
    assert rep["track_nan"] == 0 and rep["arcs_invalid_node_ref"] == 0
    assert rep["route_connected_frac"] > 0.95


def test_load_trip_never_reads_route():
    import inspect
    assert ".route" not in inspect.getsource(io.load_trip) and "route" not in io.Trip.__dataclass_fields__


def test_validate_flags_problems(grid):
    nodes, arcs = grid
    tr = pd.DataFrame({"lon": [8.0, np.nan, 200.0, 8.0], "lat": [50.0, 50.0, 50.0, 50.0], "t": [0, 1, 1, 0.5]})
    rep = io.validate_trip(io.Trip("x", "p", tr, nodes, np.r_[arcs, [[0, 999]]]), route=np.array([0, 10_000]))
    assert rep["track_nan"] == 1 and rep["track_coord_out_of_range"] == 1
    assert rep["t_non_increasing"] >= 1 and rep["arcs_invalid_node_ref"] == 1 and rep["route_invalid_ids"] == 1


# ---------------------------------------------------------------- geodesy
def test_projection_matches_haversine():
    p = LocalProjection(8.6, 49.9)
    lon2, lat2 = 8.65, 49.95
    x, y = p.forward(lon2, lat2)
    assert abs(np.hypot(x, y) - haversine_m(8.6, 49.9, lon2, lat2)) / haversine_m(8.6, 49.9, lon2, lat2) < 0.005
    lon_b, lat_b = p.inverse(x, y)
    assert np.isclose(lon_b, lon2) and np.isclose(lat_b, lat2)


def test_point_segment_and_angles():
    d, t = point_segment_distance(np.array([5.0, -3.0, 15.0]), np.array([4.0, 0.0, 0.0]), 0.0, 0.0, 10.0, 0.0)
    assert np.allclose(d, [4, 3, 5]) and np.allclose(t, [0.5, 0, 1])
    assert bearing_xy(0, 1) == 0 and bearing_xy(1, 0) == 90
    assert angle_diff(350, 10) == 20 and angle_diff(0, 180) == 180


# ---------------------------------------------------------------- preprocessing
def test_clean_track_handles_duplicates_order_and_invalid():
    tr = pd.DataFrame({"lon": [8.0, 8.001, np.nan, 8.002, 8.003, 0.0],
                       "lat": [50.0, 50.0, 50.0, 50.0, 50.0, 0.0],
                       "t": [0.0, 2.0, 3.0, 1.0, 2.0, 5.0]})
    df, info = clean_track(tr)
    assert info["n_invalid_dropped"] == 2 and info["n_duplicate_t_dropped"] == 1
    assert df.t.is_monotonic_increasing and df.t.is_unique
    assert list(df.orig_idx) == [0, 3, 1]


def test_preprocess_speed_heading():
    tr = straight_track(speed=10.0)
    df, proj, info = preprocess_track(tr)
    assert np.allclose(df.speed.iloc[1:-1], 10.0, atol=0.05)
    assert np.allclose(df.heading.dropna(), 90.0, atol=1.0)  # driving east
    assert info["n_speed_outliers"] == 0


def test_label_alignment_is_monotone():
    # route: two consecutive horizontal segments; fixes move left -> right
    ax, ay, bx, by = np.array([0., 100.]), np.zeros(2), np.array([100., 200.]), np.zeros(2)
    x = np.linspace(5, 195, 20); y = np.full(20, 4.0)
    pos, dist = align_track_to_route(x, y, ax, ay, bx, by)
    assert (np.diff(pos) >= 0).all() and pos[0] == 0 and pos[-1] == 1
    assert np.allclose(dist, 4.0)
