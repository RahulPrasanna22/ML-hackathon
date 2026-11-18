import numpy as np
import pandas as pd

from pathsnap.features import FEATURE_COLUMNS, compute_features, generate_candidates
from pathsnap.network import RoadNetwork
from pathsnap.preprocess import preprocess_track

from conftest import grid_network, straight_track


def _setup(noise=0.0, y_m=3.0):
    nodes, arcs = grid_network()
    df, proj, _ = preprocess_track(straight_track(y_m=y_m, noise=noise))
    net = RoadNetwork.for_track(nodes, arcs, proj, df.x, df.y)
    return nodes, arcs, df, net


def eastbound_bottom_arcs(net):
    """Local arc ids along the bottom grid row (lat == LAT0) pointing east."""
    from conftest import LAT0
    f, t = net.node_lonlat[net.arc_from], net.node_lonlat[net.arc_to]
    return {i for i in range(net.n_arcs)
            if abs(f[i, 1] - LAT0) < 1e-9 and abs(t[i, 1] - LAT0) < 1e-9 and t[i, 0] > f[i, 0]}


def test_network_crop_index_and_paths():
    nodes, arcs, df, net = _setup()
    assert net.n_arcs <= len(arcs) and set(net.arc_gid) <= set(range(len(arcs)))
    a = next(iter(eastbound_bottom_arcs(net)))
    px, py = (net.ax[a] + net.bx[a]) / 2, net.ay[a] + 2.0     # 2 m north of the arc midpoint
    near = net.arcs_near(px, py, 10.0)
    d, _ = net.distance_to_arcs(px, py, near)
    assert len(near) >= 2 and np.isclose(d.min(), 2.0)  # both directions of the bottom road
    dist = net.dijkstra_from(int(net.arc_from[a]), 1000)
    assert np.isclose(dist[int(net.arc_to[a])], 100.0)
    path = net.shortest_arc_path(int(net.arc_from[a]), int(net.arc_to[a]))
    assert path == [a]


def test_candidates_cover_true_road_and_respect_radius():
    _, _, df, net = _setup()
    c = generate_candidates(df, net, radius=50.0, max_k=12)
    assert (c.dist <= 50.0).all() and c.groupby("obs").size().max() <= 12
    east = eastbound_bottom_arcs(net)
    assert c.groupby("obs").apply(lambda g: bool(set(g.arc) & east), include_groups=False).all()


def test_candidate_fallback_when_nothing_in_radius():
    nodes, arcs = grid_network()
    df, proj, _ = preprocess_track(straight_track(y_m=-300.0))  # 300 m south of the network
    net = RoadNetwork.for_track(nodes, arcs, proj, df.x, df.y, buffer_m=2000)
    c = generate_candidates(df, net, radius=50.0)
    assert set(c.obs) == set(range(len(df)))  # every fix still gets >= 1 candidate


def test_features_complete_and_label_free():
    _, _, df, net = _setup(noise=2.0)
    f = compute_features(df, net)
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert not f[FEATURE_COLUMNS].isna().any().any()
    assert "y" not in f.columns and "label_dist" not in f.columns
    east = eastbound_bottom_arcs(net)
    # direction feature must separate the eastbound arc from its westbound twin
    hd_e = f[f.arc.isin(east) & (f.dist < 10)].heading_diff.mean()
    west = {int(net.reverse_arc[a]) for a in east}
    hd_w = f[f.arc.isin(west) & (f.dist < 10)].heading_diff.mean()
    assert hd_e < 20 < 160 < hd_w


# ---------------------------------------------------------------- real-data integrity checks
import pytest  # noqa: E402

from pathsnap import config, io  # noqa: E402

_TRAIN = io.list_trips(config.TRAIN_PARTITION)


@pytest.mark.skipif(not _TRAIN, reason="dataset not present")
def test_partitions_disjoint():
    assert set(_TRAIN).isdisjoint(io.list_trips(config.STUDENT_TEST_PARTITION))


@pytest.mark.skipif(not _TRAIN, reason="dataset not present")
def test_features_identical_without_route_file(tmp_path, monkeypatch):
    """Leakage guard: removing the .route file must not change a single feature value."""
    import shutil
    tid = _TRAIN[2]

    def feats_for(partition):
        trip = io.load_trip(partition, tid)
        df, proj, _ = preprocess_track(trip.track)
        return compute_features(df, RoadNetwork.for_track(trip.nodes, trip.arcs, proj, df.x, df.y))

    with_route = feats_for(config.TRAIN_PARTITION)
    dst = tmp_path / "noroute" / tid
    dst.mkdir(parents=True)
    for ext in ["track", "nodes", "arcs"]:
        shutil.copy(io.partition_dir(config.TRAIN_PARTITION) / tid / f"{tid}.{ext}", dst / f"{tid}.{ext}")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    without = feats_for("noroute")
    assert len(with_route) == len(without)
    assert np.array_equal(with_route[FEATURE_COLUMNS].to_numpy(float), without[FEATURE_COLUMNS].to_numpy(float))


@pytest.mark.skipif(not _TRAIN, reason="dataset not present")
def test_sample_dataset_build_with_labels():
    from pathsnap.dataset import build_trip
    feats, obs, meta = build_trip(config.TRAIN_PARTITION, _TRAIN[2])
    assert meta["n_obs"] == len(obs) and set(feats.y.unique()) <= {0, 1}
    assert (feats.groupby("obs").y.sum() <= 1).all()          # at most one positive candidate per fix
    assert meta["miss_rate_reliable"] < 0.02
