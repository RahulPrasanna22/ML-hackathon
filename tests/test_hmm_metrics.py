import numpy as np
import pandas as pd

from pathsnap.features import FEATURE_COLUMNS, compute_features, generate_candidates
from pathsnap.hmm import HMMParams, reconstruct_route, viterbi
from pathsnap.metrics import compress, continuity, levenshtein, route_metrics
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


def test_viterbi_recovers_connected_route_despite_noisy_scores():
    _, _, df, net = _setup(noise=3.0)
    f = compute_features(df, net)
    east = eastbound_bottom_arcs(net)
    rng = np.random.default_rng(0)
    # noisy "classifier": true arcs get 0.6 on average, others 0.4; 30 % of fixes are flipped
    p = np.where(f.arc.isin(east), 0.6, 0.4) + rng.normal(0, 0.05, len(f))
    flip = rng.random(df.shape[0]) < 0.3
    p = np.where(flip[f.obs] & f.arc.isin(east), 0.2, p)
    f["prob"] = np.clip(p, 0.01, 0.99)
    top = f.sort_values(["obs", "prob"], ascending=[True, False]).drop_duplicates("obs")
    clf_acc = top.arc.isin(east).mean()
    dec = viterbi(df, net, f, "prob", HMMParams(beta=5.0))
    hmm_acc = np.mean([a in east for a in dec["arc_local"]])
    assert hmm_acc > clf_acc and hmm_acc > 0.95
    route, unfilled = reconstruct_route(net, dec["arc_local"])
    assert unfilled == 0
    assert all(net.arc_to[a] == net.arc_from[b] for a, b in zip(route[:-1], route[1:]))


def test_viterbi_breaks_on_impossible_transition_and_gap():
    _, _, df, net = _setup()
    df = df.copy()
    f = compute_features(df, net)
    f["prob"] = 0.5
    df.loc[df.index[len(df) // 2]:, "t"] += 10_000  # huge GPS gap
    dec = viterbi(df, net, f, "prob", HMMParams(gap_break_s=600))
    assert dec["n_breaks"] >= 1 and (dec["arc_local"] >= 0).all()


def test_route_reconstruction_fills_gaps():
    _, _, df, net = _setup()
    east = sorted(eastbound_bottom_arcs(net), key=lambda a: net.ax[a])
    route, unfilled = reconstruct_route(net, np.array([east[0], east[0], east[2]]))
    assert unfilled == 0 and route == [east[0], east[1], east[2]]


def test_metrics():
    assert levenshtein([1, 2, 3], [1, 3]) == 1 and levenshtein([], [1, 2]) == 2 and levenshtein([1, 2], [2, 1]) == 2
    assert compress([1, 1, -1, 2, 2, 1]) == [1, 2, 1]
    arcs = np.array([[0, 1], [1, 2], [5, 6]])
    c, bad = continuity(np.array([0, 0, 1, 1, 2]), arcs)
    assert c == 0.5 and bad == 1
    rm = route_metrics([0, 1], [0, 1, 2], {0: 10.0, 1: 10.0, 2: 20.0})
    assert rm["route_precision_len"] == 1.0 and np.isclose(rm["route_recall_len"], 0.5) and rm["edit_distance"] == 1
