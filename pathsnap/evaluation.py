"""Server-side evaluation of a finished prediction against the reference route.

Called only AFTER `MapMatcher.match_trip` has produced its output; the reference route is
loaded here and never passed into the prediction pipeline. Results contain metrics only;
reference geometry is not returned to clients by the API.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, precision_recall_fscore_support

from .geo import point_segment_distance
from .labels import observation_labels
from .metrics import compress, continuity, route_metrics
from .pipeline import matched_points


def _spatial_deviation(net, df, arc_local, ref_local):
    """Distance (m) from each matched point to the nearest reference-route arc."""
    ok = arc_local >= 0
    if not ok.any() or len(ref_local) == 0:
        return np.array([np.nan])
    a = arc_local[ok]
    _, t = point_segment_distance(df.x.to_numpy()[ok], df.y.to_numpy()[ok], net.ax[a], net.ay[a], net.bx[a], net.by[a])
    qx = net.ax[a] + t * (net.bx[a] - net.ax[a]); qy = net.ay[a] + t * (net.by[a] - net.ay[a])
    r = np.unique(ref_local)
    best = np.full(len(qx), np.inf)
    for chunk in np.array_split(r, max(1, len(r) // 256)):
        d, _ = point_segment_distance(qx[:, None], qy[:, None], net.ax[chunk][None], net.ay[chunk][None],
                                      net.bx[chunk][None], net.by[chunk][None])
        best = np.minimum(best, d.min(axis=1))
    return best


def evaluate_prediction(result: dict, ref_route_gid: np.ndarray, reliable_m: float | None = None) -> dict:
    """Metrics for nearest-road (A), classifier-only (B) and classifier+HMM (C)."""
    net, df, feats = result["net"], result["df"], result["feats"]
    lab_local, _, lab_dist = observation_labels(df, net, ref_route_gid)
    mask = np.ones(len(df), bool) if reliable_m is None else lab_dist <= reliable_m
    arcs_global = np.c_[net.node_gid[net.arc_from], net.node_gid[net.arc_to]]  # local arc -> global node ids
    nearest = feats[feats.dist_rank == 0]
    near_arc = np.full(len(df), -1); near_arc[nearest.obs.to_numpy()] = nearest.arc.to_numpy()
    ref_local = np.array([net.gid_to_local.get(int(g), -1) for g in ref_route_gid])
    ref_local = ref_local[ref_local >= 0]
    arc_len = {int(net.arc_gid[i]): float(net.length[i]) for i in np.unique(ref_local)}
    hit = feats.arc.to_numpy() == lab_local[feats.obs.to_numpy()]
    covered = np.zeros(len(df), bool)
    covered[feats.obs.to_numpy()[hit]] = True
    f_obs = feats.obs.to_numpy()
    f_arc = feats.arc.to_numpy()
    f_mask = mask[f_obs]
    y_true = hit[f_mask].astype(int)
    if "prob" in feats and f_mask.any():
        clf_log_loss = float(log_loss(y_true, np.clip(feats.prob.to_numpy()[f_mask], 1e-7, 1 - 1e-7), labels=[0, 1]))
    else:
        clf_log_loss = None

    def block(arc_local, route_local=None, with_log_loss=True):
        acc = float(np.mean(arc_local[mask] == lab_local[mask])) if mask.any() else float("nan")
        # candidate-level view: the method "selects" one candidate per fix (positive) and rejects the rest
        y_pred = (f_arc == arc_local[f_obs])[f_mask].astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0, labels=[0, 1])
        # same physical road regardless of direction
        rev_ok = (arc_local == lab_local) | ((arc_local >= 0) & (net.reverse_arc[np.maximum(arc_local, 0)] == lab_local))
        acc_undirected = float(np.mean(rev_ok[mask])) if mask.any() else float("nan")
        cont, inval = continuity(arc_local, arcs_global)
        # reachable continuity: an arc change is plausible if the new arc can be reached
        # from the previous one through the directed network within 500 m
        seq_c = compress(arc_local)
        reach = [net.arc_to[a] == net.arc_from[b] or
                 net.shortest_arc_path(int(net.arc_to[a]), int(net.arc_from[b]), limit=500.0) is not None
                 for a, b in zip(seq_c[:-1], seq_c[1:])]
        reach_cont = float(np.mean(reach)) if reach else 1.0
        seq = route_local if route_local is not None else compress(arc_local)
        lens = dict(arc_len)
        lens.update({int(net.arc_gid[i]): float(net.length[i]) for i in seq})
        rm = route_metrics([int(net.arc_gid[i]) for i in seq], [int(g) for g in ref_route_gid], lens)
        dev = _spatial_deviation(net, df, arc_local, ref_local)
        return {"accuracy": acc, "macro_f1": float(f1), "macro_precision": float(p), "macro_recall": float(r),
                "log_loss": clf_log_loss if with_log_loss else None,
                "sequence_accuracy": float(1.0 - rm["norm_edit_distance"]),
                "accuracy_undirected": acc_undirected, "continuity": cont,
                "reachable_continuity": reach_cont, "unreachable_transitions": int(len(reach) - sum(reach)),
                "invalid_transitions": inval, **rm,
                "mean_deviation_m": float(np.nanmean(dev)), "p95_deviation_m": float(np.nanpercentile(dev, 95))}

    out = {"n_obs": int(len(df)), "n_obs_evaluated": int(mask.sum()),
           "candidate_miss_rate": float(1 - covered[mask].mean()) if mask.any() else float("nan"),
           "nearest_road": block(near_arc, with_log_loss=False),  # no probabilities -> N/A
           "classifier_only": block(result["clf_arc_local"])}
    if "hmm_arc_local" in result:
        out["classifier_hmm"] = block(result["hmm_arc_local"], result["route_local"])
        out["classifier_hmm"]["route_unfilled_gaps"] = int(result["route_unfilled_gaps"])
        out["classifier_hmm"]["n_breaks"] = int(result["n_breaks"])
    out["_labels_local"] = lab_local  # internal use (maps); stripped by the API
    return out
