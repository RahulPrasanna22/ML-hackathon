"""Evaluation metrics for candidate classification and route matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


# ------------------------------------------------------------------ candidate level
def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "cand_accuracy": accuracy_score(y, yhat),
        "cand_precision": precision_score(y, yhat, zero_division=0),
        "cand_recall": recall_score(y, yhat, zero_division=0),
        "cand_f1": f1_score(y, yhat, zero_division=0),
        "cand_macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


# ------------------------------------------------------------------ observation level
def pick_top1(cands: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Per (trip, obs) row of the highest-scoring candidate (ties -> nearest)."""
    s = cands.sort_values(["trip_id", "obs", score_col, "dist"], ascending=[True, True, False, True], kind="stable")
    return s.drop_duplicates(["trip_id", "obs"], keep="first")


def topk_accuracy(cands: pd.DataFrame, k: int, score_col: str = "score", n_obs: int | None = None) -> float:
    r = cands.groupby(["trip_id", "obs"])[score_col].rank(ascending=False, method="first")
    hits = cands[(r <= k) & (cands["y"] == 1)][["trip_id", "obs"]].drop_duplicates()
    denom = n_obs if n_obs is not None else cands[["trip_id", "obs"]].drop_duplicates().shape[0]
    return len(hits) / denom


def continuity(pred_gid: np.ndarray, arcs: np.ndarray) -> tuple[float, int]:
    """Fraction of consecutive arc CHANGES that are directly connected (to == from)."""
    g = np.asarray(pred_gid)
    g = g[g >= 0]
    if len(g) < 2:
        return 1.0, 0
    a, b = g[:-1], g[1:]
    change = a != b
    if change.sum() == 0:
        return 1.0, 0
    ok = arcs[a[change], 1] == arcs[b[change], 0]
    return float(ok.mean()), int((~ok).sum())


def observation_metrics(cands: pd.DataFrame, arcs_by_trip: dict, score_col: str = "score") -> dict:
    """cands: candidate rows with trip_id, obs, arc_gid, y, dist, <score_col>."""
    n_obs = cands[["trip_id", "obs"]].drop_duplicates().shape[0]
    top = pick_top1(cands, score_col)
    top1 = float(top["y"].sum() / n_obs)
    cov = cands.groupby(["trip_id", "obs"])["y"].max()
    cont, inval = [], 0
    for tid, g in top.groupby("trip_id"):
        c, k = continuity(g.sort_values("obs")["arc_gid"].to_numpy(), arcs_by_trip[tid])
        cont.append(c); inval += k
    return {
        "top1_accuracy": top1,
        "top3_accuracy": topk_accuracy(cands, 3, score_col, n_obs),
        "candidate_miss_rate": float(1 - cov.mean()),
        "route_continuity": float(np.mean(cont)),
        "invalid_transitions": int(inval),
        "n_obs": int(n_obs),
    }


# ------------------------------------------------------------------ route level
def levenshtein(a, b) -> int:
    """Classic edit distance (insert / delete / substitute, unit costs)."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, yv in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != yv))
        prev = cur
    return int(prev[-1])


def compress(seq) -> list:
    out = []
    for s in seq:
        if s >= 0 and (not out or out[-1] != s):
            out.append(int(s))
    return out


def route_metrics(pred_route: list[int], ref_route: list[int], arc_len: dict[int, float]) -> dict:
    """Arc-sequence comparison between predicted and reference routes (global arc ids)."""
    pr, rr = compress(pred_route), compress(ref_route)
    ps, rs = set(pr), set(rr)
    L = lambda s: sum(arc_len.get(a, 0.0) for a in s)
    inter = ps & rs
    prec_len = L(inter) / max(L(ps), 1e-9)
    rec_len = L(inter) / max(L(rs), 1e-9)
    f1_len = 2 * prec_len * rec_len / max(prec_len + rec_len, 1e-9)
    ed = levenshtein(pr, rr)
    return {
        "route_precision_len": prec_len,
        "route_recall_len": rec_len,
        "route_f1_len": f1_len,
        "route_jaccard_arcs": len(inter) / max(len(ps | rs), 1),
        "edit_distance": ed,
        "norm_edit_distance": ed / max(len(pr), len(rr), 1),
        "pred_route_arcs": len(pr),
        "ref_route_arcs": len(rr),
    }
