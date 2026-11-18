"""HMM map matching with Viterbi decoding.

Hidden state at time i : one candidate road arc c (plus its projection position on the arc)
                         of GPS fix i. The candidate <-> arc correspondence is kept in
                         `cands` (obs, arc, proj_t, prob).
Emission               : log P(fix i | arc c)  =  w_e * log(p_ic / sum_c' p_ic')
                         where p is the trained classifier's probability that c is the true
                         road (classifier output used consistently as the emission score),
                         optionally blended with a Gaussian GPS-noise term -0.5 (d/sigma)^2.
Transition             : log P(c_i -> c_{i+1}) = -| d_route - d_gc | / beta
                         d_route = directed network distance between the two projected
                         positions (bounded Dijkstra; same-arc forward moves are direct),
                         d_gc    = straight-line distance between the two fixes.
                         Unreachable (disconnected / wrong direction) pairs get -inf.
                         Heading consistency: emissions already contain the classifier's
                         heading features; additionally a switch onto the reverse twin of the
                         current arc costs `reverse_penalty` (U-turn in place).
Viterbi recurrence     : V_{i+1}(b) = E_{i+1}(b) + max_a [ V_i(a) + T(a, b) ]
                         with back-pointers; the best final state is backtracked.
Breaks                 : if no finite transition exists (disconnected network pieces,
                         missing candidates, very long GPS gaps), the chain is restarted:
                         V_{i+1}(b) = E_{i+1}(b) + max_a V_i(a) - break_penalty.
GPS gaps               : the Dijkstra search limit grows with the time gap (vmax * dt);
                         gaps longer than `gap_break_s` force a break.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMParams:
    beta: float = 10.0               # transition scale (m)
    emission_weight: float = 1.0     # weight on log classifier probability
    sigma: float = 0.0               # >0 adds a Gaussian distance emission term (m)
    prob_floor: float = 1e-6
    same_arc_tol: float = 15.0       # tolerated backward jitter on the same arc (m)
    vmax: float = 50.0               # m/s used for the search limit
    base_limit: float = 300.0        # minimum Dijkstra search radius (m)
    max_limit: float = 6000.0
    gap_break_s: float = 600.0
    break_penalty: float = 50.0
    reverse_penalty: float = 0.0     # heading consistency: log-penalty for jumping onto the reverse twin arc (U-turn)

    def to_dict(self):
        return asdict(self)


def _transition_matrix(net: RoadNetwork, a_arcs, a_pos, b_arcs, b_pos, gc: float, limit: float, p: HMMParams):
    T = np.full((len(a_arcs), len(b_arcs)), -np.inf)
    from_b = net.arc_from[b_arcs]
    for ia, (a, pa) in enumerate(zip(a_arcs, a_pos)):
        rem = net.length[a] - pa
        dist = None
        for ib, (b, pb) in enumerate(zip(b_arcs, b_pos)):
            if a == b and pb >= pa - p.same_arc_tol:
                d = max(pb - pa, 0.0)
            else:
                if dist is None:
                    dist = net.dijkstra_from(int(net.arc_to[a]), limit)
                dn = dist.get(int(from_b[ib]))
                if dn is None:
                    continue
                d = rem + dn + pb
                if d > limit + net.length[a]:
                    continue
            T[ia, ib] = -abs(d - gc) / p.beta
            if p.reverse_penalty and b == net.reverse_arc[a]:
                T[ia, ib] -= p.reverse_penalty
    return T


def viterbi(df: pd.DataFrame, net: RoadNetwork, cands: pd.DataFrame, prob_col: str, params: HMMParams) -> dict:
    """Decode the most likely candidate sequence.

    Returns dict(arc_local (n,), n_breaks, state_index (n,)) where arc_local[i] = -1 if fix i
    had no candidates.
    """
    n = len(df)
    c = cands.sort_values(["obs", "dist_rank"], kind="stable")
    obs = c["obs"].to_numpy()
    starts = np.searchsorted(obs, np.arange(n))
    ends = np.searchsorted(obs, np.arange(n), side="right")
    arcs_all = c["arc"].to_numpy()
    pos_all = (c["proj_t"].to_numpy() * net.length[arcs_all])
    prob = np.clip(c[prob_col].to_numpy(), params.prob_floor, 1.0)
    dist_all = c["dist"].to_numpy()
    xs, ys, ts = df["x"].to_numpy(), df["y"].to_numpy(), df["t"].to_numpy()

    def emission(i):
        s, e = starts[i], ends[i]
        pr = prob[s:e]
        em = params.emission_weight * np.log(pr / pr.sum())
        if params.sigma > 0:
            em = em - 0.5 * (dist_all[s:e] / params.sigma) ** 2
        return em

    back: list[np.ndarray | None] = [None] * n
    V = None
    prev_i = None
    n_breaks = 0
    for i in range(n):
        s, e = starts[i], ends[i]
        if e == s:  # no candidates for this fix: skip it, keep chain
            back[i] = None
            continue
        em = emission(i)
        if V is None:
            V = em.copy(); back[i] = np.full(e - s, -1); prev_i = i
            continue
        ps, pe = starts[prev_i], ends[prev_i]
        dt = ts[i] - ts[prev_i]
        gc = float(np.hypot(xs[i] - xs[prev_i], ys[i] - ys[prev_i]))
        if dt > params.gap_break_s:
            T = np.full((pe - ps, e - s), -np.inf)
        else:
            limit = min(max(params.base_limit, 2.0 * gc + 100.0, params.vmax * max(dt, 1.0)), params.max_limit)
            T = _transition_matrix(net, arcs_all[ps:pe], pos_all[ps:pe], arcs_all[s:e], pos_all[s:e], gc, limit, params)
        scores = V[:, None] + T
        best_prev = np.argmax(scores, axis=0)
        best = scores[best_prev, np.arange(e - s)]
        if not np.isfinite(best).any():
            n_breaks += 1
            a = int(np.argmax(V))
            best_prev = np.full(e - s, a)
            best = V[a] - params.break_penalty + np.zeros(e - s)
        else:
            # states unreachable from every predecessor fall back to a penalised break
            bad = ~np.isfinite(best)
            if bad.any():
                a = int(np.argmax(V))
                best_prev[bad] = a
                best[bad] = V[a] - params.break_penalty
        V = best + em
        V -= V.max()  # numerical stability
        back[i] = best_prev
        prev_i = i

    # backtrack
    arc_local = np.full(n, -1, dtype=np.int64)
    state = np.full(n, -1, dtype=np.int64)
    if V is None:
        return {"arc_local": arc_local, "state": state, "n_breaks": 0}
    k = int(np.argmax(V))
    i = prev_i
    while i is not None and i >= 0:
        s = starts[i]
        state[i] = k
        arc_local[i] = arcs_all[s + k]
        bp = back[i]
        k = int(bp[k])
        # move to previous fix that had candidates
        j = i - 1
        while j >= 0 and ends[j] == starts[j]:
            j -= 1
        if k < 0 or j < 0:
            break
        i = j
    return {"arc_local": arc_local, "state": state, "n_breaks": n_breaks}


def reconstruct_route(net: RoadNetwork, arc_seq_local: np.ndarray, max_fill_m: float = 3000.0) -> tuple[list[int], int]:
    """Collapse the per-fix arc sequence and fill gaps between non-adjacent consecutive arcs
    with the directed shortest path. Returns (local arc route, number of unfillable gaps)."""
    seq = [int(a) for a in arc_seq_local if a >= 0]
    route: list[int] = []
    unfilled = 0
    for a in seq:
        if not route:
            route.append(a); continue
        prev = route[-1]
        if a == prev:
            continue
        if net.arc_to[prev] != net.arc_from[a]:
            path = net.shortest_arc_path(int(net.arc_to[prev]), int(net.arc_from[a]), limit=max_fill_m)
            if path is None:
                unfilled += 1
            else:
                route.extend(path)
        route.append(a)
    return route, unfilled
